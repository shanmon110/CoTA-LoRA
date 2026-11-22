"""LowRA offline decoder quantization, ICML 2025 §§5–6 and Appendices D–G.

The public Stanford repository contains only a README. This CPU preprocessor
learns two Lloyd updates per channel, assigns real 1/2/4-bit codes globally
within equal-width groups, and initializes the adapter from the quantization error.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from numba import get_num_threads, njit, prange, set_num_threads
from safetensors import safe_open
from safetensors.torch import save_file


BITS = (1, 2, 4)
NS = np.array([2, 4, 16])
NF = np.zeros((3, 16), dtype=np.float32)
NF[0, :2] = [-1, 1]
NF[1, :4] = [-1, 0, .3379, 1]
NF[2] = [-1, -.6961928, -.5250731, -.3949175, -.28444138, -.18477343, -.09105,
         0, .0795803, .1609302, .2461123, .33791524, .44070983, .562617, .72295684, 1]


@contextmanager
def _cpu_threads(rows):
    # libgomp is shared with PyTorch. Numba otherwise leaves torch using the
    # host-wide core count after a calibration, even in a smaller CPU allocation.
    previous_torch = torch.get_num_threads()
    previous_numba = get_num_threads()
    set_num_threads(min(previous_torch,previous_numba,len(os.sched_getaffinity(0)),max(1,rows//32)))
    try:
        yield
    finally:
        set_num_threads(previous_numba)
        torch.set_num_threads(previous_torch)


def fit_books(weight):
    with _cpu_threads(len(weight)):
        return _fit_books(weight)


@njit(parallel=True, cache=True)
def _fit_books(weight):
    rows, width = weight.shape
    maps = np.zeros((rows, 3, 16), dtype=np.float32)
    edges = np.zeros((rows, 3, 15), dtype=np.float32)
    sse = np.empty((rows, 3), dtype=np.float64)
    for row in prange(rows):
        w = weight[row]
        amp = np.empty(width, dtype=np.float32)
        z = np.zeros(width, dtype=np.float32)
        for j in range(0, width, 64):
            s = np.max(np.abs(w[j:j+64]))
            amp[j:j+64] = s
            if s > 0:
                z[j:j+64] = w[j:j+64] / s
        order = np.argsort(z)
        sorted_z = z[order]
        # Prefix sums over sorted normalized weights give exact Lloyd assignments.
        ps = np.zeros((width + 1, 5), dtype=np.float64)
        for j in range(width):
            i = order[j]
            s = np.float64(amp[i])
            v = np.float64(w[i])
            ps[j+1,0] = ps[j,0] + s
            ps[j+1,1] = ps[j,1] + v
            ps[j+1,2] = ps[j,2] + s*s
            ps[j+1,3] = ps[j,3] + s*v
            ps[j+1,4] = ps[j,4] + v*v
        for k in range(3):
            n = NS[k]
            book = NF[k, :n].copy()
            cut = (book[:-1] + book[1:]) / 2
            if k == 1:
                cut[:] = np.array([-.5, .16895762, .66895762], dtype=np.float32)
            accumulated = cut * 0
            for _ in range(2):
                begin = 0
                for b in range(n):
                    end = width if b == n - 1 else np.searchsorted(sorted_z, cut[b], side='right')
                    denom = ps[end,0] - ps[begin,0]
                    if denom > 0:
                        book[b] = (ps[end,1] - ps[begin,1]) / denom
                    begin = end
                cut = (book[:-1] + book[1:]) / 2
                accumulated += cut
            # Paper specifies averaging thresholds; here we average the two learned iterations.
            cut = accumulated * .5
            begin = 0
            error = 0.
            for b in range(n):
                end = width if b == n - 1 else np.searchsorted(sorted_z, cut[b], side='right')
                center = np.float64(book[b])
                sums = ps[end] - ps[begin]
                error += sums[4] - 2*center*sums[3] + center*center*sums[2]
                begin = end
            maps[row,k,:n] = book
            edges[row,k,:n-1] = cut
            sse[row,k] = max(error, 0.)
    return maps, edges, sse


def assign_bits(cost, bpp, clusters, seed):
    """Cluster-level integer assignment followed by a min-cost transportation LP.

    The latter has an integral polytope, so a basic optimal solution is integer.
    """
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix, csr_matrix, eye, kron, vstack
    from sklearn.cluster import KMeans

    n = len(cost)
    # Normalize all three dimensions by the SAME constant to preserve SSE weighting.
    cost = np.asarray(cost, dtype=np.float64)
    factor = max(cost.max(), 1e-20)
    X = cost / factor
    group = KMeans(n_clusters=min(clusters, n), random_state=seed, n_init=1).fit_predict(X)
    _, group = np.unique(group, return_inverse=True)
    sizes = np.bincount(group)
    nclusters = len(sizes)
    avg = np.stack([np.bincount(group, weights=X[:, j])/sizes for j in range(3)], axis=1)
    constraints = vstack((kron(eye(nclusters), [[1., 1., 1.]]), csr_matrix(np.tile(BITS, nclusters)[None])))
    budget = math.floor(bpp*n)
    quotas = milp(avg.ravel(), integrality=np.ones(avg.size), bounds=Bounds(0, np.repeat(sizes, 3)),
                  constraints=LinearConstraint(constraints, np.r_[sizes, -np.inf], np.r_[sizes, budget]),
                  options={'mip_rel_gap':0.0})
    if not quotas.success:
        raise RuntimeError(f'LowRA cluster assignment failed: {quotas.message}')
    counts = np.rint(quotas.x).astype(int).reshape(-1,3)
    out = np.empty(n, dtype=np.uint8)
    for c, quota in enumerate(counts):
        ids = np.flatnonzero(group == c)
        active = np.flatnonzero(quota)
        if len(active) == 1:
            out[ids] = BITS[active[0]]
        elif len(active) == 2:
            # Exact two-option transportation by sorting marginal costs.
            a, b = active
            order = np.argsort(cost[ids, a] - cost[ids, b], kind='stable')
            out[ids] = BITS[b]
            out[ids[order[:quota[a]]]] = BITS[a]
        else:
            # Each variable belongs to one row and one quota constraint.
            m = len(ids)
            ii = np.r_[np.repeat(np.arange(m), 3), m + np.tile(np.arange(3), m)]
            jj = np.tile(np.arange(m*3), 2)
            constraints = coo_matrix((np.ones(len(ii)), (ii,jj)), shape=(m+3, m*3)).tocsr()
            result = linprog(X[ids].ravel(), A_eq=constraints, b_eq=np.r_[np.ones(m),quota],
                             bounds=(0,1), method='highs-ds')
            if not result.success:
                raise RuntimeError(f'LowRA row assignment failed: {result.message}')
            sol = result.x.reshape(m,3)
            if np.max(np.abs(sol - np.rint(sol))) > 1e-5:
                raise RuntimeError('LowRA transportation solver returned non-integral assignment')
            out[ids] = np.asarray(BITS)[sol.argmax(1)]
        if not np.array_equal([np.sum(out[ids]==k) for k in BITS], quota):
            raise RuntimeError('Row bit count does not match cluster quota')
    if out.sum() > budget:
        raise RuntimeError('LowRA assignment exceeded bit budget')
    return out


def pack_weight(weight, bits, maps, edges):
    with _cpu_threads(len(weight)):
        return _pack_weight(weight,bits,maps,edges)


@njit(parallel=True, cache=True)
def _pack_weight(weight, bits, maps, edges):
    rows, width = weight.shape
    offsets = np.zeros(rows+1, dtype=np.int32)
    for r in range(rows):
        offsets[r+1] = offsets[r] + width * int(bits[r]) // 8
    packed = np.zeros(offsets[-1], dtype=np.uint8)
    book = np.zeros((rows,16), dtype=np.float32)
    scales = np.zeros((rows,width//64), dtype=np.float32)
    dequant = np.zeros_like(weight)
    for row in prange(rows):
        bit = int(bits[row])
        k = 0 if bit == 1 else (1 if bit == 2 else 2)
        centers, cut = maps[row,k], edges[row,k,:(1<<bit)-1]
        book[row] = centers
        per_byte = 8 // bit
        for j in range(0, width, 64):
            s = np.max(np.abs(weight[row,j:j+64]))
            scales[row,j//64] = s
            for col in range(j,j+64):
                z = weight[row,col]/s if s else np.float32(0)
                code = np.searchsorted(cut, z, side='left')
                packed[offsets[row] + col//per_byte] |= code << ((col%per_byte)*bit)
                dequant[row,col] = centers[code] * s
    return packed, offsets, scales, book, dequant


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest_base(model, verified, archive):
    expected = verified['files'][archive]['sha256']
    if file_hash(model/archive) != expected:
        raise ValueError(f'Source shard differs from pinned base: {archive}')


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    tmp.replace(path)


def prepare(args):
    model, root = args.model, args.output
    if not 1 <= args.bpp < 2 or args.rank < 1 or args.alpha <= 0 or args.clusters < 1:
        raise ValueError('Invalid LowRA descriptor')
    verified = json.loads(args.snapshot.read_text())
    digest_base(model, verified, 'model.safetensors.index.json')
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    names = sorted(k for k in index if k.startswith('thinker.model.layers.') and k.endswith('_proj.weight'))
    if not names:
        raise ValueError('Missing Thinker projection weights')
    cfg = dict(base=verified['revision'], bpp=args.bpp, rank=args.rank, alpha=args.alpha,
               seed=args.seed, clusters=args.clusters, version=1, names=names)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text()) != cfg:
        raise ValueError('Existing LowRA directory was prepared for a different protocol')
    write_json(root/'protocol.json',cfg)
    if (root/'index.json').exists():
        finished = json.loads((root/'index.json').read_text())
        if finished['protocol'] != cfg or set(finished['projections']) != {n[:-7] for n in names}:
            raise ValueError('Incomplete or incompatible LowRA artifact index')
        for entry in finished['projections'].values():
            if file_hash(root/entry['file']) != entry['sha256']:
                raise ValueError(f'Corrupted LowRA shard {entry["file"]}')
        return
    for shard in sorted({index[k] for k in names}):
        digest_base(model, verified, shard)
    cache = root/'prepass'
    cache.mkdir(exist_ok=True)
    widths, errors = {}, {}
    for i, name in enumerate(names):
        dest = cache/(name+'.npz')
        if not dest.exists():
            with safe_open(model/index[name], framework='pt') as f:
                w = f.get_tensor(name).float().numpy()
            if w.shape[1] % 64:
                raise ValueError('Expected input width divisible by 64')
            maps, edges, sse = fit_books(w)
            tmp = dest.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez(stream, maps=maps, edges=edges, sse=sse, width=w.shape[1])
            tmp.replace(dest)
        with np.load(dest) as item:
            widths[name] = item['width'].item()
            errors[name] = item['sse']
        if (i+1)%12 == 0:
            print(f'Calibrated {i+1}/{len(names)} projections',flush=True)
    assignments = {}
    for width in sorted(set(widths.values())):
        selected = [n for n in names if widths[n]==width]
        allcost = np.concatenate([errors[n] for n in selected])
        bits = assign_bits(allcost,args.bpp,args.clusters,args.seed)
        split = np.cumsum([len(errors[n]) for n in selected])[:-1]
        assignments.update(zip(selected,np.split(bits,split)))
        print(f'Assigned width {width}, {len(bits)} channels, mean bits {bits.mean():.6f}',flush=True)
    records = {}
    for i, name in enumerate(names):
        target = root/(name+'.safetensors')
        bits = assignments[name]
        if not target.exists():
            with safe_open(model/index[name], framework='pt') as f:
                w = f.get_tensor(name).float()
            with np.load(cache/(name+'.npz')) as item:
                packed, offsets, scales, book, approx = pack_weight(w.numpy(), bits, item['maps'], item['edges'])
            # One LoftQ residual SVD, with a seeded randomized subspace for the large rectangular layers.
            torch.manual_seed(args.seed+i)
            u, s, v = torch.svd_lowrank(w - torch.from_numpy(approx), q=min(args.rank+8,min(w.shape)), niter=2)
            gain = (s[:args.rank]/(args.alpha/args.rank)).sqrt()
            tensors = dict(packed=torch.from_numpy(packed), offsets=torch.from_numpy(offsets),
                           bits=torch.from_numpy(bits.copy()), scales=torch.from_numpy(scales), book=torch.from_numpy(book),
                           up=(u[:,:args.rank]*gain).contiguous(), down=(gain[:,None]*v[:,:args.rank].T).contiguous())
            tmp = target.with_suffix('.tmp')
            save_file(tensors, str(tmp))
            tmp.replace(target)
        records[name[:-7]] = {'file':target.name, 'sha256':file_hash(target), 'bits_sum':int(bits.sum()),
                              'weights':int(len(bits)*widths[name]), 'packed_bits':int(bits.sum())*widths[name]}
        if (i+1)%12 == 0:
            print(f'Packed {i+1}/{len(names)} projections',flush=True)
    write_json(root/'index.json',dict(protocol=cfg,projections=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bpp', type=float, default=1.75)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16)
    parser.add_argument('--clusters', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    prepare(parser.parse_args())
