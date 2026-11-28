"""Reconstruct each arm's initial adapter and measure how far each factor actually moved.

The repository records the initialisation only as a SHA256 of the A and B factors
concatenated (`ab_init_sha256` in `adapter_config.json` and `initialization.json`).
The initial tensors are not saved, but the initialisation is fully reproducible:
`adapter_audit.initialize_pair` draws `down` with `kaiming_uniform_` on an ISOLATED CPU
RNG seeded with the run's seed, layer by layer, in `band()` order.

This tool reconstructs that draw, checks it against the recorded SHA (so the reconstruction
is verified, not assumed), and then reports the TRUE displacement of each factor:

  down   |A_trained - A_init|_F  and the relative displacement |dA|_F / |A_init|_F
  up     |B_trained - B_init|_F  and the relative displacement
  gate   |g_trained - g_init|_F  (g_init is 0, so this is |g_trained|)

and, for the pair, the quantity that actually matters for the forward pass:

  the product displacement   |(B_trained - B_init)(A_trained) - B_init A_init|_F

compared with the product itself, |B_trained A_trained|_F, because the update the model
applies is B*A*x, and two adapters can differ a lot in their factors while producing a
nearly identical product (the factors are only identified up to a rotation).

Nothing is trained and no model is loaded.  The checkpoints are read directly.
"""
import argparse
import hashlib
import json
import math
import pathlib
import struct

import numpy as np


def read_safetensors(path):
    with open(path, 'rb') as handle:
        header_len = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        out = {}
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            start, end = spec['data_offsets']
            handle.seek(header_len + 8 + start)
            raw = handle.read(end - start)
            out[name] = np.frombuffer(raw, dtype=np.float32).reshape(spec['shape'])
    return out


def kaiming_uniform_(shape, rng):
    """torch.nn.init.kaiming_uniform_(a=sqrt(5)) on a CPU generator.

    torch's kaiming_uniform_ computes bound = sqrt(3) * gain / sqrt(fan_in), with gain
    = sqrt(2 / (1 + a^2)), and then draws from U(-bound, bound) with the generator.
    For a=sqrt(5): gain = sqrt(2/6) = 1/sqrt(3), so bound = sqrt(3) * (1/sqrt(3)) / sqrt(fan_in)
    = 1 / sqrt(fan_in).
    """
    fan_in = shape[1] if len(shape) == 2 else shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    return (torch_empty(shape, rng) * 2 - 1) * bound


def torch_empty(shape, rng):
    """torch.empty on a CPU generator, then the uniform fill is done with numpy below."""
    import torch
    return torch.empty(shape, dtype=torch.float32).uniform_(-1, 1, generator=rng).numpy()


def reconstruct_down(rank, in_features, seed, count, shape):
    """The `down` draws for `count` layers, from the isolated CPU generator."""
    import torch
    rng = torch.Generator(device='cpu').manual_seed(seed)
    out = []
    for _ in range(count):
        t = torch.empty((rank, in_features), device='cpu', dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=rng)
        out.append(t.numpy())
    return out


def band(n_layers, last_n, first_layer):
    """The layer indices, the same way `adapter_audit.band` returns them."""
    first = n_layers - last_n if first_layer is None else first_layer
    return list(range(first, first + last_n))


def load_arm(run, arm):
    folder = pathlib.Path(run) / 'train' / arm
    result = json.loads((folder / 'result.json').read_text())
    checkpoint = folder / result['best_checkpoint']
    cfg = json.loads((checkpoint / 'adapter_config.json').read_text())
    tensors = read_safetensors(checkpoint / 'adapter_model.safetensors')
    init = {}
    init_file = folder / 'initialization.json'
    if init_file.exists():
        init = json.loads(init_file.read_text())
    return cfg, tensors, init, folder


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--arms', nargs='+', default=None)
    args = parser.parse_args()

    for run in args.runs:
        run = pathlib.Path(run)
        arms = args.arms or sorted(p.name for p in (run / 'train').iterdir() if p.is_dir())
        print('=' * 104)
        print(f'== how far each factor actually moved: {run.name}')
        for arm in arms:
            try:
                cfg, tensors, init, folder = load_arm(run, arm)
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                print(f'   {arm}: not readable ({exc}); skipped')
                continue
            seed = init.get('seed')
            if seed is None:
                args_file = folder / 'args.json'
                if args_file.exists():
                    seed = json.loads(args_file.read_text()).get('seed')
            if seed is None:
                print(f'   {arm}: the seed was not recorded; the initialisation cannot be '
                      f'reconstructed; skipped')
                continue
            targets = cfg['targets']
            rank, alpha = cfg['rank'], cfg['alpha']
            scale = alpha / rank
            layers = sorted({int(t.split('.')[3]) for t in targets})

            # the layer order `audit_adapter` uses: `band()` order, and within a layer the
            # projection order.  For an O-only adapter the order is the layer order.
            downs = reconstruct_down(rank, tensors[[k for k in tensors if k.endswith('.down')][0]].shape[1],
                                     seed, len(layers), None)

            # VERIFY the reconstruction against the recorded SHA: the initial A is the
            # reconstruction and the initial B is exactly zero (PEFT's convention), hashed
            # A then B per layer in band order.  A match means the reconstruction is the
            # run's actual initialisation, not a plausible guess.
            digest = hashlib.sha256()
            for i, layer in enumerate(layers):
                up0 = tensors[f'thinker.model.layers.{layer}.self_attn.ta_adapter.up']
                digest.update(np.ascontiguousarray(downs[i], dtype=np.float32).tobytes())
                digest.update(np.zeros_like(up0, dtype=np.float32).tobytes())
            recon = digest.hexdigest()
            recorded = init.get('ab_init_sha256', '')
            verified = (recon == recorded)

            print(f'   --- {arm}  (seed {seed}, {len(layers)} layers, r={rank}, alpha={alpha}, '
                  f'scale={scale})')
            print(f'       recorded ab_init_sha256  {recorded[:40]}')
            print(f'       reconstructed             {recon[:40]}   '
                  f'{"VERIFIED: the reconstruction IS the run initialisation" if verified else "NO MATCH -- do not trust the displacements below"}')

            rel_d, rel_u = [], []
            prod_before, prod_after, prod_delta = [], [], []
            for i, layer in enumerate(layers):
                down = tensors[f'thinker.model.layers.{layer}.self_attn.ta_adapter.down']
                up = tensors[f'thinker.model.layers.{layer}.self_attn.ta_adapter.up']
                gate = tensors[f'thinker.model.layers.{layer}.self_attn.ta_adapter.gate']
                d0 = downs[i]
                u0 = np.zeros_like(up)
                dd = float(np.linalg.norm(down - d0))
                du = float(np.linalg.norm(up - u0))
                rel_d.append(dd / max(float(np.linalg.norm(d0)), 1e-12))
                rel_u.append(du / max(float(np.linalg.norm(u0)), 1e-12))
                p_after = scale * (up @ down)
                p_before = scale * (u0 @ d0)
                prod_after.append(float(np.linalg.norm(p_after)))
                prod_before.append(float(np.linalg.norm(p_before)))
                prod_delta.append(float(np.linalg.norm(p_after - p_before)))
                if i == 0:
                    g = gate
                    print(f'       layer {layer}: |A_trained - A_init|_F {dd:.5f} '
                          f'(relative {rel_d[-1]:.3f}), |B_trained|_F {du:.5f}, '
                          f'|g|_F {float(np.linalg.norm(g)):.5f}')
                    print(f'                 the update B*A is {float(np.linalg.norm(p_after)):.5f} '
                          f'in Frobenius norm; the initial B*A was 0')
            print(f'       across the {len(layers)} layers:')
            print(f'         relative displacement of A   {np.mean(rel_d):.4f} '
                  f'[min {np.min(rel_d):.4f}, max {np.max(rel_d):.4f}]')
            print(f'         the initial B is zero, so the relative displacement of B is '
                  f'not finite; its absolute norm is')
            print(f'         |B_trained|_F per layer '
                  f'[min {np.mean([float(np.linalg.norm(tensors[f"thinker.model.layers.{l}.self_attn.ta_adapter.up"])) for l in layers]):.5f}]')
            print(f'         the product displacement |B_trained A_trained - B_init A_init|_F '
                  f'= {np.mean(prod_delta):.5f}, and the product itself is '
                  f'{np.mean(prod_after):.5f}')
            print()


if __name__ == '__main__':
    main()
