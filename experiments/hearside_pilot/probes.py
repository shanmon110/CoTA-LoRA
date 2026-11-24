"""Is the emotion decodable inside the model where the model's own answer does not show it?

For a model and a set of utterances, `extract` asks the trained letter question in the training option order,
exactly as the scoring does, and stores the residual stream after every decoder layer (and the layer-0 input) at
two sites: the mean over the audio positions, and the last prompt position, from which the answer is read.
`fit` trains a linear classifier per layer and site on IEMOCAP Sessions 1-2, chooses its regularisation and the
reported layer on Session 3, and tests on Session 4 and on CREMA-D. `summarize` sets the probes beside the
answers the same model gave. Declared in research/probes-20261125.md before any state was extracted.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.trr_diagnostics import d2

CLASSES = 'ABCD'
SITES = ('audio', 'answer')
SPLITS = ('train', 'dev', 'session4', 'cremad')
TESTS = ('session4', 'cremad')
PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1)
CONTROLS = 5                      # probes trained on permuted labels; their mean recall should sit near 25%
# Written down before any state was extracted.
CRITERIA = dict(gap_pp=10.0, no_gap_pp=5.0, readout_pp=3.0, readout_upper_pp=5.0)


# ------------------------------------------------------------------ extraction
def capture(layers):
    """Hooks that keep, for one forward pass, the input of the first layer and the output of every layer."""
    store, handles = [], []

    def first(module, args, kwargs):
        store.append((0, (args[0] if args else kwargs['hidden_states']).detach()))

    def after(index):
        def hook(module, args, output):
            store.append((index + 1, (output[0] if isinstance(output, tuple) else output).detach()))
        return hook

    handles.append(layers[0].register_forward_pre_hook(first, with_kwargs=True))
    handles += [layer.register_forward_hook(after(i)) for i, layer in enumerate(layers)]
    return store, handles


def pooled(store, count, audio, width):
    """(sites, layers + 1, batch, dim) from the hooks of one pass over a batch of left-padded prompts."""
    import torch
    states = {}
    for index, hidden in store:
        if hidden.shape[:2] == audio.shape and index not in states:      # the pass over the whole prompt, once per layer
            states[index] = hidden.float()
    if sorted(states) != list(range(count + 1)):
        raise ValueError('The forward pass did not go through every decoder layer with the full prompt')
    if not audio.any(dim=1).all():
        raise ValueError('A prompt has no audio positions')
    stack = torch.stack([states[i] for i in range(count + 1)])                                  # layers, batch, positions, dim
    mask = audio.unsqueeze(0).unsqueeze(-1).to(stack.dtype)
    mean = (stack * mask).sum(dim=2) / mask.sum(dim=2)
    assert stack.shape[-1] == width
    return torch.stack([mean, stack[:, :, -1]]).cpu()


def extract(manifest, model_config, method, checkpoint, output, *, batch_size=2):
    import torch
    from numpy.lib.format import open_memmap

    output = Path(output)
    if (output / 'done.json').exists():
        return
    cfg = d2.load(model_config)
    if method == 'frozen-base':
        from experiments.ms_swift_aqa.evaluate_frozen import load_frozen
        model, template = load_frozen(cfg)
    else:
        from experiments.ms_swift_librispeech.train import load_best
        model, template = load_best(cfg, method, Path(checkpoint))
        model.requires_grad_(False)
    rows = d2.read_jsonl(manifest)
    layers = model.thinker.model.layers
    width, audio_token = model.thinker.config.text_config.hidden_size, model.thinker.config.audio_token_id
    tokens = [template.tokenizer.encode(c, add_special_tokens=False) for c in CLASSES]
    if any(len(t) != 1 for t in tokens):
        raise ValueError('Expected four single-token answers')
    tokens = [t[0] for t in tokens]
    output.mkdir(parents=True, exist_ok=True)
    arrays = {site: open_memmap(output / f'{site}.npy', mode='w+', dtype=np.float16, shape=(len(rows), len(layers) + 1, width)) for site in SITES}
    store, handles = capture(layers)
    answers = []
    try:
        with torch.inference_mode(), template.generate_context():
            template.set_mode('transformers')
            for start in range(0, len(rows), batch_size):
                items = rows[start:start + batch_size]
                enc = [template.encode(dict(messages=[r['messages'][0]], audios=list(r['audios']))) for r in items]
                batch = {k: v.to(model.device) for k, v in template.data_collator(enc).items()
                         if k in ('input_ids', 'attention_mask', 'input_features', 'feature_attention_mask')}
                if not batch['attention_mask'][:, -1].bool().all():
                    raise ValueError('Expected left padded prompts')
                store.clear()
                seq = template.generate(model, **batch, max_new_tokens=1, do_sample=False, num_beams=1,
                                        use_model_defaults=False, prefix_allowed_tokens_fn=lambda *_: tokens)
                both = pooled(store, len(layers), batch['input_ids'].eq(audio_token) & batch['attention_mask'].bool(), width)
                for s, site in enumerate(SITES):
                    arrays[site][start:start + len(items)] = both[s].transpose(0, 1).clamp(-6e4, 6e4).numpy().astype(np.float16)   # half precision on disk
                answers += [CLASSES[tokens.index(t)] for t in seq[:, -1].tolist()]
    finally:
        for handle in handles:
            handle.remove()
    for array in arrays.values():
        array.flush()
    d2.write_jsonl(output / 'rows.jsonl', [dict(id=r['id'], label=r['label'], cluster=r['cluster'], answer=a) for r, a in zip(rows, answers, strict=True)])
    d2.write(output / 'done.json', dict(rows=len(rows), layers=len(layers), width=width, manifest_sha256=d2.digest(manifest), batch_size=batch_size))


def manifest(source, kind, output):
    """One row per utterance with the trained question in the training option order, a label and a bootstrap cluster."""
    rows = []
    for r in d2.read_jsonl(source):
        if kind == 'training':                       # train-fixed.jsonl and dev-fixed.jsonl of the pilot
            rows.append(dict(id=r['id'], audios=r['audios'], messages=r['messages'], label=r['messages'][1]['content'],
                             cluster=r['id'].rsplit('_', 1)[0]))
        elif r['rotation'] == 0:                     # an evaluation manifest with four option orders
            rows.append(dict(id=r['utterance'], audios=r['audios'], messages=r['messages'], label=r['label'], cluster=r['dialogue']))
    if any(r['messages'][1]['content'] != r['label'] or r['label'] not in CLASSES for r in rows):
        raise ValueError('In the training option order the answer letter is the class letter')
    d2.write_jsonl(output, rows)
    return len(rows)


# ------------------------------------------------------------------ probes
def uar(labels, predicted):
    return float(np.mean([np.mean(predicted[labels == c] == c) for c in range(4)]))


def train_probe(x, y, penalty, device, steps=100):
    """Class-balanced multinomial logistic regression with an L2 penalty, on standardised features."""
    import torch
    x, y = torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(y, device=device)
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / scale
    weight = torch.zeros(x.shape[1], 4, device=device, requires_grad=True)
    bias = torch.zeros(4, device=device, requires_grad=True)
    counts = torch.bincount(y, minlength=4).float()
    balance = (len(y) / (4 * counts))[y]
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.cross_entropy(x @ weight + bias, y, reduction='none') * balance).mean() + penalty * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    weight, bias = weight.detach(), bias.detach()
    return lambda z: (((torch.as_tensor(z, dtype=torch.float32, device=device) - mean) / scale) @ weight + bias).argmax(1).cpu().numpy()


def labels_of(folder):
    return np.array([CLASSES.index(r['label']) for r in d2.read_jsonl(Path(folder) / 'rows.jsonl')])


def fit(root, output, *, device=None, seed=42):
    """`root` holds one folder of extracted states per split. Writes per-utterance correctness for every layer and site."""
    import torch
    root, output = Path(root), Path(output)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    y = {s: labels_of(root / s) for s in SPLITS}
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(y['train']) for _ in range(CONTROLS)]
    layers = d2.load(root / 'train' / 'done.json')['layers'] + 1
    table, correct = {}, {}
    for site in SITES:
        x = {s: np.load(root / s / f'{site}.npy', mmap_mode='r') for s in SPLITS}
        for layer in range(layers):
            feats = {s: np.asarray(x[s][:, layer], dtype=np.float32) for s in SPLITS}
            best = None
            for penalty in PENALTIES:
                predict = train_probe(feats['train'], y['train'], penalty, device)
                score = uar(y['dev'], predict(feats['dev']))
                if best is None or score > best[0]:
                    best = (score, penalty, predict)
            controls = [train_probe(feats['train'], labels, best[1], device) for labels in shuffled]     # same probe, permuted labels
            entry = dict(dev=best[0], penalty=best[1])
            for split in TESTS:
                right = best[2](feats[split]) == y[split]
                correct[f'{site}/{layer}/{split}'] = right
                entry[split] = uar(y[split], np.where(right, y[split], -1))
                entry[f'{split}_control'] = float(np.mean([uar(y[split], control(feats[split])) for control in controls]))
            table[f'{site}/{layer}'] = entry
            print(f"{site} layer {layer}: dev {entry['dev']:.3f} session4 {entry['session4']:.3f} cremad {entry['cremad']:.3f}", flush=True)
    chosen = {site: max(range(layers), key=lambda l: table[f'{site}/{l}']['dev']) for site in SITES}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'correct.npz', **correct)
    d2.write(output / 'probes.json', dict(layers=layers, chosen_layer=chosen, table=table, penalties=PENALTIES, seed=seed,
                                          rows={s: len(y[s]) for s in SPLITS}))
    return chosen


# ------------------------------------------------------------------ comparison with the model's own answers
def answers(folder, arm, ids):
    """Per utterance: the share of the four option orders in which the model's letter was right."""
    by = {}
    for r in d2.read_jsonl(Path(folder) / 'runs' / arm / 'mcq' / 'predictions.jsonl'):
        by.setdefault(r['id'].rsplit('#', 1)[0], []).append(r['prediction'].strip() == r['reference'])
    if any(len(by.get(i, ())) != 4 for i in ids):
        raise ValueError(f'{arm}: expected four option orders for every utterance')
    return np.array([np.mean(by[i]) for i in ids])


class Resampler:
    def __init__(self, labels, clusters, resamples, seed):
        names = sorted(set(clusters))
        self.member = np.zeros((len(names), len(labels), 4))
        for i, (label, cluster) in enumerate(zip(labels, clusters)):
            self.member[names.index(cluster), i, label] = 1
        self.weights = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1 / len(names)), size=resamples)
        self.count = self.member.sum(axis=1)

    def score(self, correct):
        return float(np.mean((self.member * correct[None, :, None]).sum(axis=(0, 1)) / self.count.sum(axis=0)))

    def draws(self, correct):
        right = (self.member * correct[None, :, None]).sum(axis=1)                                  # clusters, classes
        return ((self.weights @ right) / (self.weights @ self.count)).mean(axis=1)


def contrast(sampler, first, second):
    boot = sampler.draws(first) - sampler.draws(second)
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return dict(delta_pp=100 * (sampler.score(first) - sampler.score(second)), interval_95_pp=[100 * float(low), 100 * float(high)])


def summarize(root, outputs, *, resamples=10000, seed=42):
    """`root/<model>/` holds states and probes; `outputs[split]` is a crosscorpus folder with every model's answers."""
    root = Path(root)
    models = sorted(p.name for p in root.iterdir() if (p / 'probes' / 'probes.json').exists())
    result = dict(models=models, criteria=CRITERIA, corpora={})
    for split in TESTS:
        rows = d2.read_jsonl(root / 'frozen-base' / split / 'rows.jsonl')
        labels = np.array([CLASSES.index(r['label']) for r in rows])
        sampler = Resampler(labels, [r['cluster'] for r in rows], resamples, seed)
        ids = [r['id'] for r in rows]
        entry = dict(utterances=len(rows), models={})
        probe = {}
        for model in models:
            info = d2.load(root / model / 'probes' / 'probes.json')
            correct = np.load(root / model / 'probes' / 'correct.npz')
            given = answers(outputs[split], model, ids)
            probe[model] = {site: correct[f"{site}/{info['chosen_layer'][site]}/{split}"].astype(float) for site in SITES}
            probe[model]['given'] = given
            training_order = np.array([r['answer'] == r['label'] for r in d2.read_jsonl(root / model / split / 'rows.jsonl')], dtype=float)
            entry['models'][model] = dict(
                answer_all_orders=100 * sampler.score(given), answer_training_order=100 * sampler.score(training_order),
                chosen_layer=info['chosen_layer'],
                probe={site: 100 * sampler.score(probe[model][site]) for site in SITES},
                control={site: 100 * info['table'][f"{site}/{info['chosen_layer'][site]}"][f'{split}_control'] for site in SITES},
                profile={site: [100 * info['table'][f'{site}/{l}'][split] for l in range(info['layers'])] for site in SITES},
                probe_minus_answer={site: contrast(sampler, probe[model][site], given) for site in SITES},
                answer_site_minus_audio_site=contrast(sampler, probe[model]['answer'], probe[model]['audio']))
        base = probe.get('frozen-base')
        if base is not None:
            better = max(SITES, key=lambda s: sampler.score(base[s]))
            entry['base_best_site'] = better
            for model in models:
                if model != 'frozen-base':
                    entry['models'][model]['answer_minus_base_probe'] = contrast(sampler, probe[model]['given'], base[better])
        entry['readings'] = judge(entry)
        result['corpora'][split] = entry
    return result


def judge(entry):
    out = dict(perception_use_gap=None, information_reaches_the_answer_position=None, what_full_depth_training_adds=None)
    base = entry['models'].get('frozen-base')
    if base is None:
        return out
    gap = max((base['probe_minus_answer'][s] for s in SITES), key=lambda c: c['delta_pp'])
    if gap['delta_pp'] >= CRITERIA['gap_pp'] and gap['interval_95_pp'][0] > 0:
        out['perception_use_gap'] = 'present'
    elif gap['interval_95_pp'][1] < CRITERIA['no_gap_pp']:
        out['perception_use_gap'] = 'absent'
    else:
        out['perception_use_gap'] = 'unclear'
    route = base['answer_site_minus_audio_site']
    if route['delta_pp'] <= -CRITERIA['gap_pp'] and route['interval_95_pp'][1] < 0:
        out['information_reaches_the_answer_position'] = 'no'
    elif route['interval_95_pp'][0] > -CRITERIA['no_gap_pp']:
        out['information_reaches_the_answer_position'] = 'yes'
    else:
        out['information_reaches_the_answer_position'] = 'unclear'
    full = entry['models'].get('A-all', {}).get('answer_minus_base_probe')
    if full is not None:
        if full['delta_pp'] >= CRITERIA['gap_pp'] and full['interval_95_pp'][0] > 0:
            out['what_full_depth_training_adds'] = 'more than a linear readout of the base'
        elif full['delta_pp'] <= CRITERIA['readout_pp'] and full['interval_95_pp'][1] <= CRITERIA['readout_upper_pp']:
            out['what_full_depth_training_adds'] = 'no more than a linear readout of the base'
        else:
            out['what_full_depth_training_adds'] = 'between'
    return out


def render(result):
    lines = ['# Linear probes of the emotion beside the answers the model gave', '',
             'Mean per-class recall in %. Probes: trained on Sessions 1-2, regularisation and layer chosen on Session 3. '
             'Control: the same probe trained on permuted labels, mean of five permutations (25 expected).', '']
    for split, entry in result['corpora'].items():
        lines += [f"## {split} ({entry['utterances']} utterances)", '',
                  '| model | answer, all orders | answer, training order | probe at audio positions (layer) | probe at answer position (layer) | controls | best probe minus answer [95% CI] |',
                  '|---|---:|---:|---:|---:|---|---|']
        for model, m in entry['models'].items():
            best = max((m['probe_minus_answer'][s] for s in SITES), key=lambda c: c['delta_pp'])
            lines.append(f"| {model} | {m['answer_all_orders']:.1f} | {m['answer_training_order']:.1f} | "
                         + ' | '.join(f"{m['probe'][s]:.1f} ({m['chosen_layer'][s]})" for s in SITES)
                         + f" | {m['control']['audio']:.1f}, {m['control']['answer']:.1f} | "
                         f"{best['delta_pp']:+.1f} [{best['interval_95_pp'][0]:+.1f}, {best['interval_95_pp'][1]:+.1f}] |")
        lines += ['', 'Readings declared before extraction:', '']
        lines += [f"- {k}: **{v or 'not available'}**" for k, v in entry['readings'].items()]
        for model, m in entry['models'].items():
            if 'answer_minus_base_probe' in m:
                c = m['answer_minus_base_probe']
                lines.append(f"- {model}: its answers minus the best probe of the frozen base ({entry['base_best_site']} site): "
                             f"{c['delta_pp']:+.1f} [{c['interval_95_pp'][0]:+.1f}, {c['interval_95_pp'][1]:+.1f}] pp")
        lines.append('')
    return '\n'.join(lines) + '\n'


def unchanged(root, model, split, last_layer):
    """Largest difference between a model's states and the frozen base's up to a layer the model did not train."""
    worst = 0.0
    for site in SITES:
        a = np.load(Path(root) / model / split / f'{site}.npy', mmap_mode='r')
        b = np.load(Path(root) / 'frozen-base' / split / f'{site}.npy', mmap_mode='r')
        worst = max(worst, float(np.abs(np.asarray(a[:, :last_layer + 1], dtype=np.float32) - np.asarray(b[:, :last_layer + 1], dtype=np.float32)).max()))
    return worst


def agree(states, folder, arm):
    """Utterances whose letter during extraction differs from the one the earlier scoring recorded for the training order."""
    earlier = {r['id'].rsplit('#', 1)[0]: r['prediction'].strip() for r in d2.read_jsonl(Path(folder) / 'runs' / arm / 'mcq' / 'predictions.jsonl')
               if r['id'].endswith('#r0')}
    rows = d2.read_jsonl(Path(states) / 'rows.jsonl')
    return dict(rows=len(rows), different=sum(earlier[r['id']] != r['answer'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('manifest')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--kind', choices=['training', 'evaluation'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('extract')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--method', choices=['frozen-base', 'ta-lora'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p = commands.add_parser('fit')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('unchanged')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--last-layer', type=int, required=True)
    p = commands.add_parser('agree')
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--session4', type=Path, required=True)
    p.add_argument('--cremad', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'manifest':
        print(manifest(args.source, args.kind, args.output), 'utterances')
    elif args.command == 'extract':
        extract(args.manifest, args.model_config, args.method, args.checkpoint, args.output, batch_size=args.batch_size)
    elif args.command == 'fit':
        print(json.dumps(fit(args.root, args.output)))
    elif args.command == 'agree':
        found = agree(args.states, args.folder, args.arm)
        print(json.dumps(found))
        raise SystemExit(int(found['different'] > 0.01 * found['rows']))
    elif args.command == 'unchanged':
        print(json.dumps({split: unchanged(args.root, args.model, split, args.last_layer) for split in TESTS}))
    else:
        summary = summarize(args.root, dict(session4=args.session4, cremad=args.cremad))
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'probes.json', summary)
        (args.output / 'probes.md').write_text(render(summary))
        print(render(summary))
