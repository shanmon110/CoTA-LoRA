"""Is what the last four layers learn more than a shift of the four letter logits?

`extract` asks the trained question exactly as the scoring does and keeps the logits of the letters A-D at the
position the answer is read from. `analyze` then builds a surrogate with three free numbers: the frozen base's
logits plus one constant per letter, fitted on IEMOCAP Sessions 1-2 in the training option order, which is what
the adapters were trained on. If this surrogate gains what the last-four-layer arm gains in the training order
and loses what it loses under the other orders, that arm has learned little beyond a letter bias.
Declared in research/letter-bias-20261125.md before any logit was extracted.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import crosscorpus
from experiments.trr_diagnostics import d2

CLASSES = 'ABCD'
CORPORA = ('session4', 'cremad')
# Written down before any logit was extracted.
CRITERIA = dict(min_recovered=2 / 3, max_not_recovered=1 / 3, max_better_than_surrogate_pp=5.0)


def rows_of(source, kind):
    """Rows with id, label, rotation, option_order and cluster, from a training or an evaluation manifest."""
    out = []
    for r in d2.read_jsonl(source):
        if kind == 'training':
            out.append(dict(id=r['id'], audios=r['audios'], messages=r['messages'], label=r['messages'][1]['content'], rotation=0,
                            option_order=list(CLASSES), cluster=r['id'].rsplit('_', 1)[0]))
        else:
            out.append(dict(id=r['id'], audios=r['audios'], messages=r['messages'], label=r['label'], rotation=r['rotation'],
                            option_order=r['option_order'], cluster=r['dialogue']))
    return out


def extract(source, kind, model_config, method, checkpoint, output, *, batch_size=4):
    import torch

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
    rows = rows_of(source, kind)
    tokens = [template.tokenizer.encode(c, add_special_tokens=False) for c in CLASSES]
    if any(len(t) != 1 for t in tokens):
        raise ValueError('Expected four single-token answers')
    tokens = [t[0] for t in tokens]
    seen = []
    handle = model.thinker.lm_head.register_forward_hook(lambda module, args, out: seen.append(out.detach()))
    logits, answers = [], []
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
                seen.clear()
                seq = template.generate(model, **batch, max_new_tokens=1, do_sample=False, num_beams=1,
                                        use_model_defaults=False, prefix_allowed_tokens_fn=lambda *_: tokens)
                if len(seen) != 1 or seen[0].shape[0] != len(items):
                    raise ValueError('Expected one pass through the output head per batch')
                four = seen[0][:, -1, tokens].float().cpu()
                chosen = [tokens.index(t) for t in seq[:, -1].tolist()]
                if four.argmax(dim=1).tolist() != chosen and not torch.allclose(four.max(dim=1).values, four[torch.arange(len(items)), chosen]):
                    raise ValueError('The captured logits do not give the letters the model answered')
                logits.append(four.numpy())
                answers += [CLASSES[i] for i in chosen]
    finally:
        handle.remove()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'logits.npy', np.concatenate(logits).astype(np.float32))
    d2.write_jsonl(output / 'rows.jsonl', [dict(id=r['id'], label=r['label'], rotation=r['rotation'], option_order=r['option_order'],
                                                cluster=r['cluster'], answer=a) for r, a in zip(rows, answers, strict=True)])
    d2.write(output / 'done.json', dict(rows=len(rows), batch_size=batch_size, source_sha256=d2.digest(source)))


# ------------------------------------------------------------------ analysis
def load(folder):
    rows = d2.read_jsonl(Path(folder) / 'rows.jsonl')
    return np.load(Path(folder) / 'logits.npy'), rows


def fit_bias(logits, target, steps=3000, rate=0.5):
    """One constant per letter, added to the logits, that maximises the class-balanced likelihood of the target letters."""
    counts = np.bincount(target, minlength=4)
    weight = (len(target) / (4 * counts))[target]
    bias = np.zeros(4)
    onehot = np.eye(4)[target]
    for _ in range(steps):
        z = logits + bias
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        bias -= rate * ((p - onehot) * weight[:, None]).mean(axis=0)
        bias -= bias.mean()
    return bias


class Corpus:
    """Rows of one corpus under four option orders, with per-cluster bookkeeping for the bootstrap."""

    def __init__(self, rows, resamples, seed):
        self.label = np.array([CLASSES.index(r['label']) for r in rows])
        self.order = np.array([[CLASSES.index(c) for c in r['option_order']] for r in rows])       # letter index -> emotion index
        self.training = np.array([r['rotation'] == 0 for r in rows])
        names = sorted({r['cluster'] for r in rows})
        index = np.array([names.index(r['cluster']) for r in rows])
        self.member = np.zeros((len(names), len(rows)))
        self.member[index, np.arange(len(rows))] = 1
        self.weights = np.random.default_rng(seed).multinomial(len(names), np.full(len(names), 1 / len(names)), size=resamples)

    def right(self, letters):
        return (self.order[np.arange(len(letters)), letters] == self.label).astype(float)

    def recall(self, right, keep):
        """Mean per-class recall on the kept rows: the value and its bootstrap draws."""
        hit = np.stack([self.member @ (right * keep * (self.label == c)) for c in range(4)], axis=1)       # clusters, classes
        count = np.stack([self.member @ (keep * (self.label == c)).astype(float) for c in range(4)], axis=1)
        return float((hit.sum(0) / count.sum(0)).mean()), ((self.weights @ hit) / (self.weights @ count)).mean(axis=1)


def analyze(root, *, resamples=10000, seed=42):
    """`root/<model>/<split>/` holds logits. Splits: train (base only), session4, cremad."""
    root = Path(root)
    train_logits, train_rows = load(root / 'frozen-base' / 'train')
    bias = fit_bias(train_logits, np.array([CLASSES.index(r['label']) for r in train_rows]))
    last = [m for m in crosscorpus.FAMILIES['A-last4'] if (root / m / 'session4' / 'done.json').exists()]
    result = dict(letter_bias=dict(zip(CLASSES, map(float, bias))), fitted_on=len(train_rows), last4_models=last, criteria=CRITERIA, corpora={})
    picks = np.random.default_rng(seed + 1).integers(0, max(len(last), 1), size=(resamples, max(len(last), 1)))
    for split in CORPORA:
        base_logits, rows = load(root / 'frozen-base' / split)
        corpus = Corpus(rows, resamples, seed)
        letters = {'frozen-base': base_logits.argmax(1), 'surrogate': (base_logits + bias).argmax(1)}
        for model in sorted(p.name for p in root.iterdir() if p.name != 'frozen-base' and (p / split / 'done.json').exists()):
            logits, other = load(root / model / split)
            if [r['id'] for r in other] != [r['id'] for r in rows]:
                raise ValueError(f'{model}/{split}: rows differ from the frozen base')
            letters[model] = logits.argmax(1)
        parts = dict(training_order=corpus.training, other_orders=~corpus.training)
        score = {name: {part: corpus.recall(corpus.right(chosen), keep) for part, keep in parts.items()} for name, chosen in letters.items()}

        def family(part):
            value = float(np.mean([score[m][part][0] for m in last]))
            stacked = np.stack([score[m][part][1] for m in last])
            return value, stacked[picks.T, np.arange(resamples)].mean(axis=0)

        def difference(a, b):
            low, high = np.nanpercentile(a[1] - b[1], [2.5, 97.5])
            return dict(delta_pp=100 * (a[0] - b[0]), interval_95_pp=[100 * float(low), 100 * float(high)])

        entry = dict(rows=len(rows), recall={name: {part: 100 * s[part][0] for part in parts} for name, s in score.items()},
                     letter_shares={name: (np.bincount(chosen[corpus.training], minlength=4) / corpus.training.sum()).tolist() for name, chosen in letters.items()})
        if last:
            entry['recall']['A-last4 (mean)'] = {part: 100 * family(part)[0] for part in parts}
            base, surrogate = score['frozen-base'], score['surrogate']
            entry['surrogate_minus_base'] = {part: difference(surrogate[part], base[part]) for part in parts}
            entry['last4_minus_base'] = {part: difference(family(part), base[part]) for part in parts}
            entry['last4_minus_surrogate'] = {part: difference(family(part), surrogate[part]) for part in parts}
            gain = family('training_order')
            ratio = (surrogate['training_order'][1] - base['training_order'][1]) / np.where(gain[1] - base['training_order'][1] == 0, np.nan,
                                                                                            gain[1] - base['training_order'][1])
            low, high = np.nanpercentile(ratio, [2.5, 97.5])
            entry['share_of_gain_recovered'] = dict(
                value=(surrogate['training_order'][0] - base['training_order'][0]) / (gain[0] - base['training_order'][0]),
                interval_95=[float(low), float(high)])
            entry['same_letter_as_last4'] = {name: float(np.mean([np.mean(letters[name] == letters[m]) for m in last]))
                                             for name in ('frozen-base', 'surrogate')}
            if len(last) > 1:
                entry['same_letter_as_last4']['another seed of A-last4'] = float(np.mean(
                    [np.mean(letters[a] == letters[b]) for i, a in enumerate(last) for b in last[i + 1:]]))
            entry['reading'] = judge(entry)
        result['corpora'][split] = entry
    return result


def judge(entry):
    recovered, harm = entry['share_of_gain_recovered'], entry['surrogate_minus_base']['other_orders']
    beyond = entry['last4_minus_surrogate']['other_orders']['delta_pp']          # an arm that hears better would beat the surrogate here
    if (recovered['value'] >= CRITERIA['min_recovered'] and recovered['interval_95'][0] > CRITERIA['max_not_recovered']
            and harm['interval_95_pp'][1] < 0 and beyond < CRITERIA['max_better_than_surrogate_pp']):
        return 'mostly a letter bias'
    if recovered['value'] < CRITERIA['max_not_recovered']:
        return 'not a letter bias'
    return 'partly a letter bias'


def render(result):
    lines = ['# The last four layers against a three-number surrogate', '',
             'Surrogate: the frozen base with one constant added to each letter logit, fitted on '
             f"{result['fitted_on']} training utterances in the training option order. Constants: "
             + ', '.join(f'{k} {v:+.2f}' for k, v in result['letter_bias'].items()) + '. Mean per-class recall in %.', '']
    for split, entry in result['corpora'].items():
        lines += [f"## {split} ({entry['rows']} rows)", '', '| model | training order | other orders | letters chosen in the training order, A B C D % |', '|---|---:|---:|---|']
        for name, recall in entry['recall'].items():
            shares = entry['letter_shares'].get(name)
            lines.append(f"| {name} | {recall['training_order']:.1f} | {recall['other_orders']:.1f} | "
                         + (' '.join(f'{100 * s:.0f}' for s in shares) if shares else '') + ' |')
        if 'reading' in entry:
            def cell(c):
                return f"{c['delta_pp']:+.1f} [{c['interval_95_pp'][0]:+.1f}, {c['interval_95_pp'][1]:+.1f}]"
            r = entry['share_of_gain_recovered']
            lines += ['', '| difference pp [95% CI] | training order | other orders |', '|---|---|---|']
            lines += [f"| {key.replace('_', ' ')} | {cell(entry[key]['training_order'])} | {cell(entry[key]['other_orders'])} |"
                      for key in ('surrogate_minus_base', 'last4_minus_base', 'last4_minus_surrogate')]
            lines += ['', f"Share of the last-four-layer gain in the training order that the surrogate recovers: {100 * r['value']:.0f}% "
                          f"[{100 * r['interval_95'][0]:.0f}, {100 * r['interval_95'][1]:.0f}].",
                      'Same letter as A-last4: ' + ', '.join(f'{k} {100 * v:.1f}%' for k, v in entry['same_letter_as_last4'].items()) + '.',
                      '', f"Reading declared before extraction: **{entry['reading']}**"]
        lines.append('')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('extract')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--kind', choices=['training', 'evaluation'], required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--method', choices=['frozen-base', 'ta-lora'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p = commands.add_parser('analyze')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        extract(args.source, args.kind, args.model_config, args.method, args.checkpoint, args.output, batch_size=args.batch_size)
    else:
        summary = analyze(args.root)
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'letters.json', summary)
        (args.output / 'letters.md').write_text(render(summary))
        print(render(summary))
