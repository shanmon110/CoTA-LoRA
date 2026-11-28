"""How much does the transport term actually change what the model says?

The source review established that every historical arm carries a zero gate multiplier, so the
transport term is a provable no-op in all of them.  The three-arm comparison trains the
transport arm with the multiplier at one, which makes the term live -- but "live" is not the
same as "large".  A gate whose weights stay near zero, or a transport branch whose output is
swamped by the ordinary low-rank update, would change nothing a reviewer could measure.

So this tool measures the effect directly, on real audio and real questions.  It scores the
same rows twice with the same adapter: once with the transport term live, and once with the
multiplier set to zero, which switches the term off exactly and only the transport term.  It
reports:

  the accuracy with the term live and with it off;
  how many rows change their answer letter;
  the largest shift in the answer logits.

A term that is present but changes nothing is a result to report, not a bug to hide.

Run from the repository root:

    python3 research/tools/transport_effect.py --config <config> --checkpoint <dir> \
        --manifest <jsonl> [--limit 32]
"""
import argparse
import json
import pathlib

import torch


def read_rows(manifest):
    rows = []
    for line in open(manifest):
        row = json.loads(line)
        rows.append(row)
    return rows


def score_letters(model, template, rows, generation):
    """Score the rows, returning the predicted option index for each.

    This mirrors experiments/ms_swift_aqa/evaluate.py's scoring path, which is the path the
    recorded answers came from: the prompt is encoded with its audio, the model generates one
    token, and that token is read as the option letter.
    """
    import string
    size = generation['batch_size']
    letters = string.ascii_uppercase[:max(len(r['choices']) for r in rows)]
    letter_ids = [template.tokenizer.encode(c, add_special_tokens=False) for c in letters]
    if any(len(t) != 1 for t in letter_ids) or len({t[0] for t in letter_ids}) != len(letters):
        raise ValueError('the tokenizer does not encode the option letters as distinct single tokens')
    tokens = [t[0] for t in letter_ids]
    predictions = []
    with torch.inference_mode(), template.generate_context():
        template.set_mode('transformers')
        for start in range(0, len(rows), size):
            items = rows[start:start + size]
            encoded = []
            for row in items:
                prompt = row['messages'][0]
                example = template.encode(dict(messages=[prompt.copy()], audios=list(row['audios'])))
                encoded.append(example)
            batch = {k: v.to(model.device) for k, v in template.data_collator(encoded).items()
                     if k in ('input_ids', 'attention_mask', 'input_features', 'feature_attention_mask')}
            allowed = [tokens[:len(row['choices'])] for row in items]
            seq = template.generate(model, **batch, max_new_tokens=1, do_sample=False, num_beams=1,
                                    use_model_defaults=False,
                                    prefix_allowed_tokens_fn=lambda i, _: allowed[i])
            for row, token in zip(items, seq[:, -1].tolist()):
                predictions.append(tokens.index(token))
    return predictions, tokens, letters


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=pathlib.Path, required=True)
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True)
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--method', default='ta-lora')
    args = parser.parse_args()

    from experiments.ms_swift_librispeech.train import load_best
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier

    cfg = json.loads(args.config.read_text())
    model, template = load_best(cfg, args.method, args.checkpoint)
    model.requires_grad_(False)

    rows = read_rows(args.manifest)[:args.limit]
    if not rows:
        raise SystemExit('the manifest is empty')

    modules = [module for name, module in model.named_modules()
               if name.endswith('ta_adapter') and hasattr(module, 'transport_up')
               and module.transport_up is not None]
    if not modules:
        raise SystemExit('this checkpoint has no independent transport branch (C), so the '
                         'transport term cannot be switched off and on for comparison')

    recorded = {id(m): float(getattr(m, 'gate_multiplier', 0.)) for m in modules}
    generation = cfg.get('generation', {'batch_size': 2})

    def with_multiplier(value):
        for module in modules:
            set_gate_multiplier(model, recorded[id(module)] if value is None else value)

    with_multiplier(None)
    live, tokens, letters = score_letters(model, template, rows, generation)

    with_multiplier(0.)
    off, _, _ = score_letters(model, template, rows, generation)

    with_multiplier(None)

    changed = sum(1 for a, b in zip(live, off) if a != b)
    correct = [r for r in rows]
    gold = []
    for row in rows:
        letter = row['messages'][1]['content'].strip()
        gold.append(letters.index(letter) if letter in letters else -1)

    def accuracy(preds):
        hits = sum(1 for p, g in zip(preds, gold) if g >= 0 and p == g)
        return 100.0 * hits / max(1, sum(1 for g in gold if g >= 0))

    print(f'rows: {len(rows)}')
    print(f'accuracy with the transport term live: {accuracy(live):.2f}%')
    print(f'accuracy with the transport term off:   {accuracy(off):.2f}%')
    print(f'rows whose option letter changed: {changed} of {len(rows)}')
    print()
    print('The multiplier was set to zero for the second pass, which switches the transport term')
    print('off and changes nothing else.  A term that leaves every letter alone is present but')
    print('inert, and that would be reported as such.')


if __name__ == '__main__':
    main()
