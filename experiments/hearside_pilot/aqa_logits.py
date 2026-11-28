"""The option logits of every arm on the AQA manifests, for the surrogate test.

`aqa.py run` tracks the arm's ANSWER, not its option logits.  The surrogate test needs the
logits, because the whole question is whether a few degrees of freedom applied to the arm's own
logits reconstruct its behaviour.

This module is the AQA counterpart of `experiments/hearside_pilot/letters.py extract`: it runs
the same models over the same MCQ manifest and saves the option logits, so the surrogate can be
fitted and tested on them.

Two things are saved per arm:

  logits.npy   (rows, n_options) the logit of each option, in the display order of the row
  rows.jsonl   id, shift, the gold letter, the option count, the question id

The rows must line up across arms, so the order is the manifest's order and the tool checks it.

Run from the repository root:

    python3 -m experiments.hearside_pilot.aqa_logits --folder <manifest folder> \
        --model-config <config> --method <method> --arm <arm> --output <folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.trr_diagnostics import d2

LETTERS = 'ABCDEFGH'


def read_manifest(folder):
    """The MCQ manifest rows, with the gold letter and the option count."""
    rows = []
    for line in open(Path(folder) / 'mcq.jsonl'):
        row = json.loads(line)
        prompt = row['messages'][0]['content']
        options = []
        for line_text in prompt.split('\n'):
            if len(line_text) > 2 and line_text[0] in LETTERS and line_text[1] == '.':
                options.append(line_text[2:].strip())
        if len(options) != len(row['choices']):
            raise ValueError(f"{row['id']}: {len(options)} options in the prompt, "
                             f"{len(row['choices'])} in the manifest")
        rows.append(dict(id=row['id'], shift=row['shift'], choices=row['choices'],
                         options=options, gold=row['messages'][1]['content'],
                         question=row['source'], n=len(options),
                         messages=row['messages'], audios=row['audios']))
    return rows


def extract(folder, model_config, method, checkpoint, arm, output, *, batch_size=2):
    import torch

    output = Path(output)
    target = output / arm / 'mcq'
    if (target / 'done.json').exists():
        return
    rows = read_manifest(folder)
    cfg = d2.load(model_config)
    if method == 'frozen-base':
        from experiments.ms_swift_aqa.evaluate_frozen import load_frozen
        model, template = load_frozen(cfg)
    else:
        from experiments.ms_swift_librispeech.train import load_best
        model, template = load_best(cfg, method, Path(checkpoint))
        model.requires_grad_(False)
    token_ids = [template.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    if any(len(t) != 1 for t in token_ids):
        raise ValueError('Expected the option letters to be single tokens')
    token_ids = [t[0] for t in token_ids]

    seen = []
    handle = model.thinker.lm_head.register_forward_hook(
        lambda module, args, out: seen.append(out.detach()))
    chunks = []
    try:
        with torch.inference_mode(), template.generate_context():
            template.set_mode('transformers')
            for start in range(0, len(rows), batch_size):
                items = rows[start:start + batch_size]
                encoded = [template.encode(dict(messages=[row['messages'][0]], audios=row['audios']))
                           for row in items]
                batch = {k: v.to(model.device) for k, v in template.data_collator(encoded).items()
                         if k in ('input_ids', 'attention_mask', 'input_features', 'feature_attention_mask')}
                seen.clear()
                seq = template.generate(model, **batch, max_new_tokens=1, do_sample=False, num_beams=1,
                                        use_model_defaults=False,
                                        prefix_allowed_tokens_fn=lambda i, _: token_ids[:items[i]['n']])
                if len(seen) != 1 or seen[0].shape[0] != len(items):
                    raise ValueError('Expected one pass through the output head per batch')
                for k, item in enumerate(items):
                    chunks.append(seen[0][k, -1, token_ids[:item['n']]].float().cpu().numpy())
    finally:
        handle.remove()

    width = max(len(c) for c in chunks)
    padded = np.full((len(chunks), width), -np.inf, dtype=np.float32)
    for k, chunk in enumerate(chunks):
        padded[k, :len(chunk)] = chunk
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / 'logits.npy', padded)
    d2.write_jsonl(target / 'rows.jsonl', [dict(id=r['id'], shift=r['shift'], gold=r['gold'],
                                                n=r['n'], question=r['question'],
                                                options=r['options']) for r in rows])
    d2.write(target / 'done.json', dict(rows=len(rows), batch_size=batch_size,
                                         manifest_sha256=d2.digest(Path(folder) / 'mcq.jsonl')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--method', required=True, choices=('frozen-base', 'ta-lora'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    extract(args.folder, args.model_config, args.method, args.checkpoint, args.arm,
            args.output, batch_size=args.batch_size)
