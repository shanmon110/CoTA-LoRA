"""Generate text with the Swift Omni template and compute corpus-level English WER."""
import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from tempfile import TemporaryDirectory
from pathlib import Path

import torch


def words(text):
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'").replace('‘', "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def edit_distance(a, b):
    costs = list(range(len(b) + 1))
    for i, ref in enumerate(a, 1):
        prev, costs[0] = costs[0], i
        for j, hyp in enumerate(b, 1):
            prev, costs[j] = costs[j], min(costs[j] + 1, costs[j-1] + 1, prev + (ref != hyp))
    return costs[-1]


def continuations(output, prompt, eos, cap):
    if (output.ndim != 2 or output.size(0) != prompt.size(0) or
            not torch.equal(output[:, :prompt.size(1)], prompt)):
        raise ValueError('Generated sequences lack the exact padded prompt prefix')
    for row in output[:, prompt.size(1):].tolist():
        if not 0 < len(row) <= cap:
            raise ValueError('Invalid generated width')
        end = next((i for i, token in enumerate(row) if token in eos), None)
        if end is None and len(row) != cap:
            raise ValueError('Generation stopped early without EOS')
        yield (row if end is None else row[:end]), end is None


def score(model, template, manifest, generation, predictions):
    manifest, predictions = Path(manifest), Path(predictions)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if not rows or len(set(r['id'] for r in rows)) != len(rows):
        raise ValueError('Empty manifest or duplicate IDs')
    if generation['do_sample'] or generation['num_beams'] != 1:
        raise ValueError('The shared ASR protocol requires greedy decoding')
    batch_size, cap = generation['batch_size'], generation['max_new_tokens']
    tokenizer = template.tokenizer
    eos = generation.get('eos_token_id', [tokenizer.eos_token_id])
    eos = [eos] if isinstance(eos, int) else eos
    directory = predictions.parent
    directory.mkdir(parents=True, exist_ok=True)
    stage = directory / (predictions.name + f'.{os.getpid()}.tmp')
    training = model.training
    model.eval()
    start = time.monotonic()
    errors = num_words = capped = 0
    try:
        with template.generate_context(), torch.inference_mode(), stage.open('x') as stream:
            template.set_mode('transformers')
            for index in range(0, len(rows), batch_size):
                chunk = rows[index:index+batch_size]
                encoded = []
                for r in chunk:
                    msgs = r['messages']
                    if [m['role'] for m in msgs] != ['user', 'assistant']:
                        raise ValueError(f"Invalid ASR item {r['id']}")
                    encoded.append(template.encode({'messages': [dict(msgs[0])], 'audios': list(r['audios'])}))
                inputs = template.data_collator(encoded)
                inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)
                          and k in {'input_ids','attention_mask','input_features','feature_attention_mask'}}
                ids, valid = inputs['input_ids'], inputs['attention_mask'].bool()
                if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
                    raise ValueError('Left padding required for batched generation')
                audio_id = model.config.thinker_config.audio_token_id
                frames = inputs['feature_attention_mask'].sum(-1)
                expected = ((frames + 1) // 2) // 2
                if (expected < 1).any() or not torch.equal((ids.eq(audio_id) & valid).sum(-1), expected):
                    raise ValueError('Expanded audio tokens do not match real feature frames')
                generated = template.generate(model, **inputs, max_new_tokens=cap, num_beams=1,
                                              do_sample=False, use_model_defaults=False, use_cache=True,
                                              eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                                              return_dict_in_generate=False)
                for r, (tokens, truncated) in zip(chunk, continuations(generated, ids, eos, cap), strict=True):
                    hypothesis = tokenizer.decode(tokens, skip_special_tokens=True)
                    ref = r['messages'][1]['content']
                    ref_words = words(ref)
                    dist = edit_distance(ref_words, words(hypothesis))
                    num_words += len(ref_words)
                    errors += dist
                    capped += int(truncated)
                    print(json.dumps(dict(id=r['id'], reference=ref, prediction=hypothesis, word_errors=dist,
                                          reference_words=len(ref_words), capped_without_eos=truncated)), file=stream)
        if not num_words:
            raise ValueError('Undefined WER for a wordless reference corpus')
        stage.replace(predictions)
    finally:
        model.train(training)
        stage.unlink(missing_ok=True)
    return dict(wer=errors / num_words, word_errors=errors, reference_words=num_words, utterances=len(rows),
                capped_without_eos=capped, capped_without_eos_rate=capped / len(rows),
                generation=generation, seconds=time.monotonic()-start,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


if __name__ == '__main__':
    from .train import load_best
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--method', choices=['lora', 'ta-lora'], required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    model, template = load_best(cfg, args.method, args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp)/'eval'
        torch.cuda.reset_peak_memory_stats()
        result = score(model, template, args.manifest, cfg['generation'], stage/'predictions.jsonl')
        result.update(checkpoint=str(args.checkpoint.resolve()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (stage/'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        stage.rename(args.output)
