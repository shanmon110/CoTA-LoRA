"""Greedy one-token multiple-choice evaluation using only the options offered on each item."""
import argparse
import json
from collections import Counter
from pathlib import Path
from string import ascii_uppercase
import time

import torch

from experiments.ms_swift_librispeech.train import digest, load_best, save_json
from .prompt import format_question


def correct_letters(row):
    ref = row['messages'][1]['content']
    options = row['choices']
    answer = options[ascii_uppercase.index(ref)]
    equal = [letter for letter, text in zip(ascii_uppercase, options) if text == answer]
    if row.get('correct_letters', equal) != equal:
        raise ValueError(f"{row['id']}: equivalent letters differ from gold option text")
    return equal


def validate_row(row, *, allow_empty_options=False):
    prompt = format_question(row['question'], row['choices'], allow_empty_options=allow_empty_options)
    messages = row['messages']
    if len(messages) != 2 or messages[0] != {'role': 'user', 'content': prompt}:
        raise ValueError(f"{row['id']}: prompt differs from question and ordered choices")
    ref = messages[1]['content']
    if messages[1]['role'] != 'assistant' or len(ref) != 1 or ref not in ascii_uppercase[:len(row['choices'])]:
        raise ValueError(f"{row['id']}: invalid answer letter {ref!r}")
    correct_letters(row)
    if prompt.count('<audio>') != 1 or len(row['audios']) != 1 or not row['id']:
        raise ValueError(f"{row['id']}: expected one audio input and a question ID")
    return ref


def read_rows(manifest, *, allow_empty_options=False):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Expected nonempty manifest with unique question IDs: {manifest}')
    for row in rows:
        validate_row(row, allow_empty_options=allow_empty_options)
    return rows


def check_encoded_prompt(template, ids, text, feature_mask):
    if not text.startswith('<audio>\n') or not torch.as_tensor(feature_mask).any():
        raise ValueError('Expected a question with nonempty audio features')
    token_ids = torch.as_tensor(ids)
    audio_id = template.tokenizer.convert_tokens_to_ids(template.processor.audio_token)
    if not token_ids.eq(audio_id).any():
        raise ValueError('Audio placeholder missing from encoded prompt')
    decoded = template.tokenizer.decode(token_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    question_and_options = text.removeprefix('<audio>\n')
    if question_and_options not in decoded:
        raise ValueError(f'Question or choices changed during Swift tokenization: {question_and_options[:100]!r}')


def inspect_training_prompt(dataset, template):
    row = dataset.dataset[0]
    encoded = dataset[0]
    labels = torch.as_tensor(encoded['labels'])
    first_target = labels.ne(-100).nonzero()[0].item()
    ids = torch.as_tensor(encoded['input_ids'])
    check_encoded_prompt(template, ids[:first_target], row['messages'][0]['content'], encoded['feature_attention_mask'])
    answer = row['messages'][-1]['content']
    if template.tokenizer.encode(answer, add_special_tokens=False) != [ids[first_target].item()]:
        raise ValueError('First supervised token differs from the gold answer')
    return dict(encoded_question_and_choices_verified=True, first_supervised_letter=answer)


def summarize(predictions):
    if not predictions:
        raise ValueError('No questions were scored')
    result = {'utterances': len(predictions), 'correct': sum(r['is_correct'] for r in predictions)}
    result['accuracy'] = result['correct'] / len(predictions)
    for group in ('domain', 'category', 'question_relation', 'question_type'):
        total, correct = Counter(), Counter()
        for row in predictions:
            key = row.get(group)
            if key:
                if group == 'category' and row.get('domain'):
                    key = f"{row['domain']}/{key}"
                total[key] += 1
                correct[key] += int(row['is_correct'])
        result['by_' + group] = {key: dict(correct=correct[key], total=n, accuracy=correct[key]/n)
                                 for key, n in sorted(total.items())}
        if total:
            result['macro_' + group + '_accuracy'] = sum(correct[k] / n for k, n in total.items()) / len(total)
    return result


def score(model, template, manifest, generation, predictions, *, allow_empty_options=False):
    rows = read_rows(manifest, allow_empty_options=allow_empty_options)
    size = generation['batch_size']
    if not isinstance(size, int) or size < 1:
        raise ValueError('Expected a positive scoring batch size')
    letters = ascii_uppercase[:max(len(r['choices']) for r in rows)]
    letter_ids = [template.tokenizer.encode(c, add_special_tokens=False) for c in letters]
    if any(len(t) != 1 for t in letter_ids) or len({t[0] for t in letter_ids}) != len(letters):
        raise ValueError('Tokenizer must encode all option letters as distinct single tokens')
    tokens = [t[0] for t in letter_ids]
    records = []
    predictions = Path(predictions)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    tmp = predictions.with_suffix('.tmp')
    with torch.inference_mode(), template.generate_context(), tmp.open('w') as stream:
        template.set_mode('transformers')
        for start in range(0, len(rows), size):
            items = rows[start:start + size]
            encoded = []
            for row in items:
                prompt = row['messages'][0]
                example = template.encode(dict(messages=[prompt.copy()], audios=list(row['audios'])))
                check_encoded_prompt(template, example['input_ids'], prompt['content'], example['feature_attention_mask'])
                encoded.append(example)
            batch = {k: v.to(model.device) for k, v in template.data_collator(encoded).items()
                     if k in ('input_ids', 'attention_mask', 'input_features', 'feature_attention_mask')}
            if not batch['attention_mask'][:, -1].bool().all():
                raise ValueError('Expected left-padded prompts')
            allowed = [tokens[:len(row['choices'])] for row in items]
            seq = template.generate(model, **batch, max_new_tokens=1, do_sample=False, num_beams=1,
                                    use_model_defaults=False, prefix_allowed_tokens_fn=lambda i, _: allowed[i])
            if not torch.equal(seq[:, :-1], batch['input_ids']):
                raise ValueError('Generation prefix changed')
            for row, token in zip(items, seq[:, -1].tolist(), strict=True):
                index = tokens.index(token)
                if index >= len(row['choices']):
                    raise ValueError(f"{row['id']}: predicted an unavailable option")
                record = dict(id=row['id'], reference=row['messages'][1]['content'], prediction=letters[index],
                              answer_text=row['choices'][index])
                record['correct_letters'] = correct_letters(row)
                record['is_correct'] = record['prediction'] in record['correct_letters']
                record.update({k: row[k] for k in ('domain', 'category', 'difficulty', 'question_relation', 'question_type') if row.get(k)})
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), file=stream)
    result = summarize(records)
    tmp.replace(predictions)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model, template = load_best(config, args.method, args.checkpoint)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metrics = score(model, template, args.manifest, config['generation'], args.output/'predictions.jsonl')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    seconds = time.perf_counter() - start
    from .tuner_suite import checkpoint_identity
    metrics.update(eval_seconds=seconds, manifest_sha256=digest(args.manifest),
                   **checkpoint_identity(args.checkpoint, args.method), method=args.method)
    save_json(args.output/'metrics.json', metrics)
