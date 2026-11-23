"""Training and selection manifests for the hear-side pilot.

Sessions 1-2 train, Session 3 selects the checkpoint, Session 4 is only ever evaluated and
Session 5 is not read. Two versions of each manifest are written from the same utterances in
the same order:

fixed     the original question, options always in the order A angry, B happy, C sad, D neutral
shuffled  each utterance asked under one of the four cyclic option orders, balanced within class
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotate(row, shift, classes, template, options):
    """The same utterance with the options moved `shift` places: letter j now names classes[j + shift]."""
    order = [classes[(j + shift) % len(classes)] for j in range(len(classes))]
    label = row['messages'][1]['content']
    question = template.format(*(options[c] for c in order))
    # Column names that ms-swift's dataset loader does not interpret ("label" is one it does).
    return dict(row, option_shift=shift, emotion=label,
                messages=[dict(role='user', content='<audio>' + question),
                          dict(role='assistant', content=classes[order.index(label)])])


def balanced_shifts(rows, classes, seed):
    """One option order per row; within every class each order is used equally often (to within one)."""
    by_class = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row['messages'][1]['content']].append(index)
    shifts, rng = {}, Random(seed)
    for label in classes:
        members = by_class[label]
        rng.shuffle(members)
        for place, index in enumerate(members):
            shifts[index] = place % len(classes)
    return [shifts[i] for i in range(len(rows))]


def build(source, output, cfg, *, train_sessions=(1, 2), dev_sessions=(3,), seed=42):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    classes, question = cfg['classes'], cfg['prompts']['question']
    template, options = cfg['prompts']['question_template'], cfg['prompts']['options']
    if template.format(*(options[c] for c in classes)) != question:
        raise ValueError('question_template with the options in class order must reproduce the question')
    rows = read_jsonl(source)
    for row in rows:
        user, answer = row['messages']
        if user != dict(role='user', content='<audio>' + question) or answer['content'] not in classes or len(row['audios']) != 1:
            raise ValueError(f"{row['id']}: row does not match the declared question and classes")
    if {row['session'] for row in rows} & {4, 5}:
        raise ValueError('The source manifest must not contain the evaluation sessions')
    output.mkdir(parents=True)
    summary = {}
    for split, sessions in (('train', train_sessions), ('dev', dev_sessions)):
        part = [row for row in rows if row['session'] in sessions]
        shifts = balanced_shifts(part, classes, seed)
        versions = {'fixed': part, 'shuffled': [rotate(row, shift, classes, template, options) for row, shift in zip(part, shifts)]}
        for name, content in versions.items():
            path = output / f'{split}-{name}.jsonl'
            write_jsonl(path, content)
            summary[f'{split}-{name}'] = dict(
                sha256=digest(path), utterances=len(content), sessions=list(sessions),
                speakers=sorted({row['speaker'] for row in content}),
                classes=dict(sorted(Counter(row.get('emotion', row['messages'][1]['content']) for row in content).items())),
                answer_letters=dict(sorted(Counter(row['messages'][1]['content'] for row in content).items())))
    if set(summary['train-fixed']['speakers']) & set(summary['dev-fixed']['speakers']):
        raise ValueError('Training and selection speakers overlap')
    summary['source'] = dict(path=str(Path(source).resolve()), sha256=digest(source), seed=seed)
    (output / 'splits.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='IEMOCAP train.jsonl (Sessions 1-3)')
    parser.add_argument('--protocol', type=Path, required=True, help='Evaluation config holding the question and options')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, json.loads(args.protocol.read_text())), indent=2))
