"""Order consistency of the surrogate and the adapters: the same letter, or the same emotion.

The surrogate in letters.py is the frozen base with one constant per letter. If the last four
layers had learned an output bias over letters, the surrogate should not only reproduce their gain
in the training order (which letters.py measures) but also reproduce how their answers move when
the option order changes. This script computes that: over the utterances that were asked under all
four option orders, the share of answer pairs that give the same letter, and the share that give
the same emotion.

The definition of a pair is the one of conflict.py, which measured the same thing on the scored
predictions before the logits were extracted: of the six pairs of orders of one utterance, how many
give the same letter, and how many give the same emotion.

Run from the repository root, with the extracted logits of a letters run:

    python -m experiments.hearside_pilot.letters_order --root hearside-letters-20261125-170918/logits

The numbers below are the ones written into research/letter-bias-20261125.md and
experiments/hearside_pilot/results/letters-20261125-170918/NOTES.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import letters
from experiments.trr_diagnostics import d2

CLASSES = letters.CLASSES
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def consistency(logits, rows):
    """The share of answer pairs that give the same letter, and the same emotion."""
    by_utterance = {}
    for index, row in enumerate(rows):
        by_utterance.setdefault(row['id'].rsplit('#', 1)[0], {})[row['rotation']] = index
    complete = [v for v in by_utterance.values() if sorted(v) == [0, 1, 2, 3]]
    if not complete:
        raise ValueError('No utterance was asked under all four option orders')
    chosen = logits.argmax(1)
    same_letter, same_emotion, pairs = 0, 0, 0
    for rotations in complete:
        letters_ = [CLASSES[chosen[rotations[k]]] for k in range(4)]
        emotions = [rows[rotations[k]]['option_order'][CLASSES.index(letters_[k])] for k in range(4)]
        for i, j in PAIRS:
            pairs += 1
            same_letter += letters_[i] == letters_[j]
            same_emotion += emotions[i] == emotions[j]
    return dict(utterances=len(complete), pairs=pairs, same_letter=same_letter / pairs, same_emotion=same_emotion / pairs)


def main(root):
    root = Path(root)
    train_logits, train_rows = letters.load(root / 'frozen-base' / 'train')
    bias = letters.fit_bias(train_logits, np.array([CLASSES.index(r['label']) for r in train_rows]))
    result = dict(letter_bias=dict(zip(CLASSES, map(float, bias))), corpora={})
    for split in letters.CORPORA:
        base_logits, rows = letters.load(root / 'frozen-base' / split)
        entry = {'frozen-base': consistency(base_logits, rows),
                 'surrogate': consistency(base_logits + bias, rows)}
        for model in sorted(p.name for p in root.iterdir() if p.name != 'frozen-base' and (p / split / 'done.json').exists()):
            logits, other = letters.load(root / model / split)
            if [r['id'] for r in other] != [r['id'] for r in rows]:
                raise ValueError(f'{model}/{split}: rows differ from the frozen base')
            entry[model] = consistency(logits, other)
        result['corpora'][split] = entry
    return result


def render(result):
    lines = ['| model | same letter | same emotion |', '|---|---:|---:|']
    for split, entry in result['corpora'].items():
        lines += ['', f'## {split}', '', '| model | same letter | same emotion |', '|---|---:|---:|']
        for model, row in entry.items():
            lines.append(f"| {model} | {100 * row['same_letter']:.1f}% | {100 * row['same_emotion']:.1f}% |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = main(args.root)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'order_consistency.json', result)
        (args.output / 'order_consistency.md').write_text(render(result))
    print(render(result))
