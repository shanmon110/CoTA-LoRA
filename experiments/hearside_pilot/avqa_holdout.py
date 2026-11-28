"""Build the held-out AVQA evaluation manifest, for the same-distribution measurement.

The AQA arms are trained on AVQA's `pilot-01/train.jsonl` and their checkpoint is chosen on
`pilot-01/dev.jsonl`.  Neither is a fair place to measure the same-distribution deficit: the
dev split was used for early stopping, and the train split was used for the weights.  So this
module builds a third split, held out from both.

The pool is AVQA's own `official_train.jsonl` -- 40,425 questions over 35,353 source videos --
minus the pilot's train and dev questions.  What is left is 38,121 questions that no arm was
trained on and no checkpoint was selected on.

The split is by SOURCE VIDEO, not by question: the same video can carry several questions, and
splitting by question would put questions about the same video on both sides.  The videos are
assigned to the evaluation fold or to the hold-out fold by a hash of the video name, so the
assignment is reproducible and does not depend on the order of the file.

The evaluation takes a sample, not all 38,121 questions: the question is a cheap inference and
the arms are scored on the same sample, so a sample of a few thousand questions is enough to
measure the deficit with a tight interval, and it keeps the pass inside an hour.

The manifest has the same shape as the MMAU one that `aqa.py build` writes, so the same
evaluation code scores it and the two numbers are comparable.

Run from the repository root:

    python3 -m experiments.hearside_pilot.avqa_holdout --output <folder> [--size 3000]
"""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.trr_diagnostics import d2

PILOT = Path('/root/autodl-tmp/r1-aqa/data/pilot-01')
FULL = Path('/root/autodl-tmp/r1-aqa/avqa/full/official_train.jsonl')


def fold_of(video, folds=10):
    """A reproducible fold for a video, from a hash of its name."""
    digest = hashlib.sha256(video.encode()).hexdigest()
    return int(digest[:8], 16) % folds


def build(pilot=PILOT, full=FULL, output=None, *, size=3000, folds=10, evaluation_fold=0, seed=42):
    """Write `<output>/holdout.jsonl` and `<output>/holdout.json`."""
    used = set()
    for name in ('train', 'dev'):
        path = Path(pilot) / f'{name}.jsonl'
        if path.exists():
            used |= {json.loads(line)['id'] for line in open(path)}
    pool = [json.loads(line) for line in open(full)]
    held = [r for r in pool if r['id'] not in used]

    # the evaluation fold, by source video, so no video appears on both sides
    evaluation = [r for r in held if fold_of(r['source_id'], folds) == evaluation_fold]
    reserve = [r for r in held if fold_of(r['source_id'], folds) != evaluation_fold]

    # a reproducible sample of the evaluation fold, stratified by the gold letter so the
    # sample's letter distribution matches the fold's
    import random
    rng = random.Random(seed)
    by_letter = {}
    for r in evaluation:
        by_letter.setdefault(r['messages'][1]['content'], []).append(r)
    for letter in by_letter:
        rng.shuffle(by_letter[letter])
    total = sum(len(v) for v in by_letter.values())
    sample = []
    for letter, items in sorted(by_letter.items()):
        take = round(size * len(items) / total)
        sample += items[:take]
    rng.shuffle(sample)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    d2.write_jsonl(output / 'holdout.jsonl', sample)
    info = dict(
        source=str(Path(full).resolve()), source_sha256=d2.digest(full),
        pilot=str(Path(pilot).resolve()),
        pilot_questions_used=len(used),
        pool_questions=len(pool), held_out_pool=len(held),
        folds=folds, evaluation_fold=evaluation_fold,
        evaluation_fold_questions=len(evaluation), reserve_questions=len(reserve),
        sample=len(sample), seed=seed,
        sample_videos=len({r['source_id'] for r in sample}),
        reserve_videos=len({r['source_id'] for r in reserve}),
        sample_videos_in_reserve=len({r['source_id'] for r in sample}
                                      & {r['source_id'] for r in reserve}),
        manifest_sha256=d2.digest(output / 'holdout.jsonl'))
    d2.write(output / 'holdout.json', info)
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=3000)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--evaluation-fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, size=args.size, folds=args.folds,
                           evaluation_fold=args.evaluation_fold, seed=args.seed), indent=2))
