"""P0: is the AQA negative transfer cross-corpus or cross-task?

The AQA arms are trained on AVQA and scored on MMAU Test-mini, and every arm is below the frozen
base there.  Two explanations remain open:

  cross-corpus    the transfer from AVQA to MMAU is what costs the accuracy, so the layer-band
                  question was never really tested on the second task;
  cross-task      the AVQA task and the MMAU task are not comparable, so the second task cannot
                  answer the layer-band question at all.

The test is the same arms scored on AVQA, the split they were trained on.  Same corpus, same
task, same distribution: any remaining deficit is the adapter's own.

The pre-declared reading (research/notes-20261127-diagnosis-and-plan.md, P0):

  if the adapters are not below the frozen base on AVQA, the negative transfer is CROSS-CORPUS
  and the AQA evidence neither supports nor refutes the layer-band claim;
  if they are below the frozen base on AVQA too, the negative transfer is CROSS-TASK and the
  second task is not comparable.

Both readings are reported with the interval over the question clusters, so a small deficit that
cannot be told apart from zero is reported as such and not as a finding.

Run from the repository root:

    python3 -I research/tools/avqa_vs_mmau.py --avqa <folder> --mmau <folder>
"""
import argparse
import collections
import json
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DOMAINS = ('music', 'sound', 'speech')
OPTION_RE = re.compile(r'^([A-H])\.\s*(.*)$')
BOOT = 10000
SEED = 42


def read_predictions(folder, task='mcq'):
    path = pathlib.Path(folder) / 'runs'
    out = {}
    for arm in sorted(p.name for p in path.iterdir()):
        file = path / arm / task / 'predictions.jsonl'
        if file.exists():
            out[arm] = [json.loads(line) for line in open(file)]
    return out


def cluster_of(row):
    """The audio file is the cluster: a question with its audio is the unit of independence."""
    return row['id'].rsplit('#s', 1)[0]


def accuracy(rows, mask=None):
    if mask is None:
        mask = np.ones(len(rows), bool)
    if not mask.any():
        return float('nan')
    return 100 * float(np.mean([r['is_correct'] for r, m in zip(rows, mask) if m]))


def bootstrap_gap(arm_rows, base_rows, *, resamples=BOOT, seed=SEED):
    """The gap of the arm over the base, with a cluster bootstrap over the questions.

    The cluster is the question, so a question and its four option orders are resampled together:
    the four rows of one question are not independent observations.
    """
    arm_by = {r['id']: r for r in arm_rows}
    base_by = {r['id']: r for r in base_rows}
    ids = sorted(set(arm_by) & set(base_by))
    if not ids:
        return float('nan'), (float('nan'), float('nan'))
    clusters = sorted({cluster_of(base_by[i]) for i in ids})
    index = {c: k for k, c in enumerate(clusters)}
    # per cluster: how many rows, how many the arm got right, how many the base got right
    n_rows = np.zeros(len(clusters))
    n_arm = np.zeros(len(clusters))
    n_base = np.zeros(len(clusters))
    for i in ids:
        k = index[cluster_of(base_by[i])]
        n_rows[k] += 1
        n_arm[k] += bool(arm_by[i]['is_correct'])
        n_base[k] += bool(base_by[i]['is_correct'])
    point = 100 * (n_arm.sum() - n_base.sum()) / n_rows.sum()
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(clusters), (resamples, len(clusters)))
    total = n_rows[picked].sum(axis=1)
    gaps = 100 * (n_arm[picked].sum(axis=1) - n_base[picked].sum(axis=1)) / np.maximum(total, 1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(point), (float(lo), float(hi))


def main(avqa, mmau):
    print('# P0: is the AQA negative transfer cross-corpus or cross-task?\n')
    print('The arms are trained on AVQA and scored on MMAU Test-mini.  Here they are scored on '
          'AVQA, the split they were trained on: same corpus, same task, same distribution.\n')
    for label, folder in (('AVQA (the training distribution)', avqa), ('MMAU Test-mini', mmau)):
        arms = read_predictions(folder)
        if 'frozen-base' not in arms:
            print(f'-- {label}: no frozen base, skipped')
            continue
        base = arms['frozen-base']
        print(f'## {label}\n')
        print(f'| arm | accuracy | gap over the frozen base [95% CI] |')
        print('|---|---:|---|')
        for arm in sorted(arms):
            rows = arms[arm]
            acc = accuracy(rows)
            if arm == 'frozen-base':
                print(f'| {arm} | {acc:.2f}% | -- |')
                continue
            point, (lo, hi) = bootstrap_gap(rows, base)
            print(f'| {arm} | {acc:.2f}% | {point:+.2f} pp [{lo:+.2f}, {hi:+.2f}] |')
        print()

    print('## The pre-declared reading\n')
    arms = read_predictions(avqa)
    if 'frozen-base' not in arms:
        print('No AVQA numbers; the reading cannot be made.')
        return
    base = arms['frozen-base']
    below = []
    for arm in sorted(arms):
        if arm == 'frozen-base':
            continue
        point, (lo, hi) = bootstrap_gap(arms[arm], base)
        if hi < 0:
            below.append(arm)
    if below:
        print(f'On AVQA the arms that are below the frozen base with an interval that excludes '
              f'zero are: {", ".join(below)}.')
        print()
        print('Reading: the negative transfer is **cross-task** -- the adapters are below the '
              'frozen base even where the distribution is the one they were trained on, so the '
              'MMAU result is not measuring the transfer from AVQA to MMAU.  The second task is '
              'not comparable and chapter 5 cannot use it to say anything about the layer bands.')
    else:
        print('On AVQA no arm is below the frozen base with an interval that excludes zero.')
        print()
        print('Reading: the negative transfer is **cross-corpus** -- the adapters are not below '
              'the frozen base where the distribution is the one they were trained on, so what '
            'the MMAU result measures is the transfer from AVQA to MMAU.  The AQA evidence then '
            'neither supports nor refutes the layer-band claim, because the layer-band question '
            'was never tested on the second task.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--avqa', type=pathlib.Path,
                        default=REPO / 'hearside-avqa-same-dist' / 'avqa')
    parser.add_argument('--mmau', type=pathlib.Path,
                        default=REPO / 'hearside-aqa-20261125-224035' / 'mmau')
    args = parser.parse_args()
    main(args.avqa, args.mmau)
