import numpy as np
import pytest

from experiments.hearside_pilot import letters
from experiments.trr_diagnostics import d2

SKEW = np.array([1.5, 0.0, -1.5, 0.0])                    # the toy base leans to the first emotion and away from the third


def write(folder, logits, rows):
    folder.mkdir(parents=True)
    np.save(folder / 'logits.npy', logits.astype(np.float32))
    d2.write_jsonl(folder / 'rows.jsonl', rows)
    d2.write(folder / 'done.json', dict(rows=len(rows)))


def evidence(n, seed, rotations, sharpness=1.0, letter_shift=None):
    """Letter logits of a model that hears the emotion with noise, leans by SKEW per emotion, and reads the options."""
    rng = np.random.default_rng(seed)
    rows, logits = [], []
    for i in range(n):
        label = int(rng.integers(4))
        heard = (np.eye(4)[label] * 1.8 + rng.normal(size=4)) * sharpness + SKEW                 # per emotion
        for k in rotations:
            order = [letters.CLASSES[(j + k) % 4] for j in range(4)]
            rows.append(dict(id=f'u{seed}-{i}#r{k}', label=letters.CLASSES[label], rotation=k, option_order=order, cluster=f'c{i % 12}'))
            logits.append(heard[[letters.CLASSES.index(c) for c in order]] + (0 if letter_shift is None else letter_shift))
    return np.array(logits), rows


def build(root, **last4):
    train, train_rows = evidence(600, 0, [0])
    write(root / 'frozen-base' / 'train', train, train_rows)
    for seed, split in enumerate(letters.CORPORA, 1):
        base, rows = evidence(300, seed, range(4))
        write(root / 'frozen-base' / split, base, rows)
        for model in letters.crosscorpus.FAMILIES['A-last4']:
            write(root / model / split, evidence(300, seed, range(4), **last4)[0], rows)


def test_fitted_constants_undo_a_lean_in_the_training_order():
    logits, rows = evidence(2000, 3, [0])
    bias = letters.fit_bias(logits, np.array([letters.CLASSES.index(r['label']) for r in rows]))
    assert np.allclose(bias, -SKEW, atol=0.25) and abs(bias.sum()) < 1e-9


def test_an_arm_that_only_shifts_the_letters_is_read_as_a_letter_bias(tmp_path):
    build(tmp_path, letter_shift=-SKEW)              # right for the training order, misplaced under the others
    result = letters.analyze(tmp_path, resamples=300)
    entry = result['corpora']['session4']
    assert entry['reading'] == 'mostly a letter bias' and entry['share_of_gain_recovered']['value'] > 0.8
    assert entry['last4_minus_base']['training_order']['delta_pp'] > 5 > 0 > entry['last4_minus_base']['other_orders']['delta_pp']
    assert entry['same_letter_as_last4']['surrogate'] > 0.9 > entry['same_letter_as_last4']['frozen-base']
    assert result['letter_bias']['A'] < -1 and result['letter_bias']['C'] > 1 and 'surrogate' in letters.render(result)


def test_an_arm_that_hears_better_is_not(tmp_path):
    build(tmp_path, sharpness=4.0)
    entry = letters.analyze(tmp_path, resamples=300)['corpora']['session4']
    assert entry['reading'] != 'mostly a letter bias' and entry['last4_minus_base']['other_orders']['delta_pp'] > 5


def test_recall_follows_the_option_order():
    rows = [dict(id=f'{i}', label='B', rotation=k, option_order=[letters.CLASSES[(j + k) % 4] for j in range(4)], cluster='c') for i, k in enumerate(range(4))]
    rows += [dict(id=f'x{c}', label=c, rotation=0, option_order=list('ABCD'), cluster='d') for c in 'ACD']
    corpus = letters.Corpus(rows, 10, 0)
    chosen = np.array([1, 0, 3, 2, 0, 2, 3])                       # B is the letter B, A, D, C under the four orders
    assert corpus.right(chosen).tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert corpus.right(np.array([1, 1, 1, 1, 0, 2, 3])).tolist() == [1, 0, 0, 0, 1, 1, 1]
