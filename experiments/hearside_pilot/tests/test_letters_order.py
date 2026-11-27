import numpy as np
import pytest

from experiments.hearside_pilot import letters_order

CLASSES = 'ABCD'


def rows(n=2, orders=('ABCD', 'BCDA', 'CDAB', 'DABC')):
    """`n` utterances, each asked under four option orders."""
    out = []
    for u in range(n):
        for r, order in enumerate(orders):
            out.append(dict(id=f'u{u}#r{r}', label='A', rotation=r, option_order=list(order), cluster=f'c{u}', answer='A'))
    return out


def logits_for(answers):
    """A 4x4 matrix whose argmax is the given letter for each row."""
    out = np.zeros((len(answers), 4), dtype=np.float32)
    for i, letter in enumerate(answers):
        out[i, CLASSES.index(letter)] = 10
    return out


def test_the_same_letter_and_the_same_emotion_are_measured_over_pairs():
    # One utterance: the answers are A, B, C, D under the four orders, so no pair agrees on the
    # letter; the emotions behind them do not agree either.
    r = rows(1)
    result = letters_order.consistency(logits_for(['A', 'B', 'C', 'D']), r)
    assert result['utterances'] == 1 and result['pairs'] == 6
    assert result['same_letter'] == 0.0

    # One utterance: the same letter every time, but four different emotions.
    r = rows(1)
    result = letters_order.consistency(logits_for(['A', 'A', 'A', 'A']), r)
    assert result['same_letter'] == 1.0 and result['same_emotion'] == 0.0

    # One utterance: the same emotion every time, reached through different letters.
    r = rows(1, orders=('ABCD', 'BCDA', 'CDAB', 'DABC'))
    result = letters_order.consistency(logits_for(['A', 'B', 'C', 'D']), r)
    assert result['same_letter'] == 0.0


def test_an_utterance_missing_an_order_is_refused():
    r = rows(2)[:3]
    with pytest.raises(ValueError, match='all four option orders'):
        letters_order.consistency(logits_for(['A'] * 3), r)
