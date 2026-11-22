import numpy as np
import pytest
from scipy.stats import binomtest

from experiments.trr_diagnostics import stats

CLASSES = ['A', 'B', 'C', 'D']


def test_classification_reports_accuracy_uar_and_confusion():
    rows = [('A', 'A'), ('A', 'B'), ('B', 'B'), ('C', 'C'), ('D', 'A'), ('D', 'D'), ('D', 'D'), ('D', 'D')]
    out = stats.classification(rows, CLASSES)
    assert out['utterances'] == 8 and out['accuracy'] == pytest.approx(6 / 8)
    assert out['uar'] == pytest.approx((0.5 + 1 + 1 + 0.75) / 4)
    assert out['confusion_matrix'][3] == [1, 0, 0, 3]
    with pytest.raises(ValueError):
        stats.classification(rows[:3], CLASSES)


@pytest.mark.parametrize('b,c', [(0, 0), (0, 5), (5, 0), (3, 3), (2, 9), (40, 61), (120, 80)])
def test_mcnemar_matches_the_exact_binomial_test(b, c):
    expected = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    assert stats.mcnemar_exact(b, c) == pytest.approx(expected)


def _items(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = [CLASSES[i % 4] for i in range(n)]
    clusters = [f'd{i % 20}' for i in range(n)]
    reference = [y if rng.random() < 0.7 else CLASSES[(CLASSES.index(y) + 1) % 4] for y in labels]
    return labels, reference, clusters


def test_identical_predictions_have_zero_delta_and_zero_width():
    labels, reference, clusters = _items()
    out = stats.paired_contrast(labels, reference, reference, clusters, CLASSES, resamples=500)
    assert out['delta_uar_pp'] == 0 and out['uar_interval_95_pp'] == [0.0, 0.0]
    assert out['prediction_changed'] == 0 and out['mcnemar_exact_p'] == 1.0


def test_contrast_point_estimates_equal_direct_computation():
    labels, reference, clusters = _items()
    candidate = list(reference)
    for i in range(0, 400, 5):  # break every fifth item that the reference got right
        if candidate[i] == labels[i]:
            candidate[i] = CLASSES[(CLASSES.index(labels[i]) + 2) % 4]
    out = stats.paired_contrast(labels, candidate, reference, clusters, CLASSES, resamples=2000)
    direct_c = stats.classification(zip(labels, candidate), CLASSES)
    direct_r = stats.classification(zip(labels, reference), CLASSES)
    assert out['delta_uar_pp'] == pytest.approx(100 * (direct_c['uar'] - direct_r['uar']))
    assert out['delta_accuracy_pp'] == pytest.approx(100 * (direct_c['accuracy'] - direct_r['accuracy']))
    assert out['candidate_only_correct'] == 0 and out['reference_only_correct'] == out['prediction_changed'] > 0
    low, high = out['uar_interval_95_pp']
    assert low < out['delta_uar_pp'] < high < 0
    again = stats.paired_contrast(labels, candidate, reference, clusters, CLASSES, resamples=2000)
    assert again == out  # fixed seed


def test_clustered_errors_widen_the_interval():
    labels, reference, _ = _items()
    candidate = list(reference)
    for i in range(0, 80):  # all damage falls in the first 80 items
        candidate[i] = CLASSES[(CLASSES.index(labels[i]) + 2) % 4]
    spread = [f'd{i % 20}' for i in range(400)]
    lumped = [f'd{i // 20}' for i in range(400)]
    a = stats.paired_contrast(labels, candidate, reference, spread, CLASSES, resamples=2000)
    b = stats.paired_contrast(labels, candidate, reference, lumped, CLASSES, resamples=2000)
    assert a['delta_accuracy_pp'] == pytest.approx(b['delta_accuracy_pp'])
    width = lambda out: out['accuracy_interval_95_pp'][1] - out['accuracy_interval_95_pp'][0]
    assert width(b) > 2 * width(a)


def test_accuracy_only_contrast_allows_a_missing_class():
    labels = ['A', 'A', 'B', 'B']
    out = stats.paired_contrast(labels, ['A', 'B', 'B', 'B'], ['A', 'A', 'B', 'A'], ['x', 'x', 'y', 'y'], CLASSES,
                                resamples=200, uar=False)
    assert 'delta_uar_pp' not in out and out['delta_accuracy_pp'] == 0
    assert out['candidate_only_correct'] == 1 and out['reference_only_correct'] == 1
    with pytest.raises(ValueError):
        stats.paired_contrast(labels, labels, labels, ['x', 'x', 'y', 'y'], CLASSES, resamples=10)


def test_wer_contrast_is_a_difference_of_corpus_rates():
    reference = {'u1': (1, 10), 'u2': (0, 10), 'u3': (2, 20)}
    candidate = {'u1': (3, 10), 'u2': (0, 10), 'u3': (2, 20)}
    clusters = {'u1': 'a', 'u2': 'b', 'u3': 'c'}
    out = stats.wer_contrast(candidate, reference, clusters, resamples=500)
    assert out['reference_wer'] == pytest.approx(3 / 40) and out['candidate_wer'] == pytest.approx(5 / 40)
    assert out['delta_wer_pp'] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        stats.wer_contrast({'u1': (0, 9)}, {'u1': (0, 10)}, clusters)
    with pytest.raises(ValueError):
        stats.wer_contrast({'u1': (0, 10)}, {'u2': (0, 10)}, clusters)


def test_transcript_errors_ignore_the_models_lead_in():
    reference = "And now- and now it's really shitty timing but now I have to go."
    wrapped = "The transcript is: 'and now and now it's really shitty timing but now i have to go'"
    assert stats.transcript_errors(reference, wrapped) == (0, 14, True)
    assert stats.transcript_errors('Yeah, uh. No.', "The speech is in English and the transcript is: 'Yeah, oh no.'") == (1, 3, True)
    assert stats.transcript_errors('No.', "The speech transcription is: 'no'") == (0, 1, True)
    assert stats.transcript_errors('The speech was long.', 'The speech was long.') == (0, 4, False)
    assert stats.transcript_errors('a b c d', 'a x c') == (2, 4, False)
    assert stats.transcript_errors('a b', '') == (2, 2, False)
