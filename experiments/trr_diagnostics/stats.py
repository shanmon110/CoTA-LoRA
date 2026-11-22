"""Paired statistics for closed-set predictions scored on the same utterances."""
import re
from math import comb

import numpy as np


def classification(rows, classes):
    """rows: iterable of (reference, prediction). Returns accuracy, UAR and the confusion matrix."""
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for reference, prediction in rows:
        matrix[index[reference], index[prediction]] += 1
    support = matrix.sum(1)
    if not support.all():
        raise ValueError('Every reference class is required for UAR')
    recalls = matrix.diagonal() / support
    return dict(utterances=int(support.sum()), accuracy=float(matrix.diagonal().sum() / support.sum()),
                uar=float(recalls.mean()), recalls=dict(zip(classes, map(float, recalls))),
                confusion_matrix=matrix.tolist())


def mcnemar_exact(only_reference, only_candidate):
    """Two-sided exact binomial test on the discordant pairs."""
    n = only_reference + only_candidate
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_reference, only_candidate) + 1))
    # Keep the numerator integral until division: float(tail) overflows at
    # roughly 1024 discordant pairs, which a full Session 4 can contain.
    return min(1.0, (2 * tail) / (2 ** n))


def _cluster_counts(labels, correct, clusters, classes):
    names = sorted(set(clusters))
    position = {name: i for i, name in enumerate(names)}
    cls = {label: i for i, label in enumerate(classes)}
    out = np.zeros((len(names), len(classes)))
    for label, hit, cluster in zip(labels, correct, clusters, strict=True):
        out[position[cluster], cls[label]] += hit
    return out


def paired_contrast(labels, candidate, reference, clusters, classes, *, resamples=10000, seed=42, uar=True):
    """Candidate minus reference on the same items, with a cluster-bootstrap 95% interval.

    labels, candidate, reference: per-item gold label and the two predictions.
    clusters: per-item resampling unit (for IEMOCAP, the dialogue an utterance comes from).
    uar=False reports accuracy only, for strata in which some class may be absent.
    """
    labels, candidate, reference = list(labels), list(candidate), list(reference)
    if not labels:
        raise ValueError('No items to compare')
    hit_c = [int(p == y) for p, y in zip(candidate, labels, strict=True)]
    hit_r = [int(p == y) for p, y in zip(reference, labels, strict=True)]
    total = _cluster_counts(labels, [1] * len(labels), clusters, classes)
    corr_c = _cluster_counts(labels, hit_c, clusters, classes)
    corr_r = _cluster_counts(labels, hit_r, clusters, classes)
    if uar and not total.sum(0).all():
        raise ValueError('Every reference class is required for UAR')

    def deltas(weights):
        n = weights @ total
        valid = n.all(1)
        diff = weights @ corr_c - weights @ corr_r
        return (diff / np.where(n == 0, 1, n)).mean(1), diff.sum(1) / np.where(n.sum(1) == 0, 1, n.sum(1)), valid

    point_uar, point_acc, _ = deltas(np.ones((1, total.shape[0])))
    k = total.shape[0]
    draws = np.random.Generator(np.random.PCG64(seed)).multinomial(k, np.full(k, 1.0 / k), size=resamples)
    boot_uar, boot_acc, valid = deltas(draws.astype(float))

    def interval(values):
        return [100 * float(v) for v in np.percentile(values, [2.5, 97.5])] if values.size else None

    only_r = sum(1 for c, r in zip(hit_c, hit_r) if r and not c)
    only_c = sum(1 for c, r in zip(hit_c, hit_r) if c and not r)
    changed = sum(1 for c, r in zip(candidate, reference) if c != r)
    out = dict(utterances=len(labels), clusters=k, resamples=resamples, seed=seed,
               candidate_accuracy=sum(hit_c) / len(labels), reference_accuracy=sum(hit_r) / len(labels),
               delta_accuracy_pp=100 * float(point_acc[0]), accuracy_interval_95_pp=interval(boot_acc),
               reference_only_correct=only_r, candidate_only_correct=only_c,
               mcnemar_exact_p=mcnemar_exact(only_r, only_c),
               prediction_changed=changed, prediction_changed_rate=changed / len(labels))
    if uar:
        out.update(delta_uar_pp=100 * float(point_uar[0]), uar_interval_95_pp=interval(boot_uar[valid]),
                   valid_uar_resamples=int(valid.sum()))
    return out


def wer_contrast(candidate, reference, clusters, *, resamples=10000, seed=42):
    """Candidate minus reference corpus WER. Each argument maps id -> (word_errors, reference_words)."""
    ids = sorted(candidate)
    if ids != sorted(reference):
        raise ValueError('WER contrast needs the same utterances in both conditions')
    names = sorted({clusters[i] for i in ids})
    position = {name: j for j, name in enumerate(names)}
    sums = np.zeros((len(names), 3))
    for i in ids:
        if candidate[i][1] != reference[i][1]:
            raise ValueError(f'{i}: reference transcript differs between conditions')
        sums[position[clusters[i]]] += (candidate[i][0], reference[i][0], candidate[i][1])

    def delta(weights):
        s = weights @ sums
        return (s[:, 0] - s[:, 1]) / np.where(s[:, 2] == 0, 1, s[:, 2])

    k = len(names)
    draws = np.random.Generator(np.random.PCG64(seed)).multinomial(k, np.full(k, 1.0 / k), size=resamples)
    low, high = np.percentile(delta(draws.astype(float)), [2.5, 97.5])
    total = sums.sum(0)
    return dict(utterances=len(ids), clusters=k, candidate_wer=float(total[0] / total[2]),
                reference_wer=float(total[1] / total[2]), delta_wer_pp=100 * float(delta(np.ones((1, k)))[0]),
                interval_95_pp=[100 * float(low), 100 * float(high)], resamples=resamples, seed=seed)


WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
# "The transcript is: ...", "The speech is in English and the transcript is: ...", "The speech transcribed is: ..."
LEAD_IN = re.compile(r"^\s*the (?:speech|audio|transcript|transcription)\b[^:]{0,60}:\s*", re.IGNORECASE)


def transcript_errors(reference, hypothesis):
    """Word edit distance after removing a wrapper such as "The transcript is:" from the hypothesis.

    Returns (word_errors, reference_words, lead_in_removed).
    """
    stripped = LEAD_IN.sub('', hypothesis, count=1)
    ref, hyp = WORD.findall(reference.lower()), WORD.findall(stripped.lower())
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, b in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (a != b))
    return row[-1], len(ref), stripped != hypothesis
