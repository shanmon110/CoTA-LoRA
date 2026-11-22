"""Verify frozen MMAU predictions against source answers and compare the three paired arms."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from string import ascii_uppercase

import numpy as np


METHODS = ('lora', 'dora', 'ta-lora')
OFFICIAL_SCORER_SHA256 = '85480e1c0dfe8ee1406e9c6e598eff0dca9e0216701f076faf69081c6aab1558'


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_delta(left, right, cluster):
    delta = np.asarray(left, dtype=int) - np.asarray(right, dtype=int)
    _, index = np.unique(cluster, return_inverse=True)
    size = np.bincount(index)
    net = np.bincount(index, weights=delta)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(100):
        sampled = rng.integers(0, len(size), size=(100, len(size)))
        draws.extend(100 * net[sampled].sum(axis=1) / size[sampled].sum(axis=1))
    return dict(delta_pp=100 * delta.mean(), interval_95_pp=np.quantile(draws, [0.025, 0.975]).tolist(),
                only_ta_correct=int(sum(delta == 1)), only_other_correct=int(sum(delta == -1)),
                clusters=len(size), resamples=len(draws), seed=42)


def load_official(data, original):
    test = read_jsonl(data / 'test.jsonl')
    source = read_json(original / 'mmau-test-mini.json')
    assert len(test) == len(source) == len({r['id'] for r in source}) == 1000
    scorer_path = original / 'evaluation.py'
    if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != OFFICIAL_SCORER_SHA256:
        raise ValueError('Official MMAU scoring script differs from pinned version')
    spec = importlib.util.spec_from_file_location('mmau_official', scorer_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    return test, source, scorer_path, official


def score_arm(path, method, test, source, scorer_path, official, official_dir=None):
    train_result = read_json(path / 'result.json')
    metrics = read_json(path / 'reload-test' / 'metrics.json')
    predictions = read_jsonl(path / 'reload-test' / 'predictions.jsonl')
    matches = []
    official_matches = []
    native_input = []
    by_domain = Counter()
    domain_counts = Counter()
    for row, gold, p in zip(test, source, predictions, strict=True):
        if row['id'] != gold['id'] or p['id'] != gold['id'] or row['choices'] != gold['choices']:
            raise ValueError(f'Misaligned options or question ID for {method}, {gold["id"]}')
        answer = gold['choices'][ascii_uppercase.index(p['prediction'])]
        is_correct = answer == gold['answer']
        if is_correct != p['is_correct']:
            raise ValueError(f'Exact match disagreement for {method}, {gold["id"]}')
        matches.append(is_correct)
        official_matches.append(official.string_match(gold['answer'], answer, gold['choices']))
        native_input.append({**gold, 'model_output': answer})
        by_domain[gold['task']] += is_correct
        domain_counts[gold['task']] += 1
    official_dir = official_dir or path / 'reload-test'
    official_dir.mkdir(parents=True, exist_ok=True)
    native_path = official_dir / 'official-input.json'
    native_path.write_text(json.dumps(native_input, ensure_ascii=False) + '\n')
    replay = subprocess.run([sys.executable, str(scorer_path), '--input', str(native_path)],
                            capture_output=True, check=True, text=True)
    (native_path.parent / 'official.log').write_text(replay.stdout)
    expected_line = f'Total Accuracy: {100 * sum(official_matches) / len(test):.2f}% over {len(test)} samples'
    if expected_line not in replay.stdout.splitlines():
        raise ValueError(f'Official script disagrees with recomputation: {method}')
    if sum(matches) != metrics['correct'] or metrics['utterances'] != len(test):
        raise ValueError(f'Metric/count disagreement for {method}')
    summary = dict(dev_accuracy=train_result['best_accuracy'], best_step=train_result['best_step'],
                   updates=train_result['global_step'], trainable=train_result['trainable_parameters'],
                   correct=sum(matches), exact_accuracy=sum(matches)/len(test),
                   official_text_accuracy=sum(official_matches)/len(test),
                   by_domain={k: dict(correct=by_domain[k], total=n, accuracy=by_domain[k]/n)
                              for k, n in sorted(domain_counts.items())},
                   train_seconds=train_result['training_seconds'],
                   training_peak_allocated_gib=train_result['peak_allocated_bytes']/2**30)
    return summary, matches, train_result


def score(run, data, original):
    test, source, scorer_path, official = load_official(data, original)
    audio_hash = read_json(data / 'audio-hashes.json')
    clusters = [audio_hash[r['audios'][0]] for r in test]
    rows = {}
    exact = {}
    trained = {}
    for method in METHODS:
        rows[method], exact[method], trained[method] = score_arm(
            run / method, method, test, source, scorer_path, official)
    for key in ('init_sha256', 'input_trajectory', 'gpu'):
        assert all(trained[m][key] == trained[METHODS[0]][key] for m in METHODS)
    return dict(n=len(test), primary='exact original answer text at constrained chosen option',
                official_secondary='MMAU string_match on constrained chosen option text', arms=rows,
                paired={m: paired_delta(exact['ta-lora'], exact[m], clusters) for m in METHODS[:-1]},
                chance_accuracy=float(np.mean([sum(x == r['answer'] for x in r['choices']) / len(r['choices']) for r in source])),
                notes='Confidence intervals sample complete audio clusters with replacement. They condition on these checkpoints and do not include training seed variability.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'data', 'mmau'):
        parser.add_argument('--' + flag, type=Path, required=True)
    a = parser.parse_args()
    result = score(a.run, a.data, a.mmau)
    path = a.run / 'verified-comparison.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path.read_text())
