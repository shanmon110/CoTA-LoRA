"""Compare two completed AQA pilots on the same frozen, MMAU questions."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from string import ascii_uppercase

from .summarize_pilot import METHODS, paired_delta, read_json, read_jsonl


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(condition, reason):
    if not condition:
        raise ValueError(reason)


def paired(left, right, clusters):
    # The common bootstrap routine historically names the first arm "ta".
    result = paired_delta(left, right, clusters)
    result['only_left_correct'] = result.pop('only_ta_correct')
    result['only_right_correct'] = result.pop('only_other_correct')
    return result


def run_environment(run):
    with (run / 'lora-train.log').open() as log:
        executable = json.loads(log.readline())[0]
    args = read_json(run / 'lora/args.json')
    # Captured at report time in the interpreter named by the actual training invocation.
    command = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ('torch','peft','transformers')}))"
    try:
        process = subprocess.run([executable, '-B', '-c', command], capture_output=True,
                                 text=True, check=True, timeout=15)
        libraries = json.loads(process.stdout)
    except (OSError, subprocess.SubprocessError):
        libraries = None
    return {'training_python': executable, 'swift_version_from_training_args': args['swift_version'],
            'installed_versions_at_report_time': libraries, 'dtype': args['torch_dtype'], 'attention': args['attn_impl']}


def collect(run, rows, clusters, checksums):
    config = read_json(run / 'config.json')
    summary = read_json(run / 'verified-comparison.json')
    training = read_json(run / 'training-inputs.json')
    final = read_json(run / 'final-inputs.json')
    selection = read_json(run / 'selection.json')
    checked(read_json(run / 'status.json')['stage'] == 'complete', f'Incomplete experiment: {run}')
    for name in ('train', 'dev', 'test'):
        recorded = final if name == 'test' else training
        checked(recorded[name + '_sha256'] == checksums[name], f'Changed {name} manifest: {run}')
    checked(final['selection_sha256'] == digest(run / 'selection.json'), f'Final selection changed: {run}')
    checked(training['config_sha256'] == digest(run / 'config.json'), f'Configuration changed: {run}')
    arms, correct = {}, {}
    for method in METHODS:
        folder = run / method
        predictions = read_jsonl(folder / 'reload-test/predictions.jsonl')
        train = read_json(folder / 'result.json')
        score = read_json(folder / 'reload-test/metrics.json')
        locked = selection[method]
        checked(locked['step'] == train['best_step'] and locked['adapter_sha256'] == score['adapter_sha256']
                and score['manifest_sha256'] == checksums['test'], f'Changed final scoring inputs: {folder}')
        correct[method] = []
        for row, pred in zip(rows, predictions, strict=True):
            letter = pred['prediction']
            options = row['choices']
            gold = row['messages'][1]['content']
            checked(row['id'] == pred['id'] and pred['reference'] == gold and letter in list(ascii_uppercase[:len(options)]),
                    f'Unmatched test question or illegal letter for {folder}, {row["id"]}')
            answer = options[ascii_uppercase.index(letter)]
            hit = answer == options[ascii_uppercase.index(gold)]
            checked(answer == pred['answer_text'] and hit == pred['is_correct'], f'Answer mismatch: {pred["id"]}')
            correct[method].append(hit)
        arm = summary['arms'][method]
        checked(sum(correct[method]) == score['correct'] == arm['correct'] and
                len(correct[method]) == summary['n'] == score['utterances'], f'Count mismatch: {folder}')
        checked(arm['dev_accuracy'] == locked['dev_accuracy'] == train['best_accuracy'] and
                arm['best_step'] == locked['step'], f'Checkpoint or development accuracy mismatch: {folder}')
        arms[method] = {**arm, 'best_checkpoint': locked['checkpoint'], 'adapter_sha256': locked['adapter_sha256'],
                        'early_stopped': train['early_stopped'], 'dev_seconds': train['dev_seconds'],
                        'training_peak_reserved_gib': train['peak_reserved_bytes'] / 2**30,
                        'eval_seconds': score['eval_seconds'], 'eval_peak_allocated_gib': score['peak_allocated_bytes'] / 2**30,
                        'eval_peak_reserved_gib': score['peak_reserved_bytes'] / 2**30, 'gpu': train['gpu']}
    for arm in arms.values():
        arm['test_rank'] = 1 + sum(other['correct'] > arm['correct'] for other in arms.values())
    pairs = [('ta-lora','lora'), ('ta-lora','dora'), ('lora','dora')]
    within = {f'{left}_minus_{right}': paired(correct[left], correct[right], clusters) for left,right in pairs}
    return {'run': str(run), 'model': config['model'], 'model_revision': config['model_revision'],
            'model_type': config.get('model_type', 'qwen2_5_omni'), 'adapter': config['adapter'],
            'training': config['training'], 'evaluation': config['generation'], 'environment': run_environment(run),
            'arms': arms, 'within_backbone_pairs': within}, correct


def compare(previous, current, data):
    rows = read_jsonl(data / 'test.jsonl')
    checksums = {name: digest(data / (name+'.jsonl')) for name in ('train', 'dev', 'test')}
    audio = read_json(data / 'audio-hashes.json')
    checked(len({r['id'] for r in rows}) == len(rows), 'Duplicate test question ID')
    clusters = [audio[r['audios'][0]] for r in rows]
    old, base = collect(previous, rows, clusters, checksums)
    new, target = collect(current, rows, clusters, checksums)
    cross = {}
    for method in METHODS:
        delta = paired(target[method], base[method], clusters)
        delta['old_rank_minus_new_rank'] = old['arms'][method]['test_rank'] - new['arms'][method]['test_rank']
        cross[method] = delta
    return dict(test_questions=len(rows), pcm_clusters=len(set(clusters)), manifest_sha256=checksums,
                previous=old, current=new, current_minus_previous_by_method=cross,
                notes='Differences include backbone, processor, adapter dimensions, memory settings, software and checkpoint selection. Intervals resample paired PCM audio clusters. They condition on one seed and these trained checkpoints; they are pointwise intervals without multiplicity adjustment.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'current', 'data'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.previous, args.current, args.data), ensure_ascii=False, indent=2))
