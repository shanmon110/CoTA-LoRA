"""Evaluate AVQA-selected Omni checkpoints on the public MMAR answer-accuracy benchmark."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re
from string import ascii_uppercase
import time

import soundfile as sf
import torch

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest, load_best
from .evaluate import read_rows, score
from .evaluate_frozen import load_frozen
from .prompt import format_question
from .run import pin
from .run_tuners import seal_selection

COMMIT = '3bce090a967db576c8ad433b290a8da582d6d2a0'
METADATA_SHA256 = '1c9a343e7bebf1037482b935c54eacd43cc63df7b2c668724b5560caf1f84fad'
SCORER_SHA256 = 'a3c57b829e40e67e3f7f0bbe7d54a112ea06ed7a781534241ecc5d3751486198'
WAV_SET_SHA256 = '7a27b5edac3537c621056f4569ea06e3c0df7a0ce8a5d0642f004dbc34be810c'
DOMAINS = ('sound', 'music', 'speech', 'mix-sound-music', 'mix-sound-speech', 'mix-music-speech', 'mix-sound-music-speech')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(dataset, source_run, out):
    source = dataset / 'MMAR-meta.jsonl'
    if digest(source) != METADATA_SHA256 or digest(out/'official/evaluation.py') != SCORER_SHA256:
        raise ValueError('MMAR metadata or evaluator differs from the pinned official release')
    records = []
    domains = Counter()
    durations = []
    wav_checksums = {}
    for row in jsonl(source):
        uid, choices = row['id'], row['choices']
        source_id, start, end = uid.rsplit('_', 2)
        if not all(re.fullmatch(r'\d+(?:-\d+){1,2}', t) for t in (start, end)):
            raise ValueError(f'Unexpected clip ID {uid}')
        wav = (dataset / row['audio_path']).resolve(strict=True)
        # The release uses normalized filenames and multi-segment clips for some IDs.
        if wav.parent != dataset/'audio' or wav.name in wav_checksums or choices.count(row['answer']) != 1:
            raise ValueError(f'Audio path or answer ambiguity at {uid}')
        wav_checksums[wav.name] = digest(wav)
        info = sf.info(wav)
        duration = info.frames / info.samplerate
        if duration <= 0 or row['modality'] not in DOMAINS:
            raise ValueError(f'Invalid duration or modality for {uid}')
        domains[row['modality']] += 1
        durations.append(duration)
        records.append(dict(id=uid, source_id=source_id, audios=[str(wav)], question=row['question'], choices=choices,
                            domain=row['modality'], category=row['category'], duration=duration,
                            messages=[dict(role='user', content=format_question(row['question'], choices, allow_empty_options=True)),
                                      dict(role='assistant', content=ascii_uppercase[choices.index(row['answer'])])]))
    if len(records) != len({r['id'] for r in records}) or len(records) != 1000:
        raise ValueError('Expected 1,000 unique public MMAR items')
    if sha256(json.dumps(wav_checksums, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != WAV_SET_SHA256:
        raise ValueError('The audio paths or WAV bytes differ from the official release')
    manifest = out/'test.jsonl'
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records)
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError('Existing MMAR manifest changed')
    if not manifest.exists():
        temp = manifest.with_suffix('.tmp')
        temp.write_text(text)
        temp.replace(manifest)
    read_rows(manifest, allow_empty_options=True)
    data = Path(load(source_run/'plan.json')['data'])
    source_ids = {r['source_id'] for r in records}
    overlaps = {s: sorted(source_ids & {r['source_id'] for r in jsonl(data/(s+'.jsonl'))}) for s in ('train', 'dev')}
    state = load(source_run/'verified-comparison.json')
    first = ['lora', 'ta-lora', 'dora']
    methods = ['frozen-base', *first, *(m for m in state['scores'] if m not in first)]
    pin(out/'protocol.json', dict(official_revision=COMMIT, official_meta_sha256=METADATA_SHA256,
                                 official_scorer_sha256=SCORER_SHA256, manifest_sha256=digest(manifest),
                                 wav_set_sha256=WAV_SET_SHA256, dataset=str(dataset), source_run=str(source_run), source_verification_sha256=digest(source_run/'verified-comparison.json'),
                                 model_revision=load(source_run/'base.json')['model_revision'], methods=methods,
                                 domains=dict(domains), items=len(records), source_videos=len(source_ids),
                                 source_video_overlap=overlaps, audio_seconds_min=min(durations), audio_seconds_max=max(durations),
                                 max_new_tokens=1, eval_batch_size=load(source_run/'base.json')['generation']['batch_size'],
                                 decoded='Greedy selection among the original option letters. Map to original option text for official accuracy.'))


def infer(out, method):
    protocol = load(out/'protocol.json')
    manifest = out/'test.jsonl'
    source = Path(protocol['source_run'])
    if method not in protocol['methods'] or digest(manifest) != protocol['manifest_sha256'] or digest(source/'verified-comparison.json') != protocol['source_verification_sha256']:
        raise ValueError('Unknown checkpoint or changed frozen inputs')
    dest = out/method
    if (dest/'metrics.json').exists():
        raise FileExistsError(f'Completed predictions in {dest}')
    dest.mkdir(exist_ok=True)
    if method == 'frozen-base':
        cfg = load(source/'base.json')
        model, template = load_frozen(cfg)
        identity = dict(checkpoint=cfg['model'], model_revision=cfg['model_revision'])
    else:
        summary = load(source/'results'/f'{method}.json')
        arm = Path(summary['artifact'])
        cfg = load(arm/'protocol.json')['config']
        selection = seal_selection(arm, cfg, method)
        verified = load(source/'verified-comparison.json')['scores'][method]
        selection_file = arm/'selection.json'
        recorded = load(selection_file) if selection_file.exists() else load(arm.parent/'selection.json')[method]
        if (selection['adapter_sha256'] != verified['hash'] or selection['step'] != verified['step']
                or selection['checkpoint'] != summary['checkpoint']
                or any(selection[k] != recorded[k] for k in ('step', 'checkpoint', 'dev_accuracy', 'adapter_sha256', 'config_sha256'))):
            raise ValueError(f'Changed checkpoint or development selection for {method}')
        identity = dict(checkpoint=selection['checkpoint'], step=selection['step'], weights_sha256=selection['adapter_sha256'],
                        config_sha256=selection['config_sha256'], model_revision=cfg['model_revision'])
        model, template = load_best(cfg, method, selection['checkpoint'])
    if cfg['model_revision'] != protocol['model_revision'] or cfg['generation']['batch_size'] != protocol['eval_batch_size']:
        raise ValueError('Backbone or evaluation batch changed')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    preds = dest/'predictions.jsonl'
    metrics = score(model, template, manifest, cfg['generation'], preds, allow_empty_options=True)
    torch.cuda.synchronize()
    metrics.update(eval_seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                   method=method, identity=identity, manifest_sha256=protocol['manifest_sha256'], predictions_sha256=digest(preds))
    if metrics['utterances'] != protocol['items']:
        raise ValueError(f'Incomplete coverage for {method}')
    write(dest/'metrics.json', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    command = parser.add_subparsers(dest='action', required=True)
    prep = command.add_parser('prepare')
    prep.add_argument('--dataset', type=Path, required=True)
    prep.add_argument('--source-run', type=Path, required=True)
    command.add_parser('infer').add_argument('--method', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.dataset.resolve(strict=True), args.source_run.resolve(strict=True), args.out.resolve(strict=True))
    else:
        infer(args.out.resolve(strict=True), args.method)
