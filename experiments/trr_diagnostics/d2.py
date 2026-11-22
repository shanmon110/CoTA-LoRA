"""D2: does a model's answer depend on the F0 and energy contours of the speech it hears?

prepare   build prosody-flattened twins, per-condition manifests and acoustic QC   (CPU, no model)
run       score every condition with one model, frozen or with a trained adapter   (GPU)
report    paired contrasts, strata and the verdicts declared in the config          (CPU)

Every condition asks the same question about the same utterances; only the input changes:
the recording, a twin of it, its transcript, or nothing. Nothing here trains or selects a model.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from random import Random
from zipfile import ZipFile

from . import stats, twin

TRANSCRIPT_LINE = re.compile(r'^(\S+)\s+\[[^\]]+\]:\s*(.*)$')
ANNOTATION = re.compile(r'\[[^\]]*\]')
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def validate_prepared(prepared):
    """Check the frozen protocol and all inputs, including on resumed/report-only runs."""
    prepared = Path(prepared)
    cfg, info = load(prepared / 'config.json'), load(prepared / 'prepare.json')
    if digest(prepared / 'config.json') != info['config_sha256']:
        raise ValueError('Diagnostic config changed after prepare')
    for name, expected in info['manifests'].items():
        if digest(prepared / 'manifests' / f'{name}.jsonl') != expected:
            raise ValueError(f'{name}: manifest changed after prepare')
    if info.get('format_version') != 2 or 'audio_files' not in info:
        raise ValueError('Prepared inputs lack audio digests; prepare again in a new directory')
    for name, expected in info['audio_files'].items():
        if digest(name) != expected:
            raise ValueError(f'Audio changed after prepare: {name}')
    return cfg, info


def completed_predictions(prepared, folder, name, info):
    """A metrics marker commits a complete, hashed prediction file for one manifest."""
    dest = folder / name
    if not (dest / 'metrics.json').exists():
        return None
    metrics = load(dest / 'metrics.json')
    path = dest / 'predictions.jsonl'
    if (metrics.get('manifest_sha256') != info['manifests'][name] or not path.is_file()
            or metrics.get('predictions_sha256') != digest(path)):
        raise ValueError(f'{name}: completed predictions or manifest hash changed')
    rows = read_jsonl(path)
    manifest = read_jsonl(prepared / 'manifests' / f'{name}.jsonl')
    if ([r['id'] for r in rows] != [r['id'] for r in manifest]
            or any(r.get('reference') != m['messages'][1]['content'] for r, m in zip(rows, manifest))):
        raise ValueError(f'{name}: predictions do not cover the manifest and references in order')
    return rows


def code_digests():
    here = Path(__file__).parent
    paths = [here / name for name in ('d2.py', 'stats.py', 'twin.py')]
    paths += [here.parent / name / 'evaluate.py' for name in ('ms_swift_iemocap', 'ms_swift_librispeech')]
    return {str(p.relative_to(here.parent)): digest(p) for p in paths}


# ----------------------------------------------------------------------------- prepare

def read_transcripts(archive, sessions):
    """Utterance id -> transcript line from the IEMOCAP release archive."""
    found = {}
    with ZipFile(archive) as release:
        for name in release.namelist():
            if not name.endswith('.txt') or not any(f'/Session{s}/dialog/transcriptions/' in name for s in sessions):
                continue
            for line in release.read(name).decode('latin1').splitlines():
                match = TRANSCRIPT_LINE.match(line)
                if match and found.setdefault(match[1], match[2]) != match[2]:
                    raise ValueError(f'{match[1]}: conflicting transcripts')
    return found


def clean_transcript(text):
    """Drop bracketed non-lexical annotations such as [LAUGHTER]; they would leak the answer."""
    cleaned = ' '.join(ANNOTATION.sub(' ', text).split())
    return cleaned, cleaned != ' '.join(text.split())


def lexical_key(text):
    return ' '.join(WORD.findall(text.lower()))


def select_items(rows, transcripts, question, classes):
    """Keep utterances that have lexical content, so every condition can be scored on the same set."""
    items, excluded = [], Counter()
    for row in rows:
        messages = row['messages']
        if (len(messages) != 2 or messages[0] != {'role': 'user', 'content': '<audio>' + question}
                or messages[1]['role'] != 'assistant' or messages[1]['content'] not in classes
                or len(row['audios']) != 1):
            raise ValueError(f"{row['id']}: manifest row does not match the declared question and classes")
        if row['id'] not in transcripts:
            excluded['missing_transcript'] += 1
            continue
        text, annotated = clean_transcript(transcripts[row['id']])
        key = lexical_key(text)
        if not key:
            excluded['no_lexical_content'] += 1
            continue
        items.append(dict(id=row['id'], speaker=row.get('speaker'), session=row.get('session'),
                          dialogue=row['id'].rsplit('_', 1)[0], scripted='script' in row['id'],
                          reference=messages[1]['content'], transcript=text, transcript_had_annotation=annotated,
                          lexical_key=key, source_audio=row['audios'][0]))
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Duplicate utterance ids')
    labels = defaultdict(set)
    for item in items:
        labels[item['lexical_key']].add(item['reference'])
    for item in items:
        # Identical words carrying different labels: the words alone cannot decide these.
        item['same_text_different_label'] = len(labels[item.pop('lexical_key')]) > 1
    return items, dict(excluded)


def balanced_subset(items, size, classes, seed):
    """A small class-balanced sample for smoke runs."""
    rng, chosen = Random(seed), []
    for label in classes:
        pool = sorted(item['id'] for item in items if item['reference'] == label)
        chosen += rng.sample(pool, min(len(pool), math.ceil(size / len(classes))))
    keep = set(chosen)
    return [item for item in items if item['id'] in keep]


_WORKER = {}


def _init_worker(params, backend=None):
    _WORKER.update(params=params, backend=backend or twin.make_backend(params))


def _transform(job):
    uid, source, targets = job
    params, backend = _WORKER['params'], _WORKER['backend']
    samples, rate = twin.read_wav(source)
    twins, scales = twin.make_twins(samples, rate, backend, params, list(targets))
    record = dict(id=uid, rate=rate, seconds=samples.size / rate, clipped_samples={}, peak_scale=scales,
                  stats={'orig': twin.contour_stats(samples, rate, backend, params)})
    for name, path in targets.items():
        record['clipped_samples'][name] = twin.write_wav(path, twins[name], rate)
        saved, saved_rate = twin.read_wav(path)
        record['stats'][name] = twin.contour_stats(saved, saved_rate, backend, params)
    return record


def _median(values):
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def summarize_qc(records, conditions, limits):
    """Per-condition contour movement relative to the original, and the acoustic acceptance checks."""
    summary = {}
    for name in ['orig', *conditions]:
        ratios = {}
        for key in ('f0_sd_semitones', 'energy_sd_db'):
            ratios[key] = _median(r['stats'][name][key] / r['stats']['orig'][key] for r in records
                                  if r['stats'][name][key] is not None and r['stats']['orig'][key])
        summary[name] = dict(median_f0_sd_semitones=_median(r['stats'][name]['f0_sd_semitones'] for r in records),
                             median_energy_sd_db=_median(r['stats'][name]['energy_sd_db'] for r in records),
                             median_f0_sd_ratio_to_orig=ratios['f0_sd_semitones'],
                             median_energy_sd_ratio_to_orig=ratios['energy_sd_db'],
                             utterances_without_voicing=sum(r['stats'][name]['f0_sd_semitones'] is None for r in records),
                             clipped_samples=sum(r['clipped_samples'].get(name, 0) for r in records),
                             peak_scaled_utterances=sum(r['peak_scale'].get(name, 1.0) < 1.0 for r in records),
                             max_rms_relative_error=max((abs(r['stats'][name]['rms'] / r['stats']['orig']['rms'] - 1)
                                                         for r in records if r['stats']['orig']['rms'] > 0), default=0.0))
    keep_lo, keep_hi = limits['kept_ratio']
    expectations = {'sham': ('keep', 'keep'), 'flat_f0': ('flat', 'keep'),
                    'flat_energy': ('keep', 'flat'), 'flat_both': ('flat', 'flat')}
    checks = {}
    for name in conditions:
        checks[f'{name}:rms_matched'] = summary[name]['max_rms_relative_error'] <= limits.get('max_rms_relative_error', 0.02)
        checks[f'{name}:no_clipping'] = summary[name]['clipped_samples'] == 0
        for key, mode, limit in (('median_f0_sd_ratio_to_orig', expectations[name][0], limits['flat_f0_max_ratio']),
                                 ('median_energy_sd_ratio_to_orig', expectations[name][1], limits['flat_energy_max_ratio'])):
            value = summary[name][key]
            checks[f'{name}:{key}:{mode}'] = (value is not None and
                                             (value <= limit if mode == 'flat' else keep_lo <= value <= keep_hi))
    return dict(conditions=summary, checks=checks, passed=all(checks.values()))


def git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def prepare(config, output, *, workers=4, limit=None, backend=None):
    cfg = load(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'{output} exists; diagnostics always go to a new directory')
    data, classes, question = cfg['data'], cfg['classes'], cfg['prompts']['question']
    params = {**twin.DEFAULTS, **cfg['twin']}
    audio = cfg['conditions']['audio']
    twins = [name for name in audio if name != 'orig']
    if audio[0] != 'orig' or set(twins) - set(twin.AUDIO_CONDITIONS):
        raise ValueError('Audio conditions must start with orig and otherwise name known twins')
    manifest_sha = digest(data['manifest'])
    if data.get('manifest_sha256') not in (None, manifest_sha):
        raise ValueError('The source manifest differs from the one pinned in the config')
    source_rows = read_jsonl(data['manifest'])
    for row in source_rows:
        session = re.match(r'^Ses(\d{2})[FM]_', row['id'])
        if (session is None or int(session[1]) not in data['sessions']
                or row.get('session', int(session[1])) != int(session[1])):
            raise ValueError(f"{row['id']}: session differs from the declared development sessions")
    items, excluded = select_items(source_rows, read_transcripts(data['archive'], data['sessions']),
                                   question, classes)
    full = len(items)
    if limit:
        items = balanced_subset(items, limit, classes, cfg['bootstrap']['seed'])
    start = time.monotonic()
    output.mkdir(parents=True)
    write(output / 'config.json', cfg)
    for name in twins:
        (output / 'audio' / name).mkdir(parents=True)
    jobs = [(item['id'], item['source_audio'], {name: str(output / 'audio' / name / f"{item['id']}.wav") for name in twins})
            for item in items]
    if backend is not None or workers <= 1:
        _init_worker(params, backend)
        records = [_transform(job) for job in jobs]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(params,)) as pool:
            records = list(pool.imap(_transform, jobs, chunksize=4))
    write_jsonl(output / 'qc.jsonl', records)

    (output / 'manifests').mkdir()
    shared = [{k: v for k, v in item.items() if k != 'source_audio'} for item in items]

    def rows(content, audio_of, answer=lambda item: item['reference']):
        return [dict(meta, audios=audio_of(item),
                     messages=[dict(role='user', content=content(item)), dict(role='assistant', content=answer(item))])
                for meta, item in zip(shared, items, strict=True)]

    def path_of(name):
        return lambda item: [item['source_audio'] if name == 'orig' else str(output / 'audio' / name / f"{item['id']}.wav")]

    manifests = {}
    for name in audio:
        manifests[name] = rows(lambda item: '<audio>' + question, path_of(name))
    for name, template in cfg['conditions']['text'].items():
        manifests[name] = rows(lambda item, t=template: t.format(transcript=item['transcript'], question=question),
                               lambda item: [])
    asr = cfg.get('asr')
    if asr:
        ids = sorted(item['id'] for item in items)
        subset = set(ids if not asr.get('subset') or asr['subset'] >= len(ids) else Random(asr['seed']).sample(ids, asr['subset']))
        for name in asr['conditions']:
            if name not in audio:
                raise ValueError(f'ASR condition {name} is not an audio condition')
            manifests['asr-' + name] = [row for row in rows(lambda item: '<audio>' + cfg['prompts']['asr'], path_of(name),
                                                            answer=lambda item: item['transcript']) if row['id'] in subset]
    hashes = {}
    for name, content in manifests.items():
        write_jsonl(output / 'manifests' / f'{name}.jsonl', content)
        hashes[name] = digest(output / 'manifests' / f'{name}.jsonl')

    used = backend or _WORKER.get('backend') or twin.make_backend(params)
    audio_files = sorted({path for content in manifests.values() for row in content for path in row['audios']})
    info = dict(format_version=2, name=cfg['name'], config_sha256=digest(output / 'config.json'), source_manifest_sha256=manifest_sha,
                utterances=len(items), utterances_before_limit=full, smoke_limit=limit, excluded=excluded,
                class_counts=dict(Counter(item['reference'] for item in items)),
                dialogues=len({item['dialogue'] for item in items}),
                same_text_different_label=sum(item['same_text_different_label'] for item in items),
                transcripts_with_annotations_removed=sum(item['transcript_had_annotation'] for item in items),
                audio_seconds=sum(r['seconds'] for r in records), twin=params,
                vocoder=dict(name=used.name, version=getattr(used, 'version', None)),
                qc=summarize_qc(records, twins, cfg['decision']['qc']), manifests=hashes,
                audio_files={path: digest(path) for path in audio_files},
                code_commit=git_head(), code_sha256=code_digests(), prepare_seconds=time.monotonic() - start)
    write(output / 'prepare.json', info)
    return info


# ----------------------------------------------------------------------------- run

def run(prepared, model_config, method, checkpoint, arm, *, skip_asr=False):
    import torch
    from experiments.ms_swift_iemocap.evaluate import score as score_choice
    from experiments.ms_swift_librispeech.evaluate import score as score_asr
    from experiments.ms_swift_librispeech.train import load_best

    prepared = Path(prepared)
    cfg, info = validate_prepared(prepared)
    model_cfg = load(model_config)
    identity = dict(arm=arm, method=method, model_config=str(Path(model_config).resolve()),
                    model_config_sha256=digest(model_config), base_model=str(Path(model_cfg['model']).resolve()),
                    model_revision=model_cfg['model_revision'], prepared_config_sha256=info['config_sha256'],
                    prepared_sha256=digest(prepared / 'prepare.json'), code_sha256=code_digests())
    if method == 'frozen-base':
        if checkpoint:
            raise ValueError('The frozen base takes no checkpoint')
    else:
        if not checkpoint:
            raise ValueError('An adapter arm needs --checkpoint')
        identity.update(checkpoint=str(Path(checkpoint).resolve()),
                        adapter_config_sha256=digest(Path(checkpoint) / 'adapter_config.json'),
                        weights_sha256=digest(Path(checkpoint) / 'adapter_model.safetensors'))
    folder = prepared / 'runs' / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'identity.json').exists():
        if load(folder / 'identity.json') != identity:
            raise ValueError(f'Arm {arm!r} was already used with a different model; choose another --arm')
    else:
        write(folder / 'identity.json', identity)

    # The audio conditions are the diagnostic itself; text-only and ASR inputs are extra context.
    jobs = [(name, score_choice, cfg['generation'], True) for name in cfg['conditions']['audio']]
    jobs += [(name, score_choice, cfg['generation'], False) for name in cfg['conditions']['text']]
    if cfg.get('asr') and not skip_asr:
        jobs += [('asr-' + name, score_asr, cfg['asr']['generation'], False) for name in cfg['asr']['conditions']]
    pending = []
    for job in jobs:
        if completed_predictions(prepared, folder, job[0], info) is None:
            pending.append(job)
        else:
            print(f'{job[0]}: already scored and verified, kept', flush=True)
    if not pending:
        return []
    if method == 'frozen-base':
        from experiments.ms_swift_aqa.evaluate_frozen import load_frozen
        model, template = load_frozen(model_cfg)
    else:
        model, template = load_best(model_cfg, method, Path(checkpoint))
        model.requires_grad_(False)
    failed = []
    for name, scorer, generation, required in pending:
        dest = folder / name
        manifest = prepared / 'manifests' / f'{name}.jsonl'
        if digest(manifest) != info['manifests'][name]:
            raise ValueError(f'{name}: manifest changed after prepare')
        (dest / 'failure.json').unlink(missing_ok=True)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            metrics = scorer(model, template, manifest, generation, dest / 'predictions.jsonl')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               gpu_name=torch.cuda.get_device_name())
            metrics.update(condition=name, eval_seconds=time.perf_counter() - start, manifest_sha256=info['manifests'][name],
                           predictions_sha256=digest(dest / 'predictions.jsonl'))
            write(dest / 'metrics.json', metrics)
            headline = metrics.get('uar', metrics.get('wer'))
            print(f"{name}: {'UAR' if 'uar' in metrics else 'WER'} {headline:.4f} in {metrics['eval_seconds']:.0f}s", flush=True)
        except Exception as exc:
            if required:
                raise
            dest.mkdir(parents=True, exist_ok=True)
            write(dest / 'failure.json', dict(condition=name, error=repr(exc), traceback=traceback.format_exc()))
            failed.append(name)
            print(f'{name}: FAILED ({exc!r}); recorded in {dest}/failure.json, continuing', flush=True)
    return failed


# ----------------------------------------------------------------------------- report

def _predictions(folder, name, manifest):
    rows = read_jsonl(folder / name / 'predictions.jsonl')
    if [r['id'] for r in rows] != [m['id'] for m in manifest]:
        raise ValueError(f'{name}: predictions do not cover the manifest in order')
    if any(r['reference'] != m['reference'] for r, m in zip(rows, manifest)):
        raise ValueError(f'{name}: references differ from the manifest')
    return [r['prediction'] for r in rows]


def decide(decision, contrasts, wer):
    """Apply the thresholds fixed in the config before any model was run."""
    verdicts = {}

    def get(name):
        c = contrasts.get(name)
        return (c['delta_uar_pp'], *c['uar_interval_95_pp']) if c and c['uar_interval_95_pp'] else None

    if (v := get('sham-vs-orig')):
        verdicts['vocoder_artifacts'] = 'pass' if abs(v[0]) <= decision['max_artifact_pp'] else 'fail'
    if (v := get('flat_both-vs-sham')):
        delta, low, high = v
        band = decision['equivalence_pp']
        # uses: a drop of at least min_use_pp whose interval excludes zero.
        # at_most_small: the whole interval lies inside +-equivalence_pp.
        verdicts['f0_energy_use'] = ('uses' if delta <= -decision['min_use_pp'] and high < 0 else
                                     'at_most_small' if -band <= low and high <= band else 'inconclusive')
    if (v := get('orig-vs-text')):
        verdicts['audio_beyond_transcript'] = 'yes' if v[1] > 0 else 'no' if v[2] <= 0 else 'inconclusive'
    if (w := wer.get('asr-flat_both-vs-asr-sham')):
        verdicts['intelligibility_kept'] = 'pass' if w['delta_wer_pp'] <= decision['max_wer_increase_pp'] else 'fail'
    return verdicts


def report(prepared, arm):
    prepared = Path(prepared)
    cfg, info = validate_prepared(prepared)
    folder = prepared / 'runs' / arm
    identity = load(folder / 'identity.json')
    if identity.get('prepared_sha256') != digest(prepared / 'prepare.json'):
        raise ValueError('Prepared metadata changed since this arm was started')
    classes, boot = cfg['classes'], cfg['bootstrap']
    manifest = read_jsonl(prepared / 'manifests' / 'orig.jsonl')
    labels = [m['reference'] for m in manifest]
    clusters = [m[boot['cluster']] for m in manifest]
    names = [n for n in [*cfg['conditions']['audio'], *cfg['conditions']['text']]
             if completed_predictions(prepared, folder, n, info) is not None]
    preds = {n: _predictions(folder, n, manifest) for n in names}
    conditions = {n: stats.classification(zip(labels, preds[n]), classes) for n in names}

    def contrast(candidate, reference, keep=None, uar=True):
        idx = [i for i in range(len(labels)) if keep is None or keep[i]]
        out = stats.paired_contrast([labels[i] for i in idx], [preds[candidate][i] for i in idx],
                                    [preds[reference][i] for i in idx], [clusters[i] for i in idx], classes,
                                    resamples=boot['resamples'], seed=boot['seed'], uar=uar)
        if out['clusters'] < 2:  # one dialogue cannot be resampled; keep the point estimate only
            out.update({key: None for key in ('accuracy_interval_95_pp', 'uar_interval_95_pp') if key in out})
        return out

    contrasts, skipped = {}, []
    for candidate, reference in cfg['contrasts']:
        key = f'{candidate}-vs-{reference}'
        if candidate in preds and reference in preds:
            contrasts[key] = contrast(candidate, reference)
        else:
            skipped.append(key)

    strata = {'scripted': [m['scripted'] for m in manifest], 'improvised': [not m['scripted'] for m in manifest],
              'same_text_different_label': [m['same_text_different_label'] for m in manifest]}
    if 'text' in preds:
        strata['transcript_alone_correct'] = [p == y for p, y in zip(preds['text'], labels)]
        strata['transcript_alone_wrong'] = [p != y for p, y in zip(preds['text'], labels)]
    for speaker in sorted({m['speaker'] for m in manifest if m.get('speaker')}):
        strata[f'speaker:{speaker}'] = [m['speaker'] == speaker for m in manifest]
    by_stratum = {}
    for candidate, reference in cfg['strata_contrasts']:
        if candidate in preds and reference in preds:
            by_stratum[f'{candidate}-vs-{reference}'] = {name: contrast(candidate, reference, keep, uar=False)
                                                         for name, keep in strata.items() if any(keep)}

    asr, wer = {}, {}
    for name in (cfg.get('asr') or {}).get('conditions', []):
        rows = completed_predictions(prepared, folder, f'asr-{name}', info)
        if rows is not None:
            asr[f'asr-{name}'] = {r['id']: (r['word_errors'], r['reference_words']) for r in rows}
    dialogue = {m['id']: m[boot['cluster']] for m in manifest}
    for candidate, reference in (cfg.get('asr') or {}).get('contrasts', []):
        if f'asr-{candidate}' in asr and f'asr-{reference}' in asr:
            wer[f'asr-{candidate}-vs-asr-{reference}'] = stats.wer_contrast(
                asr[f'asr-{candidate}'], asr[f'asr-{reference}'], dialogue, resamples=boot['resamples'], seed=boot['seed'])

    failures = sorted(p.parent.name for p in folder.glob('*/failure.json'))
    raw_verdicts = decide(cfg['decision'], contrasts, wer)
    reasons = []
    if info['smoke_limit'] is not None:
        reasons.append('smoke_run')
    if not info['qc']['passed']:
        reasons.append('acoustic_qc_failed')
    for control in ('vocoder_artifacts', 'intelligibility_kept'):
        if raw_verdicts.get(control) != 'pass':
            reasons.append(f"{control}:{raw_verdicts.get(control, 'missing')}")
    if 'f0_energy_use' not in raw_verdicts:
        reasons.append('primary_contrast_missing')
    verdicts = dict(raw_verdicts)
    if reasons:
        verdicts['f0_energy_use'] = 'not_interpretable'
    result = dict(arm=arm, identity=identity, smoke_limit=info['smoke_limit'],
                  utterances=len(labels), dialogues=len(set(clusters)), conditions=conditions, contrasts=contrasts,
                  strata=by_stratum, asr_wer={k: sum(e for e, _ in v.values()) / sum(w for _, w in v.values()) for k, v in asr.items()},
                  wer_contrasts=wer, qc=info['qc'], decision=cfg['decision'],
                  verdicts=verdicts, raw_verdicts=raw_verdicts,
                  interpretation=dict(eligible=not reasons, reasons=reasons),
                  skipped_contrasts=skipped, failed_conditions=failures)
    write(folder / 'report.json', result)
    text = render(result)
    (folder / 'report.md').write_text(text)
    return result, text


def render(result):
    def pp(value):
        return 'n/a' if value is None else f'{value:+.2f}'

    def ci(pair):
        return 'n/a' if pair is None else f'[{pair[0]:+.2f}, {pair[1]:+.2f}]'

    lines = [f"# D2 report: {result['arm']}", '']
    if not result['interpretation']['eligible']:
        lines += ['**F0/energy interpretation: NOT INTERPRETABLE.** ' + ', '.join(result['interpretation']['reasons']),
                  'Point estimates below are descriptive; failed or missing controls do not support a prosody-use conclusion.', '']
    if result['smoke_limit']:
        lines += [f"**SMOKE RUN on {result['utterances']} utterances. Numbers are not results.**", '']
    lines += [f"{result['utterances']} utterances from {result['dialogues']} dialogues. Intervals are 95% cluster "
              'bootstrap over dialogues; they describe this one model, not training-seed variation.', '',
              '| condition | UAR | accuracy |', '|---|---:|---:|']
    lines += [f"| {name} | {100 * m['uar']:.2f} | {100 * m['accuracy']:.2f} |" for name, m in result['conditions'].items()]
    lines += ['', '| contrast (candidate - reference) | dUAR pp | 95% CI | dAcc pp | only ref / only cand correct | McNemar p | answers changed |',
              '|---|---:|---|---:|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        lines.append(f"| {name} | {pp(c['delta_uar_pp'])} | {ci(c['uar_interval_95_pp'])} | {pp(c['delta_accuracy_pp'])} | "
                     f"{c['reference_only_correct']} / {c['candidate_only_correct']} | {c['mcnemar_exact_p']:.3g} | "
                     f"{100 * c['prediction_changed_rate']:.1f}% |")
    for name, strata in result['strata'].items():
        lines += ['', f'Strata for {name} (accuracy only; post-hoc, descriptive):', '',
                  '| stratum | n | reference acc | candidate acc | dAcc pp | 95% CI |', '|---|---:|---:|---:|---:|---|']
        for stratum, c in strata.items():
            lines.append(f"| {stratum} | {c['utterances']} | {100 * c['reference_accuracy']:.1f} | "
                         f"{100 * c['candidate_accuracy']:.1f} | {pp(c['delta_accuracy_pp'])} | {ci(c['accuracy_interval_95_pp'])} |")
    if result['asr_wer']:
        lines += ['', '| ASR input | WER % |', '|---|---:|']
        lines += [f'| {name} | {100 * value:.2f} |' for name, value in result['asr_wer'].items()]
        lines += [f"\n{name}: {pp(w['delta_wer_pp'])} pp, 95% CI {ci(w['interval_95_pp'])}" for name, w in result['wer_contrasts'].items()]
    lines += ['', f"Acoustic QC of the twins: {'pass' if result['qc']['passed'] else 'FAIL'}", '',
              '| twin | F0 SD / original | energy SD / original |', '|---|---:|---:|']
    for name, q in result['qc']['conditions'].items():
        if name != 'orig':
            f0_ratio, energy_ratio = q['median_f0_sd_ratio_to_orig'], q['median_energy_sd_ratio_to_orig']
            lines.append(f"| {name} | {'n/a' if f0_ratio is None else format(f0_ratio, '.2f')} | "
                         f"{'n/a' if energy_ratio is None else format(energy_ratio, '.2f')} |")
    lines += ['', 'Verdicts under the thresholds fixed in the config:', '']
    lines += [f'- {name}: **{value}**' for name, value in result['verdicts'].items()] or ['- none available']
    if result['failed_conditions']:
        lines += ['', 'Conditions that failed to run: ' + ', '.join(result['failed_conditions'])]
    if result['skipped_contrasts']:
        lines += ['', 'Contrasts skipped for missing conditions: ' + ', '.join(result['skipped_contrasts'])]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='Class-balanced smoke subset; results are not reportable')
    r = commands.add_parser('run')
    r.add_argument('--prepared', type=Path, required=True)
    r.add_argument('--model-config', type=Path, required=True)
    r.add_argument('--method', required=True, choices=('frozen-base', 'lora', 'ta-lora', 'dora'))
    r.add_argument('--checkpoint', type=Path)
    r.add_argument('--arm', required=True, help='Name of this model in the output, e.g. frozen-base')
    r.add_argument('--skip-asr', action='store_true')
    s = commands.add_parser('report')
    s.add_argument('--prepared', type=Path, required=True)
    s.add_argument('--arm', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        summary = prepare(args.config, args.output, workers=args.workers, limit=args.limit)
        print(json.dumps({k: summary[k] for k in ('utterances', 'excluded', 'class_counts', 'dialogues',
                                                   'same_text_different_label', 'audio_seconds', 'prepare_seconds')}, indent=2))
        print('acoustic QC:', 'pass' if summary['qc']['passed'] else 'FAIL', json.dumps(summary['qc']['checks'], indent=2))
    elif args.command == 'run':
        failures = run(args.prepared, args.model_config, args.method, args.checkpoint, args.arm, skip_asr=args.skip_asr)
        print(report(args.prepared, args.arm)[1])
        sys.exit(1 if failures else 0)
    else:
        print(report(args.prepared, args.arm)[1])
