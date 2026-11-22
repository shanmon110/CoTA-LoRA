"""Download AVQA training media and make a deterministic, source-grouped audio-only pilot."""
import argparse
import hashlib
import io
import json
import re
import threading
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import ascii_uppercase
from urllib.parse import quote

import requests

from .prompt import format_question


OFFICIAL = 'a5056c1d82849aa0981f50fa4c5a57bd16658c52'
WAV_REV = '9508c74bb202fa7d7ab6a9c329dd66d7ecca970e'
VIDEO_REV = 'e9538b7abd3dcf018f85c9e431d51f542728c13b'
OFFICIAL_URL = f'https://raw.githubusercontent.com/AlyssaYoung/AVQA/{OFFICIAL}/data/annotation/train_qa.json'
ANNOTATION_SHA256 = '55ec202636c3e25f279995eb68ddd14264004cbe0d042d1f559b03662a8575ee'
HF = 'https://huggingface.co'
LOCAL = threading.local()


def json_load(path):
    return json.loads(path.read_bytes())


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_bytes(data)
    temp.replace(path)


def save_json(path, data):
    atomic(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())


def save_rows(path, rows):
    atomic(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http(url):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    for attempt in range(5):
        try:
            response = LOCAL.session.get(url, timeout=(12, 45))
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if getattr(error.response, 'status_code', 0) in (401, 403, 404, 410) or attempt == 4:
                raise
            delay = min(90, max(2 ** attempt, int(response.headers.get('Retry-After', '0') or '0'))) if 'response' in locals() else 2 ** attempt
            time.sleep(delay)


def source_metadata(root):
    src = root / 'sources'
    sources = {'official-train.json': OFFICIAL_URL,
               'joy.json': f'{HF}/api/datasets/Joysw909/AVQA/revision/{WAV_REV}',
               'joy_annotations.json': f'{HF}/datasets/Joysw909/AVQA/resolve/{WAV_REV}/train_r1aqa_line.json'}
    for name, url in sources.items():
        if not (src / name).exists():
            atomic(src / name, http(url))
    if sha(src / 'official-train.json') != ANNOTATION_SHA256:
        raise ValueError('Annotation checksum differs from the pinned official GitHub file')
    original = json_load(src / 'official-train.json')
    by_id = {row['id']: row for row in original}
    if len(by_id) != len(original) or len(original) != 40425:
        raise ValueError('Unexpected official question identifiers or row count')
    inventory = json_load(src / 'joy.json')
    if inventory['sha'] != WAV_REV:
        raise ValueError('Unexpected WAV repository revision')
    available = {item['rfilename'] for item in inventory['siblings']}
    mapped = {}
    for line in (src / 'joy_annotations.json').read_text().splitlines():
        row = json.loads(line)
        official = by_id[row['id']]
        if any(row[k] != value for k, value in official.items()):
            raise ValueError(f'Mirror annotation differs for {row["id"]}')
        name = row['video_name']
        filename = row['audio_path'].split('/AVQA/')[-1]
        yt, offset = name.rsplit('_', 1)
        if not re.fullmatch('[A-Za-z0-9_-]{11}', yt) or Path(filename).name != f'{yt}_{int(offset)}.wav':
            raise ValueError(f'Media filename differs for {name}')
        if filename in available:
            assert mapped.setdefault(name, filename) == filename
    return original, mapped


def rank(seed, scope, value):
    return hashlib.sha256(f'{seed}|{scope}|{value}'.encode()).hexdigest()


def pilot_candidates(original, mapped, seed, train_count, dev_count):
    from math import floor
    by_split = {'train': [], 'dev': []}
    excluded = []
    for row in original:
        name = row['video_name']
        opts = row['multi_choice']
        if not (len(opts) == 4 and isinstance(row['answer'], int) and 0 <= row['answer'] < 4):
            raise ValueError('Invalid original AVQA choices or zero-based gold')
        reasons = []
        if name not in mapped:
            reasons.append('no-WAV-in-pinned-audio-distribution')
        if len({c.strip().casefold() for c in opts}) < 4:
            reasons.append('nonunique-options')
        if reasons:
            excluded.append({'id':row['id'], 'video_name':name, 'reasons':reasons})
            continue
        source = name.rsplit('_', 1)[0]
        split = 'dev' if int(rank(seed, 'partition', source), 16) % 5 == 0 else 'train'
        by_split[split].append(row)
    selected = {}
    for split, count in [('train', train_count), ('dev', dev_count)]:
        pool = by_split[split]
        if count < 3 or count > len(pool):
            raise ValueError('Requested split size out of range')
        frequency = Counter(row['question_relation'] for row in pool)
        ideal = {k:count*v/len(pool) for k,v in frequency.items()}
        quota = {k:max(1, floor(v)) for k,v in ideal.items()}
        while sum(quota.values()) < count:
            k = max(quota, key=lambda kind: (ideal[kind]-quota[kind],kind))
            quota[k] += 1
        while sum(quota.values()) > count:
            k = max((k for k in quota if quota[k]>1), key=lambda kind: (quota[kind]-ideal[kind],kind))
            quota[k] -= 1
        ordered = sorted(pool, key=lambda r: rank(seed, 'question', r['id']))
        result = []
        for row in ordered:
            kind = row['question_relation']
            if quota[kind]:
                result.append(row)
                quota[kind] -= 1
        assert len(result) == count
        selected[split] = result
    train_src, dev_src = [{r['video_name'].rsplit('_',1)[0] for r in selected[s]} for s in ('train','dev')]
    if train_src & dev_src:
        raise ValueError('Shared YouTube source across pilot splits')
    return selected, excluded


def inspect(data):
    with wave.open(io.BytesIO(data)) as wav:
        rate, channels, width, n = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
        pcm = wav.readframes(n)
    if channels != 1 or rate != 16000 or width != 2 or not 0 < n <= 320000 or len(pcm) != n*width:
        raise ValueError(f'Unsupported or truncated audio: {rate} Hz, {channels} channels, {n} frames')
    return dict(samples=n, rate=rate, sha256=hashlib.sha256(data).hexdigest())


def extract_audio(data):
    import av
    with av.open(io.BytesIO(data)) as movie:
        if not movie.streams.audio:
            raise ValueError('Source lacks audio stream')
        res = av.AudioResampler(format='s16', layout='mono', rate=16000)
        out = io.BytesIO()
        with wave.open(out, 'wb') as w:
            w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            for frame in movie.decode(movie.streams.audio[0]):
                for block in res.resample(frame):
                    w.writeframes(block.to_ndarray().tobytes())
            for block in res.resample(None):
                w.writeframes(block.to_ndarray().tobytes())
    return out.getvalue()


def acquire(root, clip, mapped):
    wave_name = mapped.get(clip)
    wav_url = f'{HF}/datasets/Joysw909/AVQA/resolve/{WAV_REV}/{quote(wave_name)}' if wave_name else None
    video_url = f'{HF}/datasets/juyil/AVQA-videos/resolve/{VIDEO_REV}/videos/{quote(clip)}.mp4'
    path = root/'audio'/(clip+'.wav')
    meta = root/'media-index'/(clip+'.json')
    if path.exists() and meta.exists():
        record = json_load(meta)
        blob = path.read_bytes()
        if inspect(blob) != record['audio']:
            raise ValueError(f'Cached WAV checksum or metadata mismatch for {clip}')
        return record
    errors = []
    for url in filter(None, (wav_url, video_url)):
        try:
            payload = http(url)
            data = payload if url == wav_url else extract_audio(payload)
            info = inspect(data)
            record = dict(clip=clip, path=str(path.resolve()), url=url, source_sha256=hashlib.sha256(payload).hexdigest(),audio=info)
            atomic(path,data)
            save_json(meta,record)
            return record
        except Exception as exc:
            # URLs in transport errors can contain expiring credentials; record the public path and status.
            errors.append(dict(url=url, error=type(exc).__name__, status=getattr(getattr(exc,'response',None),'status_code',None)))
    return dict(clip=clip, errors=errors)


def swift_row(row, media):
    choices = row['multi_choice']
    name = row['video_name']
    return dict(id=f'avqa-{row["id"]}', audios=[media['path']], source_id=name.rsplit('_',1)[0], clip_id=name,
                original_id=row['id'], question=row['question_text'], choices=choices,
                question_relation=row['question_relation'], question_type=row['question_type'],
                messages=[dict(role='user',content=format_question(row['question_text'],choices)),
                          dict(role='assistant',content=ascii_uppercase[row['answer']])])


def run(args):
    root = args.root.resolve()
    out = root / args.stage
    out.mkdir(parents=True,exist_ok=True)
    original, mapped = source_metadata(root)
    if args.stage == 'pilot':
        splits, excluded = pilot_candidates(original,mapped,args.seed,args.train,args.dev)
        save_rows(out/'ineligible.jsonl',excluded)
        save_json(out/'selection.json', {k: [r['id'] for r in rows] for k,rows in splits.items()})
    else:
        splits = {'official_train': original}
    clips = sorted({r['video_name'] for rows in splits.values() for r in rows})
    from concurrent.futures import as_completed
    records, errors = {}, []
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(acquire,root,clip,mapped):clip for clip in clips}
        for i,job in enumerate(as_completed(jobs),1):
            result=job.result()
            if 'errors' in result: errors.append(result)
            else: records[result['clip']]=result
            if i%50==0 or i==len(jobs):
                save_json(out/'progress.json', dict(processed=i,clips=len(jobs),valid=len(records),errors=len(errors)))
                if i%500==0 or i==len(jobs): print(i,len(jobs),'valid',len(records),'errors',len(errors),flush=True)
    save_rows(out/'unavailable-media.jsonl',sorted(errors,key=lambda x:x['clip']))
    if args.stage=='pilot' and errors:
        raise RuntimeError(f'{len(errors)} selected clips unavailable; pilot manifests withheld')
    metrics={}
    for k,rows in splits.items():
        ok=[swift_row(r,records[r['video_name']]) for r in rows if r['video_name'] in records]
        path=out/(k+'.jsonl')
        save_rows(path,ok)
        metrics[k]=dict(questions=len(ok),source_videos=len({r['source_id'] for r in ok}),clips=len({r['clip_id'] for r in ok}),
                        relations=Counter(r['question_relation'] for r in ok),sha256=sha(path))
    save_json(out/'provenance.json',dict(official_repository='AlyssaYoung/AVQA',official_commit=OFFICIAL,official_train_url=OFFICIAL_URL,
        wav_distribution='Joysw909/AVQA',wav_revision=WAV_REV,video_fallback='juyil/AVQA-videos',video_revision=VIDEO_REV,
        official_questions=len(original),official_clips=len({r['video_name'] for r in original}),mapped_training_wav_clips=len(mapped),
        verified_wav_clips=len(records),unavailable_clips=len(errors),source_video_grouping='strip final underscore and 6-digit clip start',
        seed=args.seed,sha256_annotations=sha(root/'sources'/'official-train.json'),sets=metrics))
    print(metrics,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('pilot','full'),default='pilot')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train',type=int,default=2048)
    parser.add_argument('--dev',type=int,default=256)
    parser.add_argument('--workers',type=int,default=32)
    run(parser.parse_args())
