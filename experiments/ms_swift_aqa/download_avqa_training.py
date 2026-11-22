"""Fetch the full AVQA training audio distribution and verify the original clip/QA mapping."""
import argparse
import hashlib
import json
import io
import wave
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import requests

from .prepare_avqa import (HF, OFFICIAL, acquire, atomic, extract_audio, inspect, json_load,
                           save_json, save_rows, sha, source_metadata, swift_row)


REPO = 'gijs/avqa-processed'
REV = '0df80b57db91faa828df0819098885adc1c865bd'
CHUNK = 4 * 1024 * 1024
LOCAL = threading.local()
CLIP_LOCKS = [threading.Lock() for _ in range(73)]


def session(direct=False):
    name = 'direct' if direct else 'proxy'
    if not hasattr(LOCAL, name):
        client = requests.Session()
        client.trust_env = not direct
        setattr(LOCAL, name, client)
    return getattr(LOCAL, name)


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download_shard(root, item, connections):
    relative = item['path']
    size = item['size']
    expected = item['lfs']['oid']
    name = Path(relative).name
    final = root / 'shards' / name
    cache = root / 'shards' / (name + '.ranges')
    cache.mkdir(parents=True, exist_ok=True)
    url = f'{HF}/datasets/{REPO}/resolve/{REV}/{relative}?download=true'
    if final.exists() and final.stat().st_size == size and file_hash(final) == expected:
        return final
    # Request a single unbounded signed URL. Range requests are sent directly to the CDN.
    def redirect(direct):
        base = url.replace(HF, 'https://hf-mirror.com') if direct else url
        signed = session(direct).get(base, allow_redirects=False, timeout=25)
        if signed.status_code not in (301,302,307,308):
            raise RuntimeError(f'Expected CDN redirect for {relative}, got HTTP {signed.status_code}')
        return signed.headers['Location']

    try:
        direct = True
        location = redirect(True)
    except (requests.RequestException, RuntimeError):
        direct = False
        location = redirect(False)
    path = cache / 'payload'
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, size)

    def fetch(start):
        end = min(size, start+CHUNK)-1
        done = cache / str(start)
        if done.exists():
            return
        for attempt in range(6):
            try:
                use_direct = direct and attempt < 4
                endpoint = location if use_direct == direct else redirect(False)
                with session(use_direct).get(endpoint, headers={'Range':f'bytes={start}-{end}'},
                                   timeout=(15,90), stream=True) as response:
                    wanted = f'bytes {start}-{end}/{size}'
                    if response.status_code!=206 or response.headers.get('Content-Range') != wanted:
                        raise RuntimeError(f'{relative} range {start} HTTP {response.status_code}')
                    data = response.content
                if len(data)!=end-start+1 or os.pwrite(fd,data,start)!=len(data):
                    raise ValueError(f'Incomplete range at {start}')
                done.touch()
                return
            except (requests.RequestException, OSError, RuntimeError, ValueError) as exc:
                if attempt==5:
                    raise RuntimeError(f'Failed {name} range {start} ({type(exc).__name__})') from None
                time.sleep(min(30,2**attempt))
    try:
        with ThreadPoolExecutor(connections) as pool:
            for _ in pool.map(fetch, range(0,size,CHUNK)):
                pass
        os.fsync(fd)
    finally:
        os.close(fd)
    if file_hash(path)!=expected:
        for done in cache.iterdir():
            if done.name.isdigit():
                done.unlink()
        raise ValueError(f'Archive hash mismatch, marked for fresh acquisition: {name}')
    path.replace(final)
    for mark in cache.iterdir():
        if mark.name.isdigit():
            mark.unlink()
    cache.rmdir()
    return final


def convert_shard(root, item, connections, expected, qa):
    name = Path(item['path']).name
    receipt = root/'full'/'shards'/(name+'.json')
    if receipt.exists():
        info = json_load(receipt)
        if info['input_sha256'] != item['lfs']['oid']:
            raise ValueError('Receipt has a different source digest')
        return info
    shard=download_shard(root,item,connections)
    counts=Counter()
    issues=[]
    new_clips=[]
    discrepancies=[]
    pilot_silences={r[2] for r in json_load(root/'pilot'/'silent-audit.json')} if (root/'pilot'/'silent-audit.json').exists() else set()
    with pa.memory_map(str(shard),'r') as data:
        reader=pa.ipc.open_stream(data)
        if reader.schema.names != ['audio','file_name','question','answer']:
            raise ValueError(f'Unexpected schema: {reader.schema}')
        for batch in reader:
            for i in range(batch.num_rows):
                counts['rows'] += 1
                filename = batch['file_name'][i].as_py()
                clip=Path(filename).stem
                if clip not in expected:
                    issues.append({'filename':filename,'reason':'unknown-clip'})
                    continue
                question = batch['question'][i].as_py().split('\nChoices:',1)[0].strip()
                candidate_answers=qa[clip].get(question,set())
                answer=batch['answer'][i].as_py().strip()
                if answer not in candidate_answers:
                    counts['question_or_answer_text_differs']+=1
                # Gold for our manifests always comes from the original official JSON.
                path=root/'audio'/(clip+'.wav')
                meta=root/'media-index'/(clip+'.json')
                lock=CLIP_LOCKS[hash(clip)%len(CLIP_LOCKS)]
                with lock:
                    if path.is_file() and meta.is_file():
                        if clip in pilot_silences:
                            audio = batch['audio'][i].as_py()
                            if audio and audio.get('bytes'):
                                with wave.open(io.BytesIO(extract_audio(audio['bytes'])), 'rb') as w:
                                    nonzero = any(w.readframes(w.getnframes()))
                                discrepancies.append(dict(clip=clip, pilot_all_zero=True, alternate_all_zero=not nonzero))
                        counts['previously_stored']+=1
                        continue
                    audio=batch['audio'][i].as_py()
                    if not audio or not audio.get('bytes'):
                        issues.append({'clip':clip,'reason':'missing-bytes-in-distribution'})
                        continue
                    blob=audio['bytes']
                    try:
                        wav=extract_audio(blob)
                        details=inspect(wav)
                    except Exception as exc:
                        issues.append({'clip':clip,'reason':'invalid-audio','kind':type(exc).__name__})
                        continue
                    atomic(path,wav)
                    save_json(meta,dict(clip=clip,path=str(path.resolve()),archive_repo=REPO,archive_revision=REV,
                        shard=item['path'],shard_sha256=item['lfs']['oid'],source_sha256=hashlib.sha256(blob).hexdigest(),audio=details))
                    counts['new_wavs']+=1
                    new_clips.append(clip)
    info=dict(shard=name,input_sha256=item['lfs']['oid'],bytes=item['size'],counts=counts,issues=issues,new_clips=new_clips,pilot_silence_cross_check=discrepancies)
    save_json(receipt,info)
    # The indexed, validated 16-kHz WAVs are kept. Avoid keeping a second 38.65-GB representation.
    shard.unlink()
    return info


def run(args):
    root=args.root.resolve()
    out=root/'full'
    out.mkdir(parents=True,exist_ok=True)
    src=root/'sources'/'gijs-training-blobs.json'
    if not src.exists():
        u=f'{HF}/api/datasets/{REPO}/tree/{REV}/train?limit=100'
        response=requests.get(u,timeout=30)
        response.raise_for_status()
        atomic(src,response.content)
    items=[x for x in json_load(src) if x['path'].endswith('.arrow')]
    if len(items)!=78 or sum(x['size'] for x in items)!=38652670600:
        raise ValueError('Unexpected training archive inventory')
    original, mapped=source_metadata(root)
    expected={r['video_name'] for r in original}
    qa=defaultdict(lambda:defaultdict(set))
    for row in original:
        answer=row['answer']
        qa[row['video_name']][row['question_text'].strip()].add(
            f'{chr(65+answer)}. {row["multi_choice"][answer]}'.strip())
    reports=[]
    with ThreadPoolExecutor(args.shards) as pool:
        futures={pool.submit(convert_shard,root,item,args.connections,expected,qa):item for item in items}
        for job in as_completed(futures):
            reports.append(job.result())
            counts=Counter()
            for report in reports: counts.update(report['counts'])
            progress=dict(phase='downloading-and-decoding',verified_shards=len(reports),total_shards=len(items),
                verified_archive_bytes=sum(r['bytes'] for r in reports),total_archive_bytes=sum(x['size'] for x in items),
                counts=counts)
            save_json(out/'progress.json',progress)
            print(progress,flush=True)
    if sum(r['counts']['rows'] for r in reports)!=len(original):
        raise ValueError('Serialized distribution row count differs from original')
    valid={}
    remaining=sorted(clip for clip in expected if not (root/'media-index'/(clip+'.json')).is_file())
    if remaining:
        with ThreadPoolExecutor(12) as pool:
            for rec in pool.map(lambda c:acquire(root,c,mapped),remaining):
                if rec.get('errors'):
                    print('unavailable',rec['clip'],flush=True)
    missing=[]
    for clip in sorted(expected):
        meta=root/'media-index'/(clip+'.json')
        if not meta.exists():
            missing.append(clip)
            continue
        record=json_load(meta)
        path=Path(record['path'])
        if inspect(path.read_bytes())!=record['audio']:
            raise RuntimeError(f'Invalid cached WAV at final verification: {clip}')
        valid[clip]=record
    manifest=out/'official_train.jsonl'
    save_rows(manifest,(swift_row(row,valid[row['video_name']]) for row in original if row['video_name'] in valid))
    save_rows(out/'unavailable-media.jsonl',({'clip':c} for c in missing))
    missing_qas=sum(row['video_name'] not in valid for row in original)
    counts=Counter()
    for report in reports: counts.update(report['counts'])
    result=dict(stage='complete',official_repo='AlyssaYoung/AVQA',official_revision=OFFICIAL,
                official_annotation_sha256=sha(root/'sources'/'official-train.json'),archive_repo=REPO,archive_revision=REV,
                distribution_shards=78,verified_archive_bytes=sum(x['size'] for x in items),official_questions=len(original),
                official_clips=len(expected),usable_questions=len(original)-missing_qas,missing_questions=missing_qas,
                usable_clips=len(valid),missing_clips=len(missing),media_gb=sum(Path(r['path']).stat().st_size for r in valid.values())/1e9,
                sha256_manifest=sha(manifest),counts=counts,
                pilot_silence_cross_check=[c for report in reports for c in report.get('pilot_silence_cross_check',[])],
                media_sources=Counter(r.get('archive_repo') or r['url'].split('/datasets/')[1].split('/resolve/')[0] for r in valid.values()))
    save_json(out/'provenance.json',result); save_json(out/'progress.json',result)
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--shards',type=int,default=8)
    parser.add_argument('--connections',type=int,default=10)
    args=parser.parse_args()
    try: run(args)
    except BaseException as exc:
        # Avoid persisting signed CDN URLs from upstream transport exceptions.
        save_json(args.root/'full'/'failure.json',{'error_type':type(exc).__name__, 'detail':str(exc) if not isinstance(exc,requests.RequestException) else 'upstream HTTP error'})
        raise
