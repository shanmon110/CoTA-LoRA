"""Check media overlap before training and save immutable manifests shared by the three arms."""
import argparse
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def audio_key(path):
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError(f'Invalid audio: {path}')
    mono = samples.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor)
    pcm = np.clip(np.rint(mono * 32768), -32768, 32767).astype('<i2')
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def freeze(train, dev, test, dest):
    paths = dict(train=train, dev=dev, test=test)
    rows = {k: [json.loads(line) for line in p.read_text().splitlines()] for k, p in paths.items()}
    sources = {k: {r['source_id'] for r in rows[k]} for k in ('train', 'dev')}
    if sources['train'] & sources['dev']:
        raise ValueError('AVQA source videos overlap between train and dev')
    for k, group in rows.items():
        if not group or len({r['id'] for r in group}) != len(group):
            raise ValueError(f'Empty split or duplicate question IDs within {k}')
        if any(len(r['audios']) != 1 for r in group):
            raise ValueError(f'Expected one audio per question in {k}')
    media = sorted({r['audios'][0] for group in rows.values() for r in group})
    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes = dict(zip(media, pool.map(audio_key, media), strict=True))
    forbidden = set()
    record = {'fingerprint': 'SHA256 of mono 16 kHz rounded int16 PCM; different edits/codecs may evade detection',
              'excluded': [], 'splits': {}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        staging = Path(tmp) / 'data'
        staging.mkdir()
        for k in ('test', 'dev', 'train'):
            group = rows[k]
            duplicate_sources = {r['source_id'] for r in group if hashes[r['audios'][0]] in forbidden} if k != 'test' else set()
            retained = []
            for row in group:
                if k != 'test' and row['source_id'] in duplicate_sources:
                    record['excluded'].append(dict(split=k, id=row['id'], source_id=row['source_id'],
                                                   reason='source contains matching PCM in held-out split'))
                else:
                    retained.append(row)
            if not retained:
                raise ValueError(f'No retained examples in {k}')
            digests = {hashes[r['audios'][0]] for r in retained}
            assert digests.isdisjoint(forbidden)
            forbidden.update(digests)
            p = staging / (k + '.jsonl')
            p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in retained))
            record['splits'][k] = dict(source=str(paths[k].resolve()), source_sha256=hashlib.sha256(paths[k].read_bytes()).hexdigest(),
                                       input_questions=len(group), questions=len(retained), unique_pcm=len(digests),
                                       choices=dict(Counter(len(r['choices']) for r in retained)),
                                       sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        (staging / 'audio-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
        (staging / 'audit.json').write_text(json.dumps(record, indent=2) + '\n')
        staging.rename(dest)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'dev', 'test', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(freeze(a.train, a.dev, a.test, a.output), indent=2))
