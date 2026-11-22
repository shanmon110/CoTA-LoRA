"""Re-read the audio behind both targets and check for identical 16 kHz mono PCM across splits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import combinations
import json
from pathlib import Path

from ..freeze_data import audio_key
from .report import replace


def audit(protocol):
    data = Path(protocol['avqa_mmau'])
    manifests = {k:data/f'{k}.jsonl' for k in ('train','dev','test')}
    manifests['ser'] = Path(protocol['ser'])
    rows = {}
    for name, path in manifests.items():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol['hashes'][name]:
            raise ValueError(f'Changed {name} manifest')
        rows[name] = [json.loads(line) for line in raw.splitlines()]
    paths = sorted({r['audios'][0] for group in rows.values() for r in group})
    with ThreadPoolExecutor(max_workers=4) as pool:
        checksums = dict(zip(paths,pool.map(audio_key,paths),strict=True))
    prior = json.loads((data/'audio-hashes.json').read_text())
    for name in ('train','dev','test'):
        for item in rows[name]:
            path = item['audios'][0]
            if checksums[path] != prior[path]:
                raise ValueError(f"Prior decoded-audio checksum changed for {item['id']}")
    unique = {name:{checksums[r['audios'][0]] for r in group} for name,group in rows.items()}
    overlaps = {f'{a} / {b}':len(unique[a] & unique[b]) for a,b in combinations(rows,2)}
    source_train = {r['source_id'] for r in rows['train']}
    source_dev = {r['source_id'] for r in rows['dev']}
    if any(overlaps.values()) or source_train & source_dev:
        raise ValueError(f'Splits contain overlapping media: {overlaps}')
    sessions = sorted({int(r['session']) for r in rows['ser']})
    if sessions != [5]:
        raise ValueError(f'Unexpected final IEMOCAP sessions {sessions}')
    return dict(manifest_sha256=protocol['hashes'], pcm='16 kHz mono rounded PCM signed 16-bit little-endian',
                split_rows={name:len(group) for name,group in rows.items()},
                unique_pcm={name:len(keys) for name,keys in unique.items()},
                split_pcm_intersections=overlaps, avqa_train_dev_common_source_videos=0,
                ser_sessions=sessions, ser_reference_counts=dict(Counter(r['messages'][1]['content'] for r in rows['ser'])))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    out = cli.parse_args().out
    result = audit(json.loads((out/'protocol.json').read_text()))
    replace(out/'split-audio-audit.json',json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
