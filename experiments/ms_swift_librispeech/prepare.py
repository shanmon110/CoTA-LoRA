"""Convert verified LibriSpeech transcripts to local ms-swift audio chat rows."""
import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

import soundfile as sf


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(config, output):
    corpus = Path(config['corpus']).resolve(strict=True)
    processor = json.loads((Path(config['model']) / 'preprocessor_config.json').read_text())
    source = Path(config['data_source'])
    original = json.loads((source / 'splits.json').read_text())
    seen_ids, seen_speakers, info = set(), set(), {}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'data'
        stage.mkdir()
        for split, name in [('train-clean-100', 'train'), ('dev-clean', 'dev-clean'), ('test-clean', 'test-clean')]:
            path = source / f'{name}.jsonl'
            if digest(path) != original[split]['sha256']:
                raise ValueError(f'{split}: source digest mismatch')
            ids, speakers, durations = set(), set(), []
            rows = []
            for line in path.read_text().splitlines():
                item = json.loads(line)
                uid, speaker = item['id'], item['speaker']
                wav = Path(item['wav']).resolve(strict=True)
                if (wav.parents[2] != corpus / split or wav.stem != uid or wav.parent.parent.name != speaker
                        or uid in ids or not item['txt'].strip() or '<|' in item['txt']):
                    raise ValueError(f'{split}: invalid record {uid}')
                audio = sf.info(wav)
                if audio.samplerate != 16000 or audio.channels != 1 or not 0 < audio.frames <= processor["n_samples"]:
                    raise ValueError(f'{uid}: input outside native Whisper feature window')
                ids.add(uid)
                speakers.add(speaker)
                durations.append(audio.duration)
                rows.append(dict(id=uid, audios=[str(wav)], messages=[
                    dict(role='user', content='<audio>' + config['prompt']),
                    dict(role='assistant', content=item['txt'])]))
            if (len(ids) != original[split]['utterances'] or seen_ids & ids or seen_speakers & speakers):
                raise ValueError(f'{split}: count or split overlap')
            seen_ids.update(ids)
            seen_speakers.update(speakers)
            target = stage / (name + '.jsonl')
            target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            longest = max(range(len(rows)), key=durations.__getitem__)
            info[split] = dict(utterances=len(rows), speakers=len(speakers), sha256=digest(target),
                               source_sha256=digest(path), median_seconds=statistics.median(durations),
                               max_seconds=max(durations), longest_id=rows[longest]['id'])
            if name != 'test-clean':
                # Include the longest clip and typical recordings in the real-model smoke.
                typical = sorted(range(len(rows)), key=lambda i: abs(durations[i]-info[split]['median_seconds']))
                indices = [longest] + [i for i in typical if i != longest][:7]
                (stage / ('smoke-' + name + '.jsonl')).write_text(
                    ''.join(json.dumps(rows[i]) + '\n' for i in indices))
        (stage / 'splits.json').write_text(json.dumps(info, indent=2) + '\n')
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()), args.output)
