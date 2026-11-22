"""Join official MMAU Test-mini labels to embedded waveforms by UUID, then verify each recording."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from string import ascii_uppercase
import uuid

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .download_mmau import GITHUB_REVISION, HF_REPO, HF_REVISION, SOURCE_HASHES, digest
from .prompt import format_question


def prepare(root):
    root = root.resolve()
    for name, expected in SOURCE_HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(f'Source digest mismatch: {name}')
    originals = json.loads((root / 'mmau-test-mini.json').read_text())
    by_id = {row['id']: row for row in originals}
    if len(by_id) != len(originals) or len(by_id) != 1000:
        raise ValueError('Incomplete or duplicate MMAU Test-mini annotations')
    audio_dir = root / 'audio'
    audio_dir.mkdir(exist_ok=True)
    manifest = {}
    encoded_formats = Counter()
    hf_label_differences = []
    for batch in pq.ParquetFile(root / 'test_mini.parquet').iter_batches(batch_size=8):
        for packaged in batch.to_pylist():
            attrs = json.loads(packaged['other_attributes'])
            uid = attrs['id']
            if str(uuid.UUID(uid)) != uid or uid in manifest:
                raise ValueError(f'Invalid or duplicate audio example ID: {uid}')
            original = by_id[uid]
            if attrs['task'] != original['task'] or attrs['split'] != 'test-mini':
                raise ValueError(f'Task/split mismatch: {uid}')
            choices = original['choices']
            letters = ascii_uppercase[:len(choices)]
            gold = [letter for letter, choice in zip(letters, choices) if choice == original['answer']]
            if not gold:
                raise ValueError(f'Gold is absent from the official choices: {uid}')
            hf_options = packaged['choices']
            expected_options = [f'({letter}) {text}' for letter, text in zip(letters, choices)]
            expected_answers = {f'({letter}) {original["answer"]}' for letter in gold}
            diffs = []
            if packaged['instruction'] != original['question']:
                diffs.append('question')
            if hf_options != expected_options:
                diffs.append('choices')
            if packaged['answer'] not in expected_answers:
                diffs.append('answer')
            if diffs:
                hf_label_differences.append({'id': uid, 'fields': diffs,
                                             'hf_question': packaged['instruction'], 'hf_choices': hf_options,
                                             'hf_answer': packaged['answer']})
            audio = packaged['context']
            blob = audio['bytes']
            audio_sha = sha256(blob).hexdigest()
            with sf.SoundFile(BytesIO(blob)) as stream:
                samples = stream.read(dtype='float32', always_2d=True)
                rate = stream.samplerate
                channels = stream.channels
                frames = len(samples)
                fmt = stream.format
                nominal_frames = len(stream)
                if frames == 0 or not np.isfinite(samples).all() or (fmt != 'MP3' and frames != nominal_frames):
                    raise ValueError(f'Invalid waveform: {uid}')
            encoded_formats[fmt] += 1
            audio_path = audio_dir / (uid + '.' + fmt.lower())
            if audio_path.exists():
                if digest(audio_path) != audio_sha:
                    raise ValueError(f'Unexpected existing audio content: {audio_path}')
            else:
                temporary = audio_path.with_suffix('.part')
                temporary.write_bytes(blob)
                temporary.replace(audio_path)
            prompt = format_question(original['question'], choices)
            manifest[uid] = dict(id=uid, source_id='audio-sha256:' + audio_sha, question=original['question'], choices=choices,
                                 correct_letters=gold, answer_text=original['answer'],
                                 messages=[{'role': 'user', 'content': prompt},
                                           {'role': 'assistant', 'content': gold[0]}],
                                 audios=[str(audio_path)], domain=original['task'], category=original['category'],
                                 sub_category=original['sub-category'], difficulty=original['difficulty'],
                                 source_dataset=original['dataset'], audio_sha256=audio_sha,
                                 original_audio_path=original['audio_id'], hf_audio_path=audio['path'],
                                 duration=frames / rate, frames=frames, nominal_frames=nominal_frames, sample_rate=rate, channels=channels)
    if by_id.keys() != manifest.keys():
        raise ValueError('Official and packaged ID sets differ')
    dest = root / 'test.jsonl'
    temporary = root / 'test.jsonl.part'
    with temporary.open('w') as out:
        for uid in by_id:
            out.write(json.dumps(manifest[uid], ensure_ascii=False) + '\n')
    temporary.replace(dest)
    durations = [r['duration'] for r in manifest.values()]
    report = dict(dataset='MMAU-v05.15.25 Test-mini', repo=HF_REPO, hf_revision=HF_REVISION,
                  labels_and_evaluator=f'https://github.com/Sakshi113/MMAU/tree/{GITHUB_REVISION}',
                  source_sha256=SOURCE_HASHES, test_sha256=digest(dest), items=len(manifest),
                  decoded_waveforms=len(manifest), distinct_encoded_waveforms=len({r['audio_sha256'] for r in manifest.values()}),
                  domains=Counter(r['domain'] for r in manifest.values()), formats=encoded_formats,
                  native_sample_rates=Counter(r['sample_rate'] for r in manifest.values()),
                  duration_seconds=dict(total=sum(durations), min=min(durations), max=max(durations)),
                  option_counts=Counter(len(r['choices']) for r in manifest.values()),
                  rows_with_duplicate_choices=sum(len(set(r['choices'])) != len(r['choices']) for r in manifest.values()),
                  rows_with_repeated_gold=sum(len(r['correct_letters']) > 1 for r in manifest.values()),
                  mp3_frame_estimate_differs=sum(r['frames'] != r['nominal_frames'] for r in manifest.values()),
                  id_semantics='Public item UUID. source_id groups identical encoded media by SHA256; original recording identifiers are unavailable.',
                  annotations='Pinned official GitHub JSON; Parquet and GitHub joined by UUID in GitHub row order.',
                  hf_annotation_differences=hf_label_differences)
    (root / 'provenance.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'hf_annotation_differences'}))
    print('Annotation differences between source formats:', len(hf_label_differences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    prepare(parser.parse_args().root)
