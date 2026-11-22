"""Verify completed AVQA archive metadata locally and publish an idempotent coverage record."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import time


ANNOTATION_SHA = '55ec202636c3e25f279995eb68ddd14264004cbe0d042d1f559b03662a8575ee'
OFFICIAL_COMMIT = 'a5056c1d82849aa0981f50fa4c5a57bd16658c52'
PILOT_SHA = {'train': '7b1c681c208772ae071078b75dfedef55db3bf3893b35a5cf7fa8c8142a0d07d',
             'dev': '64b0976c45a56c7406b30935b49367f464e1e22d9bce295b5e0ac8505e288d9f',
             'test': 'f1b4ebdca86addbc05f3626d32d7fe979b44cf953d0d079ca390b4fefb739612'}
MARKER = '<!-- avqa-official-archive-v1 -->'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def lines(path):
    with path.open() as stream:
        for text in stream:
            yield json.loads(text)


def write(path, obj):
    tmp = path.with_suffix('.verification-tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def pinned_inputs(data):
    official = data/'avqa/sources/official-train.json'
    require(digest(official) == ANNOTATION_SHA, 'Official AVQA annotation hash changed')
    original = load(official)
    by_id = {r['id']: r for r in original}
    require(len(by_id) == len(original) == 40425, 'Unexpected official question inventory')
    frozen = data/'data/pilot-01'
    observed = {split: digest(frozen/(split + '.jsonl')) for split in PILOT_SHA}
    require(observed == PILOT_SHA, 'Frozen pilot manifests changed')
    return by_id


def check_available(rows, by_id, audio_dir):
    seen, clips = set(), set()
    for row in rows:
        qid = row['original_id']
        require(qid in by_id and qid not in seen, f'Duplicate/unknown QA {qid}')
        seen.add(qid)
        source = by_id[qid]
        clip = source['video_name']
        options = source['multi_choice']
        letter = chr(ord('A') + source['answer'])
        expected_prompt = '\n'.join(['<audio>', source['question_text'],
                                   *(f'{chr(65+i)}. {t}' for i, t in enumerate(options)),
                                   'Reply with the letter of the correct option.'])
        require(row['id'] == f'avqa-{qid}' and row['question'] == source['question_text'] and row['choices'] == options
                and row['messages'] == [{'role': 'user', 'content': expected_prompt}, {'role': 'assistant', 'content': letter}],
                f'Annotation, user prompt, or gold disagreement at {qid}')
        require(row['clip_id'] == clip and row['source_id'] == clip.rsplit('_', 1)[0]
                and row['question_relation'] == source['question_relation'] and row['question_type'] == source['question_type'],
                f'Source/category mismatch at {qid}')
        path = audio_dir/(clip + '.wav')
        require(row['audios'] == [str(path)], f'Unexpected WAV path at {qid}')
        if clip not in clips:
            require(path.is_file() and path.stat().st_size > 44, f'WAV missing or empty: {path}')
            clips.add(clip)
    return seen, clips


def verify(data):
    original = pinned_inputs(data)
    full = data/'avqa/full'
    manifest = full/'official_train.jsonl'
    missing_file = full/'unavailable-questions.jsonl'
    supervisor = load(full/'audit.json')
    upstream = load(full/'provenance.json')
    require(upstream['stage'] == 'complete' and upstream['official_revision'] == OFFICIAL_COMMIT
            and upstream['official_annotation_sha256'] == ANNOTATION_SHA, 'Source provenance mismatch')
    seen, clips = check_available(lines(manifest), original, data/'avqa/audio')
    all_clips = {r['video_name'] for r in original.values()}
    expected_missing = original.keys() - seen
    missing_ids, missing_clips = set(), set()
    for row in lines(missing_file):
        key = row['id']
        require(key in expected_missing and key not in missing_ids and row['video_name'] == original[key]['video_name'],
                f'Unknown/duplicate unavailable question: {key}')
        missing_ids.add(key)
        missing_clips.add(row['video_name'])
    recorded_clips = Counter(r['clip'] for r in lines(full/'unavailable-media.jsonl'))
    require(missing_ids == expected_missing and missing_clips.isdisjoint(clips) and missing_clips == all_clips - clips,
            'Available and unavailable annotations do not partition official AVQA')
    require(recorded_clips == Counter(missing_clips), 'Missing media inventory disagrees with missing annotations')
    manifest_hash = digest(manifest)
    counts = dict(official_questions=len(original), usable_questions=len(seen), usable_clips=len(clips))
    require(all(upstream[k] == v == supervisor[k] for k, v in counts.items()), 'Coverage count mismatch')
    require(upstream['missing_questions'] == supervisor['unavailable_questions'] == len(missing_ids)
            and upstream['missing_clips'] == supervisor['unavailable_clips'] == len(missing_clips)
            and upstream['sha256_manifest'] == supervisor['manifest_sha256'] == manifest_hash, 'Archive checks disagree')
    return dict(status='verified', **counts, official_clips=len(all_clips), unavailable_questions=len(missing_ids),
                unavailable_clips=len(missing_clips), manifest=str(manifest), missing=str(missing_file),
                audio_check=str(full/'audit.json'), manifest_sha256=manifest_hash, official_revision=OFFICIAL_COMMIT,
                official_annotation_sha256=ANNOTATION_SHA, missing_question_list_sha256=digest(missing_file),
                frozen_pilot_manifest_sha256=PILOT_SHA)


def publish(report, verification, value):
    paragraph = (f"\n\n{MARKER}\n## AVQA 全量训练数据归档\n\n"
                 f"已核对官方 {value['official_questions']:,} 道训练题。可用 {value['usable_questions']:,} 道，"
                 f"对应 {value['usable_clips']:,} 条独立音频片段。另有 {value['unavailable_questions']:,} 道题"
                 f"（{value['unavailable_clips']:,} 条片段）缺少音频。已核对冻结的训练、开发和测试清单哈希。"
                 f"[全量可用清单]({value['manifest']}) · [缺失题号]({value['missing']}) · [归档校验]({verification})\n")
    # Only detect this section's marker. Model metrics are never interpreted or modified.
    with report.open('r+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not any(MARKER.encode() in line for line in stream):
            stream.seek(0, 2)
            stream.write(paragraph.encode())


def watch(data, report, interval):
    full = data/'avqa/full'
    dest = data/'archive-verification.json'
    with (data/'archive-verification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pinned_inputs(data)
        previous = None
        while True:
            status = load(full/'transfer-status.json') if (full/'transfer-status.json').exists() else {}
            stage = status.get('stage')
            if stage != previous:
                print('supervisor', stage, flush=True)
                previous = stage
            if stage in ('failed', 'audit-failed'):
                write(dest, dict(status='failed', supervisor_stage=stage, reason=status.get('reason'), source=str(full/'transfer-status.json')))
                raise RuntimeError(f'Archive supervisor reported {stage}')
            if stage == 'complete':
                value = verify(data)
                write(dest, value)
                publish(report, dest, value)
                print('verified', value['usable_questions'], value['usable_clips'], flush=True)
                return
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['dry', 'watch'])
    parser.add_argument('--data', type=Path, default=Path('/root/autodl-tmp/r1-aqa'))
    parser.add_argument('--report', type=Path, default=Path(__file__).with_name('REPORT.md'))
    parser.add_argument('--interval', type=int, default=20)
    args = parser.parse_args()
    data = args.data.resolve()
    if args.action == 'dry':
        original = pinned_inputs(data)
        ids, clips = check_available(lines(data/'avqa/pilot/dev.jsonl'), original, data/'avqa/audio')
        print(f'Pinned official and pilot hashes match. Checked {len(ids)} questions and {len(clips)} clipped WAV files.')
    else:
        try:
            watch(data, args.report.resolve(strict=True), args.interval)
        except (ValueError, KeyError, OSError) as error:
            if isinstance(error, BlockingIOError):
                raise
            write(data/'archive-verification.json', dict(status='verification-failed', reason=f'{type(error).__name__}: {error}'))
            raise
