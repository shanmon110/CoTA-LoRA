"""Build controlled audio QA pairs from the existing frozen AVQA train/dev audio.

Labels describe waveform edits. They do not imply new semantic annotations of
the source recordings. Each pair uses one original clip and one fixed prompt.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from string import ascii_uppercase
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from .prompt import format_question


RATE = 16000
DOMAINS = ('count', 'loudness', 'timing')
QUESTIONS = {
    'count': {
        'train': ('How many separate sound bursts are there?',
                  'Count the distinct bursts of sound separated by silence.',
                  'How many times is the short sound played?'),
        'dev': ('How many audible repetitions occur, counting the initial one?',
                'What is the total number of sound events separated by pauses?'),
    },
    'loudness': {
        'train': ('Which of the two sound bursts is louder?',
                  'Which sound has the greater volume, the first or the second?',
                  'Between the two sound events, which is louder?'),
        'dev': ('At what position do you hear the higher-volume burst?',
                'Which occurrence of the sound is stronger in volume?'),
    },
    'timing': {
        'train': ('Which gap between sound bursts is longer?',
                  'Is there more silence between bursts one and two or between bursts two and three?',
                  'Where is the longer pause between consecutive sounds?'),
        'dev': ('Between which two consecutive bursts do you wait longer?',
                'Which pause is longer, the earlier one or the later one?'),
    },
}
ANSWERS = {
    'count': ('One', 'Two', 'Three', 'Four'),
    'loudness': ('The first burst', 'The second burst'),
    'timing': ('Between the first and second bursts', 'Between the second and third bursts'),
}


@dataclass
class Variant:
    samples: np.ndarray
    events: list
    answer: int


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    with Path(path).open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def choose_segment(samples, rng):
    # The source is never itself labelled. Favor windows that contain a
    # continuous audible sound so the known edit is easy to distinguish.
    window = int(0.64 * RATE)
    if len(samples) < window:
        raise ValueError('Source is shorter than a burst')
    for start in rng.integers(0, len(samples) - window + 1, 64):
        segment = samples[start:start + window].copy()
        rms_chunks = [rms(c) for c in np.array_split(segment, 8)]
        if min(rms_chunks) >= 0.0005 and min(rms_chunks) >= 0.09 * np.median(rms_chunks):
            segment *= min(0.65 / np.max(np.abs(segment)), 0.13 / rms(segment))
            fade = int(0.008 * RATE)
            ramp = np.linspace(0, 1, fade)
            segment[:fade] *= ramp
            segment[-fade:] *= ramp[::-1]
            if rms(segment) >= 0.035:
                return segment
    raise ValueError('Source does not contain a sufficiently continuous sound')


def render(segment, length, leading, gaps, amplitudes):
    waveform = np.zeros(length, dtype=np.float32)
    events = []
    offset = leading
    for i, amplitude in enumerate(amplitudes):
        end = offset + len(segment)
        waveform[offset:end] = segment * amplitude
        events.append([offset, end])
        if i < len(gaps):
            offset = end + gaps[i]
    return waveform, events


def derive_answer(waveform, events, domain):
    """Verify the label from decoded samples and the declared burst boundaries."""
    stops = [end for _, end in events]
    starts = [start for start, _ in events]
    if not 0 < len(events) <= 4 or any(not 0 <= s < e <= len(waveform) for s, e in events):
        raise ValueError('Invalid event boundaries')
    if any(s - e < 0.15 * RATE for s, e in zip(starts[1:], stops)):
        raise ValueError('Events must have a measurable gap')
    if len({e - s for s, e in events}) != 1:
        raise ValueError('Events must have equal lengths')
    activity = np.abs(waveform).copy()
    energy = []
    for start, end in events:
        energy.append(rms(waveform[start:end]))
        activity[start:end] = 0
    if min(energy) < 0.004 or np.max(activity) > 1e-5:
        raise ValueError('Expected audible events separated by digital silence')
    if domain == 'count':
        return len(events) - 1
    if domain == 'loudness' and len(events) == 2:
        if max(energy) / min(energy) >= 2:
            return int(np.argmax(energy))
    if domain == 'timing' and len(events) == 3:
        gaps = np.asarray(starts[1:]) - np.asarray(stops[:-1])
        if max(gaps) / min(gaps) >= 2:
            return int(np.argmax(gaps))
    raise ValueError('Edited waveform has no unambiguous answer')


def construct_pair(samples, domain, kind, rng):
    if domain not in DOMAINS or kind not in ('change', 'invariant'):
        raise ValueError('Unrecognized diagnostic domain or pair kind')
    segment = choose_segment(np.asarray(samples, dtype=np.float32), rng)
    count = len(ANSWERS[domain])
    answers = rng.choice(count, size=2, replace=False).tolist() if kind == 'change' else [int(rng.integers(count))] * 2
    length = int(rng.integers(96000, 117000))
    leading = int(rng.integers(6400, 14700))
    gap = int(rng.integers(6200, 9300))
    short_gap = int(rng.integers(3000, 5300))
    long_gap = int(rng.integers(13800, 17400))
    low_volume = float(rng.uniform(0.21, 0.36))
    variants = []
    for member, answer in enumerate(answers):
        offset = leading
        if domain == 'count':
            amplitudes = [1.0] * (answer + 1)
            gaps = [gap] * answer
        elif domain == 'loudness':
            amplitudes = [1.0, low_volume] if answer == 0 else [low_volume, 1.0]
            gaps = [gap]
        else:
            amplitudes = [1.0] * 3
            gaps = [long_gap, short_gap] if answer == 0 else [short_gap, long_gap]
        if kind == 'invariant' and member == 1:
            offset += int(rng.integers(1300, 3200))
            gain = float(rng.uniform(0.66, 0.83))
            amplitudes = [a * gain for a in amplitudes]
        waveform, events = render(segment, length, offset, gaps, amplitudes)
        if derive_answer(waveform, events, domain) != answer:
            raise ValueError('Edit does not realize its intended answer')
        variants.append(Variant(waveform, events, answer))
    return variants


def get_sources(inputs):
    sources = {split: read_jsonl(inputs / f'{split}.jsonl') for split in ('train', 'dev')}
    for split, rows in sources.items():
        if not rows or len({row['id'] for row in rows}) != len(rows) or any(
                not row['source_id'] or len(row['audios']) != 1 for row in rows):
            raise ValueError(f'Invalid source manifest for {split}')
    for values in (lambda r: r['source_id'], lambda r: Path(r['audios'][0]).resolve()):
        if not {values(r) for r in sources['train']}.isdisjoint(values(r) for r in sources['dev']):
            raise ValueError('Original train/dev source leakage')
    return sources


def build_split(stage, destination, rows, split, count, rng):
    created = []
    used = set()
    for row_index in rng.permutation(len(rows)):
        parent = rows[row_index]
        source = parent['source_id']
        if source in used:
            continue
        pair_index = len(used)
        domain = DOMAINS[pair_index % len(DOMAINS)]
        kind = 'invariant' if pair_index % 5 == 4 else 'change'
        path = Path(parent['audios'][0]).resolve()
        samples, rate = sf.read(path, dtype='float32')
        if rate != RATE or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f'Expected finite 16 kHz mono source {path}')
        try:
            pair = construct_pair(samples, domain, kind, rng)
        except ValueError as error:
            if str(error) not in ('Source does not contain a sufficiently continuous sound',
                                  'Source is shorter than a burst'):
                raise
            continue
        pair_id = f'controlled-avqa-{split}-{pair_index:04d}'
        question = str(rng.choice(QUESTIONS[domain][split]))
        choices = rng.permutation(ANSWERS[domain]).tolist()
        prompt = {'role': 'user', 'content': format_question(question, choices)}
        source_digest = digest(path)
        for member, variant in enumerate(pair):
            example_id = f'{pair_id}-{member}'
            relative = Path('wav') / split / f'{example_id}.wav'
            sf.write(stage / relative, variant.samples, RATE, subtype='PCM_16')
            label = ascii_uppercase[choices.index(ANSWERS[domain][variant.answer])]
            created.append(dict(
                id=example_id, pair_id=pair_id, pair_member=member, pair_kind=kind,
                source_id=source, source_ids=[source], source_example_id=parent['id'], source_audio_sha256=source_digest,
                question=question, choices=choices, domain=domain, audios=[str(destination / relative)],
                messages=[prompt, {'role': 'assistant', 'content': label}],
                audio_sha256=digest(stage / relative), event_boundaries=variant.events,
                duration=len(variant.samples) / RATE,
            ))
        used.add(source)
        if len(used) == count:
            break
    if len(used) != count:
        raise ValueError(f'Only {len(used)} suitable distinct sources in {split}. Expected {count}.')
    return created


def verify_split(rows, parents, split, stage, destination):
    groups = defaultdict(list)
    paths, audio_digests, parent_digests = set(), set(), {}
    labels = Counter()
    for row in rows:
        parent = parents[row['source_example_id']]
        source = parent['source_id']
        if row['source_id'] != source or row['source_ids'] != [source]:
            raise ValueError('Pair provenance differs from original source')
        if source not in parent_digests:
            parent_digests[source] = digest(parent['audios'][0])
        if row['source_audio_sha256'] != parent_digests[source]:
            raise ValueError('Original waveform changed')
        path = Path(row['audios'][0])
        if path in paths or row['audio_sha256'] in audio_digests:
            raise ValueError('Repeated generated audio')
        paths.add(path)
        audio_digests.add(row['audio_sha256'])
        local = stage / path.relative_to(destination)
        waveform, rate = sf.read(local, dtype='float32')
        if (rate != RATE or waveform.ndim != 1 or len(waveform) != round(row['duration'] * RATE)
                or digest(local) != row['audio_sha256']):
            raise ValueError(f'Generated waveform does not match its manifest: {local}')
        domain = row['domain']
        if row['question'] not in QUESTIONS[domain][split] or set(row['choices']) != set(ANSWERS[domain]):
            raise ValueError('Prompt inventory or choices differ from the generating protocol')
        answer = derive_answer(waveform, row['event_boundaries'], domain)
        label = ascii_uppercase[row['choices'].index(ANSWERS[domain][answer])]
        messages = [{'role': 'user', 'content': format_question(row['question'], row['choices'])},
                    {'role': 'assistant', 'content': label}]
        if row['messages'] != messages:
            raise ValueError('Manifest answer disagrees with the independently checked edit')
        labels[f'{domain}/{label}'] += 1
        groups[row['pair_id']].append(row)
    for group in groups.values():
        if sorted(r['pair_member'] for r in group) != [0, 1]:
            raise ValueError('Each pair must contain exactly two unique members')
        left, right = group
        for key in ('pair_kind', 'source_id', 'domain', 'question', 'choices', 'duration'):
            if left[key] != right[key]:
                raise ValueError(f'Pair does not match in {key}')
        changed = left['messages'][1] != right['messages'][1]
        if left['pair_kind'] != ('change' if changed else 'invariant'):
            raise ValueError('Label relation does not match pair kind')
    counts = Counter(f"{rs[0]['domain']}/{rs[0]['pair_kind']}" for rs in groups.values())
    return dict(rows=len(rows), pairs=len(groups), by_domain_and_kind=dict(counts), label_counts=dict(labels),
                min_duration=min(r['duration'] for r in rows), max_duration=max(r['duration'] for r in rows))


def write_pair_corpus(inputs, destination, make_rows, audit):
    sources = get_sources(inputs)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = ['wav', 'pair-train.jsonl', 'pair-dev.jsonl', 'pair-audit.json']
    if any((destination / name).exists() for name in files):
        raise FileExistsError(f'Existing pair outputs at {destination}')
    audit.update(source_directory=str(inputs.resolve()), sample_rate=RATE,
                 train_dev_source_and_original_path_overlap=0, splits={})
    all_source_audio = []
    all_edited_audio = []
    with TemporaryDirectory(dir=destination, prefix='.pairs-') as temp:
        stage = Path(temp)
        for split in ('train', 'dev'):
            (stage / 'wav' / split).mkdir(parents=True)
            rows = make_rows(stage, destination, split, sources[split])
            file = stage / f'pair-{split}.jsonl'
            file.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
            info = verify_split(rows, {r['id']: r for r in sources[split]}, split, stage, destination)
            info.update(manifest_sha256=digest(file), source_manifest_sha256=digest(inputs / f'{split}.jsonl'))
            audit['splits'][split] = info
            all_source_audio.append({r['source_audio_sha256'] for r in rows})
            all_edited_audio.append({r['audio_sha256'] for r in rows})
        if all_source_audio[0] & all_source_audio[1] or all_edited_audio[0] & all_edited_audio[1]:
            raise ValueError('Train/dev overlap in original or generated audio bytes')
        (stage / 'pair-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        for name in files:
            (stage / name).rename(destination / name)
    return audit


def build_pairs(inputs, destination, train_pairs=320, dev_pairs=120, seed=42):
    if min(train_pairs, dev_pairs) < 1:
        raise ValueError('Each split must have at least one pair')
    counts = dict(train=train_pairs, dev=dev_pairs)

    def make_rows(stage, output, split, source_rows):
        rng = np.random.default_rng(seed + (split == 'dev'))
        return build_split(stage, output, source_rows, split, counts[split], rng)

    audit = dict(protocol='Controlled waveform editing of previously frozen AVQA sources. Labels describe edits.',
                 seed=seed)
    return write_pair_corpus(inputs, destination, make_rows, audit)


def stretch_split(stage, output, split, rows, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_id']].append(row)
    stretched = []
    for members in groups.values():
        length = int(rng.integers(20 * RATE, 24 * RATE))
        for row in members:
            original = Path(row['audios'][0])
            short, rate = sf.read(original, dtype='int16')
            if rate != RATE or short.ndim != 1 or digest(original) != row['audio_sha256']:
                raise ValueError(f'Changed source audio {original}')
            long = np.zeros(length, dtype=np.int16)
            events = []
            offset = row['event_boundaries'][0][0]
            previous_end = None
            for start, end in row['event_boundaries']:
                if previous_end is not None:
                    gap = start - previous_end
                    if row['domain'] == 'loudness':
                        offset += gap + 9 * RATE
                    else:
                        offset += gap * (9 if row['domain'] == 'count' else 8)
                new_end = offset + end - start
                if new_end > length:
                    raise ValueError(f"Stretched event exceeds clip length for {row['id']}")
                long[offset:new_end] = short[start:end]
                events.append([offset, new_end])
                offset, previous_end = new_end, end
            relative = Path('wav') / split / original.name
            sf.write(stage / relative, long, RATE, subtype='PCM_16')
            stretched.append(dict(row, audios=[str(output / relative)], event_boundaries=events,
                                  audio_sha256=digest(stage / relative), duration=length / RATE,
                                  short_audio_sha256=row['audio_sha256']))
    return stretched


def build_long_pairs(inputs, short_destination, long_destination, seed=43):
    short_destination = short_destination.resolve()
    short_audit = verify_pairs(inputs, short_destination)

    def make_rows(stage, output, split, _source_rows):
        rows = read_jsonl(short_destination / f'pair-{split}.jsonl')
        rng = np.random.default_rng(seed + (split == 'dev'))
        return stretch_split(stage, output, split, rows, rng)

    audit = dict(protocol='Long controlled waveform pairs. Original burst PCM is copied exactly. '
                          'Count gaps ×9, timing gaps ×8, and loudness gap plus 9 seconds. '
                          'Within each pair the clip duration is equal and sampled independently of the label.',
                 seed=seed, short_source=str(short_destination),
                 short_manifest_sha256={split: info['manifest_sha256']
                                        for split, info in short_audit['splits'].items()})
    return write_pair_corpus(inputs, long_destination, make_rows, audit)


def verify_pairs(inputs, destination):
    destination = destination.resolve()
    sources = get_sources(inputs)
    audit = json.loads((destination / 'pair-audit.json').read_text())
    waveform_hashes = []
    source_hashes = []
    for split in ('train', 'dev'):
        path = destination / f'pair-{split}.jsonl'
        rows = read_jsonl(path)
        info = verify_split(rows, {r['id']: r for r in sources[split]}, split, destination, destination)
        info.update(manifest_sha256=digest(path), source_manifest_sha256=digest(inputs / f'{split}.jsonl'))
        if info != audit['splits'][split]:
            raise ValueError(f'Split audit failed for {split}')
        waveform_hashes.append({r['audio_sha256'] for r in rows})
        source_hashes.append({r['source_audio_sha256'] for r in rows})
    if waveform_hashes[0] & waveform_hashes[1] or source_hashes[0] & source_hashes[1]:
        raise ValueError('Train/dev overlap in original or generated audio bytes')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-pairs', type=int, default=320)
    parser.add_argument('--dev-pairs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--stretch-from', type=Path, help='Derive time-separated pairs from this verified pair corpus')
    args = parser.parse_args()
    if args.verify:
        result = verify_pairs(args.source, args.output)
    elif args.stretch_from:
        result = build_long_pairs(args.source, args.stretch_from, args.output, args.seed)
    else:
        result = build_pairs(args.source, args.output, args.train_pairs, args.dev_pairs, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
