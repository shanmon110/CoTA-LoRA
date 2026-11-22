from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from experiments.ms_swift_aqa import prepare_pairs


def signal():
    time = np.arange(16000 * 3) / 16000
    return 0.4 * np.sin(2 * np.pi * 321 * time)


@pytest.mark.parametrize('domain', ['count', 'loudness', 'timing'])
@pytest.mark.parametrize('kind', ['change', 'invariant'])
def test_constructed_pair_has_verifiable_labels_and_equal_duration(domain, kind):
    variants = prepare_pairs.construct_pair(signal(), domain, kind, np.random.default_rng(25))
    first, second = variants
    assert len(first.samples) == len(second.samples)
    assert not np.array_equal(first.samples, second.samples)
    assert (first.answer != second.answer) == (kind == 'change')
    for variant in variants:
        assert prepare_pairs.derive_answer(variant.samples, variant.events, domain) == variant.answer


def test_manifest_preserves_question_and_option_order_and_isolates_source(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for split, offset in [('train', 0), ('dev', 5)]:
        records = []
        for i in range(offset, offset + 5):
            audio = inputs / f'{i}.wav'
            sf.write(audio, np.roll(signal(), i * 18), 16000)
            records.append(dict(id=f'orig-{i}', source_id=f'video-{i}', audios=[str(audio)]))
        (inputs / f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    destination = tmp_path / 'paired'
    prepare_pairs.build_pairs(inputs, destination, 5, 5)
    train = prepare_pairs.read_jsonl(destination / 'pair-train.jsonl')
    dev = prepare_pairs.read_jsonl(destination / 'pair-dev.jsonl')
    assert len(train) == len(dev) == 10
    assert {r['source_id'] for r in train}.isdisjoint(r['source_id'] for r in dev)
    for left, right in zip(dev[::2], dev[1::2]):
        assert left['pair_id'] == right['pair_id']
        assert left['messages'][0] == right['messages'][0]
        assert left['choices'] == right['choices']
        assert Path(left['audios'][0]).is_file()
    prepare_pairs.verify_pairs(inputs, destination)
    long_dir = tmp_path / 'long'
    prepare_pairs.build_long_pairs(inputs, destination, long_dir)
    for split in ('train', 'dev'):
        short_rows = prepare_pairs.read_jsonl(destination / f'pair-{split}.jsonl')
        long_rows = prepare_pairs.read_jsonl(long_dir / f'pair-{split}.jsonl')
        for short, long in zip(short_rows, long_rows, strict=True):
            assert short['id'] == long['id']
            assert short['messages'] == long['messages']
            assert short['choices'] == long['choices']
            assert 20 <= long['duration'] < 24
            wave_short, _ = sf.read(short['audios'][0], dtype='int16')
            wave_long, _ = sf.read(long['audios'][0], dtype='int16')
            for (start, end), (new_start, new_end) in zip(short['event_boundaries'], long['event_boundaries'], strict=True):
                np.testing.assert_array_equal(wave_short[start:end], wave_long[new_start:new_end])
            if len(short['event_boundaries']) > 1:
                assert long['event_boundaries'][-1][1] - long['event_boundaries'][0][0] > short['event_boundaries'][-1][1] - short['event_boundaries'][0][0]
    prepare_pairs.verify_pairs(inputs, long_dir)
