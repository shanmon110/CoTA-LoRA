import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from experiments.hearside_pilot import probes                      # noqa: E402
from experiments.trr_diagnostics import d2                         # noqa: E402

LAYERS, WIDTH = 3, 12                                              # stored states: the layer-0 input and three layers


def states(folder, n, seed, signal, clusters=6):
    """Random states; `signal[site]` names the stored layer whose first four features carry the class."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    folder.mkdir(parents=True)
    for site in probes.SITES:
        x = rng.normal(size=(n, LAYERS + 1, WIDTH))
        if signal.get(site) is not None:
            x[np.arange(n), signal[site], labels] += 4.0
        np.save(folder / f'{site}.npy', x.astype(np.float16))
    d2.write_jsonl(folder / 'rows.jsonl', [dict(id=f'u{seed}-{i}', label='ABCD'[l], cluster=f'c{i % clusters}', answer='ABCD'[(l + (i % 2)) % 4])
                                            for i, l in enumerate(labels)])
    d2.write(folder / 'done.json', dict(rows=n, layers=LAYERS, width=WIDTH))
    return labels


def model(root, name, signal):
    for seed, (split, n) in enumerate(dict(train=400, dev=200, session4=240, cremad=240).items()):
        states(root / name / split, n, seed, signal)
    return probes.fit(root / name, root / name / 'probes', device='cpu')


def given(root, folder, arm, share):
    """Letter answers under four option orders, right in `share` of the rows."""
    rng = np.random.default_rng(len(arm))
    for split in probes.TESTS:
        rows = d2.read_jsonl(root / 'frozen-base' / split / 'rows.jsonl')
        out = [dict(id=f"{r['id']}#r{k}", reference='A', prediction='A' if rng.random() < share else 'B') for r in rows for k in range(4)]
        path = folder / split / 'runs' / arm / 'mcq'
        path.mkdir(parents=True)
        d2.write_jsonl(path / 'predictions.jsonl', out)


def test_probe_finds_a_planted_signal_at_its_layer_and_the_control_stays_at_chance(tmp_path):
    chosen = model(tmp_path, 'frozen-base', dict(audio=2, answer=None))
    info = d2.load(tmp_path / 'frozen-base' / 'probes' / 'probes.json')
    assert chosen['audio'] == 2 and info['table']['audio/2']['session4'] > 0.9 and info['table']['audio/1']['session4'] < 0.45
    assert info['table']['audio/2']['session4_control'] < 0.45 and info['table'][f"answer/{chosen['answer']}"]['cremad'] < 0.45
    correct = np.load(tmp_path / 'frozen-base' / 'probes' / 'correct.npz')
    assert correct['audio/2/session4'].shape == (240,) and correct['audio/2/session4'].mean() > 0.9


def test_readings_gap_present_and_information_not_at_the_answer_position(tmp_path):
    model(tmp_path, 'frozen-base', dict(audio=2, answer=None))
    model(tmp_path, 'A-all', dict(audio=2, answer=3))
    given(tmp_path, tmp_path / 'given', 'frozen-base', 0.5)
    given(tmp_path, tmp_path / 'given', 'A-all', 0.95)
    result = probes.summarize(tmp_path, {s: tmp_path / 'given' / s for s in probes.TESTS}, resamples=300)
    entry = result['corpora']['session4']
    assert entry['readings'] == dict(perception_use_gap='present', information_reaches_the_answer_position='no',
                                     what_full_depth_training_adds='no more than a linear readout of the base')
    assert entry['base_best_site'] == 'audio' and abs(entry['models']['frozen-base']['answer_all_orders'] - 50) < 8
    assert entry['models']['A-all']['probe']['answer'] > 90 and 'frozen-base' in probes.render(result)
    assert abs(entry['models']['frozen-base']['answer_training_order'] - 50) < 12          # the answers stored with the states


def test_readings_no_gap_when_the_states_carry_nothing(tmp_path):
    model(tmp_path, 'frozen-base', dict(audio=None, answer=None))
    model(tmp_path, 'A-all', dict(audio=None, answer=None))
    given(tmp_path, tmp_path / 'given', 'frozen-base', 0.6)
    given(tmp_path, tmp_path / 'given', 'A-all', 0.95)
    readings = probes.summarize(tmp_path, {s: tmp_path / 'given' / s for s in probes.TESTS}, resamples=300)['corpora']['cremad']['readings']
    assert readings['perception_use_gap'] == 'absent' and readings['information_reaches_the_answer_position'] == 'yes'
    assert readings['what_full_depth_training_adds'] == 'more than a linear readout of the base'


def test_resampler_scores_mean_per_class_recall(tmp_path):
    labels, clusters = np.array([0, 0, 1, 2, 3, 3]), list('aabbcc')
    sampler = probes.Resampler(labels, clusters, 200, 1)
    correct = np.array([1, 0, 1, 0, 1, 1.0])
    assert sampler.score(correct) == pytest.approx((0.5 + 1 + 0 + 1) / 4)
    assert probes.contrast(sampler, correct, correct) == dict(delta_pp=0.0, interval_95_pp=[0.0, 0.0])


def test_manifest_keeps_one_row_per_utterance_in_the_training_order(tmp_path):
    rows = [dict(id=f'u#r{k}', utterance='u', dialogue='d', rotation=k, label='B', audios=['a.wav'],
                 messages=[dict(role='user', content='q'), dict(role='assistant', content='BADC'[k])]) for k in range(4)]
    d2.write_jsonl(tmp_path / 'eval.jsonl', rows)
    assert probes.manifest(tmp_path / 'eval.jsonl', 'evaluation', tmp_path / 'out.jsonl') == 1
    assert d2.read_jsonl(tmp_path / 'out.jsonl')[0] == dict(id='u', audios=['a.wav'], messages=rows[0]['messages'], label='B', cluster='d')
    d2.write_jsonl(tmp_path / 'train.jsonl', [dict(id='Ses01F_impro01_F000', audios=['a.wav'], messages=rows[0]['messages'])])
    probes.manifest(tmp_path / 'train.jsonl', 'training', tmp_path / 'train-out.jsonl')
    assert d2.read_jsonl(tmp_path / 'train-out.jsonl')[0]['cluster'] == 'Ses01F_impro01'


def test_hooks_return_the_states_the_model_itself_reports(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[2] / 'ms_swift_librispeech' / 'tests'))
    from test_adapter_pair import tiny_omni
    from test_conditional_gate import inputs

    torch.manual_seed(0)
    tiny = tiny_omni().eval()
    ids = [[0, 1, 5, 5, 6, 3, 8, 2], [1, 6, 5, 5, 5, 9, 2, 4]]
    kwargs = inputs(ids)
    store, handles = probes.capture(tiny.thinker.model.layers)
    with torch.no_grad():
        reference = tiny(**kwargs, output_hidden_states=True).hidden_states
    for handle in handles:
        handle.remove()
    audio = kwargs['input_ids'].eq(5)
    both = probes.pooled(store, 5, audio, 16)
    assert both.shape == (2, 6, 2, 16) and len(reference) == 6
    for layer in range(5):                                           # the last reported state has the final norm applied
        assert torch.allclose(both[1, layer], reference[layer][:, -1].float(), atol=1e-5)
        expected = torch.stack([reference[layer][b][audio[b]].float().mean(0) for b in range(2)])
        assert torch.allclose(both[0, layer], expected, atol=1e-5)
    with pytest.raises(ValueError, match='every decoder layer'):
        probes.pooled(store[:3], 5, audio, 16)
