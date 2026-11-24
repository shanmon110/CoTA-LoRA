import json
from pathlib import Path

import pytest

from experiments.hearside_pilot import bands, configs, crosscorpus, formats
from experiments.hearside_pilot.tests.test_crosscorpus import PROTOCOL, answers, corpus
from experiments.trr_diagnostics import d2

FULL = lambda row, task: 0.8                                                                   # noqa: E731
BOUND = lambda row, task: (0.75 if row['rotation'] == 0 else 0.4) if task == 'mcq' else 0.5    # noqa: E731
NOTHING = lambda row, task: 0.5                                                                # noqa: E731


def scored(tmp_path, first4, mid4, last4=BOUND):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path, actors=40), PROTOCOL, folder)
    answers(folder, 'frozen-base', NOTHING)
    for family, rule in (('A-all', FULL), ('A-first4', first4), ('A-mid4', mid4), ('A-last4', last4)):
        for i, arm in enumerate(crosscorpus.FAMILIES[family]):
            answers(folder, arm, rule, seed=sum(map(ord, arm)) + i)
    return folder


def test_the_plan_differs_from_the_last_four_layers_only_in_where_the_layers_are(tmp_path):
    base = dict(task='ser', training=dict(seed=42, data_seed=42), adapter=dict(rank=1), methods={})
    (tmp_path / 'base.json').write_text(json.dumps(base))
    arms = configs.build(tmp_path / 'base.json', tmp_path, tmp_path / 'bands', 'layer-bands')
    assert sorted(arms) == ['A-first4', 'A-first4-s43', 'A-first4-s44', 'A-mid4', 'A-mid4-s43', 'A-mid4-s44']
    last = configs.arm_config(base, 'all', 4, 43)
    for name, first in (('A-first4-s43', 0), ('A-mid4-s43', 16)):
        cfg = json.loads(Path(arms[name]['config']).read_text())
        assert cfg['adapter'].pop('first_layer') == first and cfg == last
        assert arms[name]['first_layer'] == first and arms[name]['option_order'] == 'fixed' and arms[name]['seed'] == 43
    assert 'first_layer' not in last['adapter']


def test_shortcut_only_in_the_last_layers(tmp_path):
    reading = bands.judge(crosscorpus.summarize(scored(tmp_path, FULL, FULL), resamples=300))
    assert reading['order_binding'] == 'specific to the last layers'
    assert {a: b['reading'] for a, b in reading['bands'].items()} == {'A-first4': 'transfers', 'A-mid4': 'transfers', 'A-last4': 'format-bound'}
    assert reading['bands']['A-first4']['formats_with_gain'] == list(formats.FORMATS)


def test_shortcut_in_every_shallow_band(tmp_path):
    reading = bands.judge(crosscorpus.summarize(scored(tmp_path, BOUND, BOUND), resamples=300))
    assert reading['order_binding'] == 'in every shallow band'
    assert all(b['reading'] == 'format-bound' for b in reading['bands'].values())


def test_a_band_that_learns_nothing_and_a_mixed_picture(tmp_path):
    reading = bands.judge(crosscorpus.summarize(scored(tmp_path, NOTHING, BOUND), resamples=300))
    assert reading['bands']['A-first4']['reading'] == 'no gain' and reading['order_binding'] == 'mixed'


def test_same_counts_differing_answers_and_summary_names_both_corpora(tmp_path):
    folder = scored(tmp_path, FULL, FULL)
    assert all(v['different'] == 0 for v in bands.same(folder, 'A-all', 'A-all').values())
    assert bands.same(folder, 'A-all', 'A-all-s43')['mcq']['different'] > 0
    text = bands.summarize({'one': folder, 'two': folder}, tmp_path / 'summary')
    assert text.count('order binding: **specific to the last layers**') == 2 and 'declared before scoring' not in text
    assert d2.load(tmp_path / 'summary' / 'bands.json')['criteria'] == bands.CRITERIA


def test_assemble_refuses_predictions_scored_on_another_manifest(tmp_path):
    source = tmp_path / 'eval'
    (source / 'manifests').mkdir(parents=True)
    rows = [dict(id=f'u{i}#r{r}', utterance=f'u{i}', dialogue='d', speaker='s', session=4, scripted=False, rotation=r, label='A',
                 option_order=list('ABCD'), reference='A', audios=['x.wav'], messages=[dict(role='user', content='q'), dict(role='assistant', content='A')])
            for i in range(2) for r in range(4)]
    d2.write_jsonl(source / 'manifests' / 'orig.jsonl', rows)
    made = formats.build(source / 'manifests' / 'orig.jsonl', tmp_path / 'formats')
    run = source / 'runs' / 'frozen-base' / 'orig'
    run.mkdir(parents=True)
    d2.write_jsonl(run / 'predictions.jsonl', [dict(id=r['id'], reference='A', prediction='A') for r in rows])
    d2.write(run / 'metrics.json', dict(manifest_sha256='other', predictions_sha256=d2.digest(run / 'predictions.jsonl')))
    with pytest.raises(ValueError, match='another manifest'):
        bands.assemble([source], tmp_path / 'formats', PROTOCOL, tmp_path / 'a')
    d2.write(run / 'metrics.json', dict(manifest_sha256=d2.digest(source / 'manifests' / 'orig.jsonl'), predictions_sha256=d2.digest(run / 'predictions.jsonl')))
    info = bands.assemble([source], tmp_path / 'formats', PROTOCOL, tmp_path / 'b')
    assert info['arms'] == {} and info['utterances'] == 2 and made['manifests']['open']['rows'] == 2      # no format answers yet: left out
