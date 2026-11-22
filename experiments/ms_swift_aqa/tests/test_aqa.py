from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.ms_swift_aqa.evaluate import score, validate_row
from experiments.ms_swift_aqa.prompt import format_question
from experiments.ms_swift_aqa.run import check_disjoint, verify_score
from experiments.ms_swift_librispeech.train import select, digest


def test_mmar_can_keep_the_published_blank_option(tmp_path):
    from experiments.ms_swift_aqa.evaluate import read_rows

    choices = ['Entered', 'Did not enter', '']
    with pytest.raises(ValueError):
        format_question('Did the player enter?', choices)
    prompt = format_question('Did the player enter?', choices, allow_empty_options=True)
    assert 'B. Did not enter\nC. \n' in prompt
    record = dict(id='one', question='Did the player enter?', choices=choices, audios=['example.wav'],
                  messages=[dict(role='user', content=prompt), dict(role='assistant', content='A')])
    manifest = tmp_path/'examples.jsonl'
    manifest.write_text(json.dumps(record)+'\n')
    assert read_rows(manifest, allow_empty_options=True) == [record]
    with pytest.raises(ValueError):
        read_rows(manifest)


def row(name, choices, answer, group):
    prompt = format_question('What is audible?', choices)
    return dict(id=name, question='What is audible?', choices=choices, audios=['x.wav'], domain=group, category='events',
                messages=[dict(role='user', content=prompt), dict(role='assistant', content=answer)])


class TinyTemplate:
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(map(ord, text)),
                                convert_tokens_to_ids=lambda text: 1,
                                decode=lambda ids, **kw: ''.join(map(chr, ids)))
    processor = SimpleNamespace(audio_token='<|audio_pad|>')
    texts = []

    def set_mode(self, mode):
        assert mode == 'transformers'

    def generate_context(self):
        return nullcontext()

    def encode(self, obj):
        assert len(obj['messages']) == 1
        prompt = obj['messages'][0]['content'].replace('<audio>', '\x01')
        return dict(input_ids=list(map(ord, prompt)), feature_attention_mask=[1])

    def data_collator(self, enc):
        sequences = [torch.tensor(r['input_ids']) for r in enc]
        ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_side='left')
        return dict(input_ids=ids, attention_mask=ids.ne(0), feature_attention_mask=torch.ones(len(enc), 1))

    def generate(self, model, input_ids, prefix_allowed_tokens_fn, **kw):
        assert kw['num_beams'] == kw['max_new_tokens'] == 1
        choices = [prefix_allowed_tokens_fn(i, ids) for i, ids in enumerate(input_ids)]
        assert [len(c) for c in choices] == [2, 8]
        return torch.cat([input_ids, torch.tensor([[c[-1]] for c in choices])], dim=1)


def test_variable_choices_duplicate_gold_and_clean_manifest_validation(tmp_path):
    rows = [row('one', ['high','low'], 'A', 'sound'),row('two', ['x','y','z','w','v','u','t','x'], 'A', 'speech')]
    rows[1]['correct_letters'] = ['A','H']
    manifest = tmp_path/'test.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    model = SimpleNamespace(device='cpu')
    result = score(model, TinyTemplate(), manifest, dict(batch_size=2), tmp_path/'predictions.jsonl')
    assert result['accuracy'] == result['macro_domain_accuracy'] == .5
    assert result['by_category']['speech/events'] == dict(correct=1, total=1, accuracy=1.)
    result.update(manifest_sha256=digest(manifest), adapter_sha256='abc')
    (tmp_path/'metrics.json').write_text(json.dumps(result))
    assert verify_score(tmp_path, rows, manifest, {'adapter_sha256':'abc'}) == result
    lines=[json.loads(s) for s in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert [r['prediction'] for r in lines] == ['B','H']
    lines[1]['answer_text']='wrong'
    (tmp_path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in lines))
    with pytest.raises(ValueError, match='Inconsistent prediction'):
        verify_score(tmp_path, rows, manifest, {'adapter_sha256':'abc'})


def test_prompt_and_answer_validation():
    sample = row('one', ['high','low'], 'C', 'sound')
    with pytest.raises(ValueError, match='invalid answer'):
        validate_row(sample)
    sample['messages'][1]['content'] = 'A'
    sample['choices'].reverse()
    with pytest.raises(ValueError, match='prompt differs'):
        validate_row(sample)


def test_accuracy_selection_earliest_tie_and_early_stopping():
    scores=[dict(step=i, eval_accuracy=s) for i, s in enumerate([.62,.7,.7,.68,.69], 1)]
    assert select(scores, 3, 'accuracy') == (2, .7, True)
    assert select(scores[:-1], 3, 'accuracy') == (2, .7, False)


def test_split_source_leak_rejected():
    with pytest.raises(ValueError, match='share source recordings'):
        check_disjoint([{'source_id':'abcd'}], [{'source_id':'abcd'}])
