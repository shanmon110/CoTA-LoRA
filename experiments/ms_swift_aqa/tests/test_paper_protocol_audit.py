import json

import pytest

from experiments.ms_swift_aqa import paper_federated
from experiments.ms_swift_aqa.papers28 import run


def test_source_isolation_verifies_audio_even_if_question_and_answer_repeat():
    common = {'messages':[{'role':'user','content':'<audio> What?'},
                          {'role':'assistant','content':'A'}]}
    rows = [dict(common,id='a',source_id='video1',audios=['/data/a.wav']),
            dict(common,id='b',source_id='video2',audios=['/data/b.wav'])]
    expected = [dict(common,audios=[{'path':r['audios'][0]}]) for r in rows]
    assert paper_federated.align_sources(rows,expected) == rows
    with pytest.raises(ValueError,match='source'):
        paper_federated.align_sources(rows,[expected[0],expected[0]])


def test_metrics_must_bind_checkpoint_config_including_old_ser_format():
    seal = {'adapter_sha256':'weights','config_sha256':'correct'}
    with pytest.raises(ValueError,match='config'):
        run.check_score_config({'adapter_sha256':'weights','config_sha256':'different'}, seal)
    run.check_score_config({'identity':{'config_sha256':'experiment-options','adapter_config_sha256':'correct'}}, seal)
    with pytest.raises(ValueError,match='config'):
        run.check_score_config({'identity':{'config_sha256':'correct','adapter_config_sha256':'different'}}, seal)


def test_interrupted_first_save_keeps_prior_artifacts_and_permits_retry(tmp_path):
    out = tmp_path/'train'/'pissa'
    out.mkdir(parents=True)
    (out/'protocol.json').write_text('{}')
    partial = out/'checkpoint-32'
    partial.mkdir()
    (partial/'adapter_model.safetensors').write_bytes(b'partial previous checkpoint')
    assert run.resume_arguments(out, 'pissa') == []
    assert not out.exists()
    old, = list((out.parent/'_interrupted').glob('pissa-*'))
    assert (old/'checkpoint-32'/'adapter_model.safetensors').read_bytes() == b'partial previous checkpoint'


def test_archived_source_seal_is_required_for_legacy_missing_config(tmp_path):
    sealed = {'method':'lora','adapter_sha256':'a','config_sha256':'b', 'checkpoint':'/check/32','step':32}
    score = {'adapter_sha256':'a'}
    with pytest.raises(ValueError,match='config'):
        run.check_score_config(score,sealed)
    run.check_score_config(score,sealed, historical_selection=sealed.copy())
    with pytest.raises(ValueError,match='config'):
        run.check_score_config(score,sealed,historical_selection=dict(sealed,config_sha256='bad'))


def test_longlora_pairs_initial_factors_without_freezing_trainable_embed_norm():
    from copy import deepcopy
    from peft import LoraConfig, get_peft_model
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni
    from experiments.ms_swift_librispeech.adapter_audit import initialize_pair
    from experiments.ms_swift_aqa import paper_longlora
    base = tiny_omni()
    cfg = {'training':{'seed':42},'adapter':{'last_n':5,'target_modules':['o_proj']},
           'method_options':{'longlora-omni':{'groups':4}}}
    settings = paper_longlora.options('longlora-omni',cfg,{})
    vanilla = get_peft_model(deepcopy(base), LoraConfig(r=2,target_modules=['o_proj']))
    expected = initialize_pair(vanilla,'lora',42,5)['ab_init_sha256']
    model = get_peft_model(base,LoraConfig(r=2,target_modules=['o_proj'],**settings))
    got = paper_longlora.initialize(model,'longlora-omni',cfg)
    assert got['ab_init_sha256'] == expected
    assert model.thinker.model.embed_tokens.modules_to_save.default.weight.requires_grad
    assert model.thinker.model.layers[-1].post_attention_layernorm.modules_to_save.default.weight.requires_grad


def test_federated_dataset_matches_sources_after_native_swift_shuffle():
    common = {'messages':[{'role':'user','content':'same question'},{'role':'assistant','content':'A'}]}
    rows = [dict(common,id='video_a',source_id='a',audios=['/data/1.wav']),
            dict(common,id='video_b',source_id='b',audios=['/data/2.wav'])]
    actual = [dict(common,audios=[{'path':'/data/2.wav'}]),dict(common,audios=['/data/1.wav'])]
    assert [r['id'] for r in paper_federated.align_sources(rows,actual)] == ['video_b','video_a']
    with pytest.raises(ValueError,match='source'):
        paper_federated.align_sources(rows,[actual[0],actual[0]])


def test_longlora_sft_disables_default_cache_only_for_training():
    from peft import get_peft_model,LoraConfig
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni,batch
    from experiments.ms_swift_aqa.paper_longlora import attach
    b=batch()
    b.pop('use_cache')
    model = get_peft_model(tiny_omni(),LoraConfig(r=2,target_modules=['o_proj']))
    attach(model,4)
    model.train()(**b).loss.backward()
    assert model.eval()(**b).past_key_values is not None


@pytest.mark.parametrize('cut',[1,2,3])
def test_federated_resume_reconstructs_local_and_partial_aggregation(tmp_path,cut):
    import torch
    from torch import nn
    from types import SimpleNamespace
    from copy import deepcopy
    from experiments.ms_swift_aqa.paper_federated import FederatedRounds
    from peft import LoraConfig,get_peft_model
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni
    model=get_peft_model(tiny_omni(),LoraConfig(target_modules=['o_proj'],r=2))
    model.requires_grad_(False)
    for name,p in model.named_parameters():
        p.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
    base=deepcopy(model); cfg=SimpleNamespace(output_dir=str(tmp_path))
    def advance(cb, m, opt, start, stop):
        for step in range(start,stop):
            cb.on_step_begin(cfg,SimpleNamespace(global_step=step),None,m,opt)
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad: p.add_((step+1)*.01)
            cb.on_step_end(cfg,SimpleNamespace(global_step=step+1),None,m)
    control=FederatedRounds(2,2)
    advance(control,base,torch.optim.AdamW(base.parameters()),0,8)
    live=FederatedRounds(2,2)
    opt=torch.optim.AdamW(model.parameters())
    advance(live,model,opt,0,cut)
    checkpoint=tmp_path/f'checkpoint-{cut}'
    checkpoint.mkdir()
    live.on_save(cfg,SimpleNamespace(global_step=cut),None)
    resumed=FederatedRounds(2,2,resume=checkpoint)
    for name,p in model.named_parameters():
        if '.lora_' in name: p.requires_grad_(True)
    resumed.on_train_begin(cfg,SimpleNamespace(global_step=cut),None,model)
    advance(resumed,model,opt,cut,8)
    for name,p in model.named_parameters():
        if '.lora_' in name: torch.testing.assert_close(p,dict(base.named_parameters())[name],rtol=0,atol=0)
