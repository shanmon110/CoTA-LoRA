import copy

import pytest
import torch
from peft import PeftModel
from transformers import Qwen2_5OmniForConditionalGeneration

from experiments.ms_swift_aqa import paper_early_loram as arm
from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni, batch


def test_random_structured_pruning_and_zero_filled_recovery(tmp_path):
    full = tiny_omni()
    cfg = full.config
    cfg.thinker_config.text_config.num_key_value_heads = 2
    full = Qwen2_5OmniForConditionalGeneration(cfg).to(torch.bfloat16)
    full.forward = full.thinker.forward
    clean = copy.deepcopy(full)
    resume_base = copy.deepcopy(full)
    recipe = {'adapter': {'rank':2,'alpha':4,'last_n':5,
                          'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
              'training':{'seed':42},'method_options':{'loram-train-small':{'prune_fraction':0.5,'first_intact':1,
                     'last_intact':1,'mask_seed':42,'alignment':'none'}}}
    from types import SimpleNamespace
    kwargs = dict(tuner_type='lora',tuner_backend='peft',target_regex=r'thinker\.model\.layers\.\d+\.(self_attn|mlp)\.\w+_proj',
                  lora_rank=2,lora_alpha=4)
    opts = arm.options('loram-train-small', recipe, kwargs)
    from swift.tuner_plugin import tuners_map
    tuner = tuners_map[opts['tuner_type']]
    pruned = tuner.prepare_model(SimpleNamespace(**opts),full)
    arm.initialize(pruned,'loram-train-small',recipe)
    middle = pruned.thinker.model.layers[2]
    assert middle.self_attn.q_proj.out_features == 8 and middle.mlp.up_proj.out_features == 16
    assert pruned.thinker.model.layers[0].self_attn.q_proj.out_features == 16
    pruned(**batch()).loss.backward()
    with torch.no_grad():
        for n,p in pruned.named_parameters():
            if '.lora_B.' in n: p.add_(torch.randn_like(p)*.1)
    pruned.eval()
    original_param_ids = {n:id(p) for n,p in pruned.named_parameters()}
    with arm.evaluation_model(pruned,'loram-train-small') as expanded:
        assert expanded.thinker.model.layers[2].self_attn.num_heads == 2
        adapter = expanded.thinker.model.layers[2].self_attn.q_proj
        assert (adapter.lora_B.default.weight == 0).all(dim=1).sum() == 8
        with torch.no_grad(): dev_logits = expanded(**batch()).logits
    assert original_param_ids == {n:id(p) for n,p in pruned.named_parameters()}
    arm.export(pruned,tmp_path)
    restored=PeftModel.from_pretrained(clean, tmp_path/'recovered').eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits,dev_logits,rtol=0,atol=0)
    saved = tmp_path/'adapter_model.safetensors'
    pruned.save_pretrained(tmp_path)
    resumed = tuner.from_pretrained(resume_base, tmp_path, is_trainable=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(resumed(**batch()).logits, pruned(**batch()).logits, rtol=0, atol=0)
    assert saved.exists()
