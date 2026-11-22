"""Exercise Swift's native tuners on a CPU-only miniature Omni with real audio features."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from swift.arguments.tuner_args import TunerArguments
from swift.model import MODEL_MAPPING
from swift.model.utils import use_submodel_func
from swift.pipelines.train.tuner import prepare_adapter
from swift.template.templates.qwen import Qwen2_5OmniTemplate
from swift.tuner_plugin import tuners_map
from swift.utils import import_external_file
from transformers import Qwen2VLImageProcessor

from experiments.ms_swift_aqa.structural_suite import audit, initialize, options
from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni


TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
CONFIG = {
    'method_options': {
        'adapter': {'adapter_length': 8, 'adapter_act': 'gelu'},
        'llamapro': {'llamapro_num_new_blocks': 1, 'llamapro_num_groups': 1},
        'lora_llm': {}, 'longlora': {},
        'reft': {'reft_rank': 2, 'reft_layers': list(range(5)), 'reft_layer_key': 'thinker.model.layers',
                 'reft_intervention_type': 'LoreftIntervention'},
    },
    'adapter': {'target_modules': TARGETS, 'last_n': 5}, 'training': {'seed': 42},
}


class ProbeArgs(TunerArguments):
    model_meta = MODEL_MAPPING['qwen2_5_omni']
    model_type = 'qwen2_5_omni'
    task_type = 'causal_lm'
    tuner_type = 'lora'
    tuner_backend = 'peft'
    use_swift_lora = False


def model_and_args(method):
    torch.manual_seed(125)
    model = tiny_omni()
    cfg = model.config.thinker_config
    cfg.audio_start_token_id, cfg.audio_end_token_id, cfg.vision_start_token_id = 4, 6, 7
    cfg.bos_token_id, cfg.eos_token_id = 1, 2
    cfg.audio_config.n_window = 8
    use_submodel_func(model, 'thinker')
    model.model_meta = ProbeArgs.model_meta
    model.model_info = SimpleNamespace(is_moe_model=False, task_type='causal_lm')
    base = dict(model_type=ProbeArgs.model_type, tuner_type='lora', target_regex=r'thinker\.model\.layers\.\d+\.self_attn\.o_proj', lora_rank=2, lora_alpha=4)
    # Bypass the experiment's early compatibility error to exercise Swift's actual LongLoRA guard.
    settings = base if method == 'longlora' else options(method, CONFIG, base)
    for plugin in settings.pop('external_plugins', []):
        import_external_file(plugin)
    args = ProbeArgs(**{key: val for key, val in settings.items() if key in TunerArguments.__dataclass_fields__})
    args.tuner_type = method
    return model, args


def example():
    ids = torch.tensor([[1, 4, 5, 5, 6, 8, 13, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), input_features=torch.randn(1, 8, 8),
                feature_attention_mask=torch.ones(1, 8, dtype=torch.long))


@pytest.mark.parametrize('method', ['adapter', 'llamapro', 'lora_llm', 'reft'])
def test_multimodal_update_and_clean_reload(tmp_path, method):
    if method == 'reft':
        pytest.importorskip('pyreft')
    model, args = model_and_args(method)
    clean, _ = model_and_args(method)
    independent, _ = model_and_args(method) if method == 'reft' else (None, None)
    tuner = tuners_map[method]
    model = tuner.prepare_model(args, model)
    initial = initialize(model, method, CONFIG)
    batch = example()
    labels = torch.tensor([[-100] * 6 + [13, 2]])
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01, weight_decay=0.)
    model.train()
    # Exercise Swift's real template hook, including the zero-weight dummy image and audio tower.
    template = object.__new__(Qwen2_5OmniTemplate)
    template.mode = 'train'
    template.processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor())
    hook = model.register_forward_pre_hook(template.pre_forward_hook, with_kwargs=True)
    positions, _ = model.thinker.get_rope_index(batch['input_ids'], attention_mask=batch['attention_mask'],
                                               audio_seqlens=batch['feature_attention_mask'].sum(-1))
    for _ in range(2):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(**batch, position_ids=positions, labels=labels, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert initial['param_sha256'] != audit(model, method, CONFIG)['param_sha256']
    if method == 'lora_llm':
        assert len(initial['shared_ab_sha256']) == 64
        components = audit(model, method, CONFIG)['component_sha256']
        for key in ('audio_tower', 'visual', 'language_lora'):
            assert (components[key] == initial['component_sha256'][key]) == (key == 'visual')
    hook.remove()
    tuner.save_pretrained(model, str(tmp_path))
    loaded = tuner.from_pretrained(clean, str(tmp_path), is_trainable=True)
    assert initial['trainable_parameters'] == audit(loaded, method, CONFIG)['trainable_parameters']
    assert audit(loaded, method, CONFIG)['param_sha256'] == audit(model, method, CONFIG)['param_sha256']
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        reference = model.eval()(**batch).logits
        torch.testing.assert_close(reference, loaded.eval()(**batch).logits, atol=0, rtol=0)
        config = dict(max_new_tokens=1, do_sample=False, use_model_defaults=False,
                      prefix_allowed_tokens_fn=lambda _i, _ids: [9, 10], output_scores=True, return_dict_in_generate=True)
        prompt = dict(batch, input_ids=batch['input_ids'][:, :6], attention_mask=batch['attention_mask'][:, :6])
        prediction = loaded.generate(**prompt, **config)
        assert prediction.sequences[0, -1].item() in [9, 10]
        assert prediction.sequences.tolist() == model.generate(**prompt, **config).sequences.tolist()
        expected = loaded(**prompt).logits[0, -1].float()
        torch.testing.assert_close(prediction.scores[0][0, [9, 10]], expected[[9, 10]], atol=0, rtol=0)
        if method == 'reft':
            base = independent.eval()(**prompt).logits[0, -1].float()
            assert not torch.equal(base, expected)
            assert not torch.equal(base[[9, 10]], prediction.scores[0][0, [9, 10]])
    model.thinker.model.layers[0].input_layernorm.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='trainable scope'):
        audit(model, method, CONFIG)


def test_householder_sign_collapse_is_detected():
    pytest.importorskip('pyreft')
    model, args = model_and_args('reft')
    model = tuners_map['reft'].prepare_model(args, model)
    intervention = next(iter(model.base_model.interventions.values()))
    coords = intervention.rotate_layer.parametrizations.weight.original
    coords.grad = torch.zeros_like(coords)
    torch.optim.AdamW([coords], lr=.01, weight_decay=.01).step()
    assert torch.count_nonzero(intervention.rotate_layer.weight) == 0
    with pytest.raises(ValueError, match='orthogonal rotation signs'):
        audit(model, 'reft', CONFIG)


def test_native_longlora_model_guard():
    with pytest.raises(ValueError, match='LongLoRAModelType.LLAMA'):
        options('longlora', CONFIG, {'model_type': 'qwen2_5_omni'})
    model, args = model_and_args('longlora')
    with pytest.raises(AssertionError) as error:
        prepare_adapter(args, model)
    frame = error.traceback[-1]
    assert Path(frame.path).name == 'tuner.py'
    assert 'LongLoRAModelType.LLAMA in args.model_type' in str(frame.statement)
