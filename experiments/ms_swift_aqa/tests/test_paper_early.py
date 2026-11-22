import copy
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

from experiments.ms_swift_aqa import paper_early
from experiments.ms_swift_librispeech.adapter_audit import initialize_pair
from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni


def settings():
    return {'adapter': {'rank': 2, 'alpha': 4, 'last_n': 5, 'target_modules': ['o_proj']},
            'training': {'seed': 42}, 'method_options': {'lora-plus': {'lorap_lr_ratio': 16}}}


def test_loraplus_uses_matched_native_factors_and_actual_group_rates(tmp_path):
    from swift.optimizers.lorap import LorapOptimizerCallback
    from swift.tuners import Swift
    from swift.tuners.peft import LoraConfig as SwiftLoraConfig

    base = tiny_omni()
    vanilla = get_peft_model(copy.deepcopy(base), LoraConfig(r=2, lora_alpha=4, target_modules=['o_proj']))
    args = settings()
    opts = paper_early.options('lora-plus', args, {'tuner_type': 'lora', 'tuner_backend': 'peft', 'target_regex': 'o_proj'})
    model = Swift.prepare_model(base, SwiftLoraConfig(r=2, lora_alpha=4, target_modules=['o_proj'],
                                                      lorap_lr_ratio=opts['lorap_lr_ratio']))
    ours = paper_early.initialize(model, 'lora-plus', args)
    assert ours['param_sha256'] == initialize_pair(vanilla, 'lora', 42, 5)['ab_init_sha256']
    training = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, learning_rate=2e-4,
                                 optim='adamw_torch', report_to=[])
    opt = LorapOptimizerCallback(training, SimpleNamespace(model=model)).create_optimizer()
    rates = paper_early.check_loraplus_optimizer(model, opt, 2e-4, 16)
    assert rates == {'lora_A': 2e-4, 'lora_B': 0.0032}
    for group in opt.param_groups:
        group['lr'] = 2e-4
    with pytest.raises(ValueError, match='learning rate'):
        paper_early.check_loraplus_optimizer(model, opt, 2e-4, 16)


def test_pissa_restores_exact_bf16_residual_and_nonzero_weights(tmp_path):
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import batch
    base = tiny_omni()
    clean = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['o_proj'],
                                            init_lora_weights='pissa_niter_4')).eval()
    for name, p in model.named_parameters():
        if p.requires_grad:
            with torch.no_grad():
                p.add_(torch.rand_like(p) * 0.1)
    model.save_pretrained(tmp_path)
    sidecar = tmp_path/'paper_residual.safetensors'
    paper_early.save_residual(model, sidecar)
    restored = paper_early.restore_residual_adapter(clean, tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits, model(**batch()).logits, atol=0, rtol=0)
    sidecar.unlink()
    with pytest.raises(FileNotFoundError):
        paper_early.restore_residual_adapter(tiny_omni(), tmp_path)


def test_randlora_exact_native_update_and_seeded_basis_reload(tmp_path):
    from types import SimpleNamespace
    peft = __import__('peft')
    if not hasattr(peft, 'RandLoraConfig'):
        pytest.skip('Run separate PEFT 0.17.1 compatibility suite for RandLoRA')
    from swift.tuner_plugin import tuners_map
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import batch
    opts = {'r': 4, 'randlora_alpha': 8, 'projection_prng_key': 42, 'save_projection': False}
    cfg = settings()
    cfg['method_options']['randlora'] = opts
    pattern = r'thinker\.model\.layers\.\d+\.self_attn\.o_proj'
    params = paper_early.options('randlora', cfg, {'tuner_type': 'lora','tuner_backend':'peft','target_regex':pattern})
    tuner = tuners_map[params['tuner_type']]
    base = tiny_omni()
    copy_base = copy.deepcopy(base)
    model = tuner.prepare_model(SimpleNamespace(target_regex=pattern), base)
    initial = paper_early.audit(model, 'randlora', cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'randlora_lambda.' in name:
                p.add_(torch.randn_like(p))
    model.save_pretrained(tmp_path)
    restored = tuner.from_pretrained(copy_base, tmp_path).eval()
    assert initial['basis_sha256'] == paper_early.rand_basis_hash(restored)
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits,model.eval()(**batch()).logits,atol=0,rtol=0)


def test_ga_calibrates_only_training_batches_and_restores_exactly(tmp_path):
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import batch
    base = tiny_omni()
    clean = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['o_proj']))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if 'base_layer.weight' in name}
    examples = [batch() for _ in range(5)]
    examples[2]['labels'] = examples[2]['labels'].clone()
    examples[2]['labels'][0, 0] = -100
    trainer = SimpleNamespace(model=model, train_dataset=examples,
                              data_collator=lambda rows: {key: torch.cat([r[key] for r in rows], dim=1 if key=='position_ids' else 0) if torch.is_tensor(rows[0][key]) else rows[0][key]
                                                         for key in rows[0]},
                              _prepare_inputs=lambda batch: batch)
    from contextlib import nullcontext
    class Template:
        _handles = []
        def register_post_encode_hook(self, models):
            self._handles = [models[0].register_forward_pre_hook(lambda mod, args, kw: (args,kw), with_kwargs=True)]
        def remove_post_encode_hook(self):
            for handle in self._handles:
                handle.remove()
            self._handles = []
    trainer.template = Template()
    trainer.compute_loss_context_manager = nullcontext
    supervised_calls = []
    def swift_loss(model, inputs):
        assert trainer.template._handles
        supervised_calls.append(True)
        return model(**inputs).loss
    trainer.compute_loss = swift_loss
    cfg = settings()
    cfg['method_options']['lora-ga'] = {'calibration_batches': 2, 'calibration_batch_size': 1,
                                        'direction':'ArB2r', 'scale':'stable', 'stable_gamma':16}
    initial = paper_early.initialize_training(trainer, 'lora-ga', cfg)
    assert supervised_calls == [True, True] and not trainer.template._handles
    assert len(set(initial['calibration_indices'])) == 2
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
    assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if n in before)
    model.eval().save_pretrained(tmp_path)
    paper_early.save_residual(model, tmp_path / paper_early.RESIDUAL)
    restored = paper_early.restore_residual_adapter(clean, tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits, model(**batch()).logits, atol=0,rtol=0)


def test_randlora_resume_verifies_saved_basis_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_early, 'rand_basis_hash',lambda model:'restored')
    (tmp_path/'randlora-basis.json').write_text('{"sha256":"training-original"}')
    trainer=SimpleNamespace(model=None,add_callback=lambda cb:None)
    with pytest.raises(ValueError,match='basis'):
        paper_early.prepare_trainer(trainer,'randlora',resume=tmp_path)
    (tmp_path/'randlora-basis.json').write_text('{"sha256":"restored"}')
    paper_early.prepare_trainer(trainer,'randlora',resume=tmp_path)
