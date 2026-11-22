"""Tiny CPU checks through the same Swift adapter factories used by the AVQA runner."""
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip('peft')
pytest.importorskip('swift')
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
CONFIG = {
    'adapter': {'last_n': 1, 'rank': 2, 'target_modules': PROJECTIONS},
    'method_options': {
        'ia3': {},
        'adalora': {'adalora_init_r': 4, 'adalora_target_r': 2, 'adalora_tinit': 1, 'adalora_tfinal': 1},
        'bone': {'reft_rank': 2, 'init_weights': 'true'},
        'vera': {'vera_rank': 3},
        'fourierft': {'fourier_n_frequency': 7, 'fourier_scaling': 30.},
        'boft': {'boft_block_size': 2, 'boft_block_num': 0, 'boft_n_butterfly_factor': 2},
    },
}
BASE = {'model': '/tiny', 'model_type': 'qwen2_5_omni', 'tuner_backend': 'peft', 'tuner_type': 'lora',
        'lora_alpha': 4, 'lora_rank': 2, 'lora_dropout': 0., 'lora_dtype': 'float32', 'use_dora': False,
        'target_regex': r'^thinker\.model\.layers\.1\.(self_attn|mlp)\.[a-z]+_proj$'}


class TinyOmni(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.thinker = nn.Module()
        self.thinker.audio_tower = nn.ModuleDict({'q_proj': nn.Linear(8, 8)})
        self.thinker.model = nn.Module()
        self.thinker.model.layers = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            for name, mappings in [('self_attn', [(8, 8), (8, 4), (8, 4), (8, 8)]),
                                   ('mlp', [(8, 16), (8, 16), (16, 8)])]:
                parent = nn.Module()
                setattr(block, name, parent)
                for target, (din, dout) in zip(PROJECTIONS[:4] if name == 'self_attn' else PROJECTIONS[4:],
                                                mappings, strict=True):
                    setattr(parent, target, nn.Linear(din, dout))
            self.thinker.model.layers.append(block)

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs

    def forward(self, inputs_embeds, labels=None, **kwargs):
        x = inputs_embeds
        for block in self.thinker.model.layers:
            a, m = block.self_attn, block.mlp
            x = x + a.o_proj(a.q_proj(x) + torch.cat([a.k_proj(x), a.v_proj(x)], dim=-1))
            x = x + m.down_proj(m.gate_proj(x).tanh() * m.up_proj(x))
        return CausalLMOutput(loss=x.float().square().mean() if labels is not None else None, logits=x)


def attached(method, base):
    from swift.arguments.tuner_args import TunerArguments
    from swift.pipelines.train.tuner import prepare_adapter
    from experiments.ms_swift_aqa import peft_suite

    class TinyArgs(TunerArguments):
        model_meta = SimpleNamespace(model_arch=None, is_multimodal=True)
        tuner_type = method
        task_type = 'causal_lm'
        training_args = SimpleNamespace(max_steps=6)

    opts = peft_suite.options(method, CONFIG, dict(BASE))
    assert opts['tuner_type'] == method and opts['model'] == BASE['model']
    assert opts['target_regex'] == BASE['target_regex']
    args = TinyArgs(**{f.name: opts[f.name] for f in fields(TinyArgs) if f.name in opts})
    if method == 'ia3':
        from swift.tuner_plugin import tuners_map
        return tuners_map[method].prepare_model(args, base)
    return prepare_adapter(args, base)


@pytest.mark.parametrize(('method', 'count'), [('bone', 128), ('vera', 85), ('boft', 320),
                                               ('fourierft', 49), ('adalora', 540), ('ia3', 56)])
def test_swift_native_coverage_and_clean_reload(method, count, tmp_path):
    from swift.tuners import Swift
    from experiments.ms_swift_aqa import peft_suite
    torch.manual_seed(44)
    clean = TinyOmni().to(torch.bfloat16)
    model = attached(method, deepcopy(clean))
    initial = peft_suite.audit(model, method, CONFIG)
    assert initial['trainable_parameters'] == count
    x = torch.randn(2, 4, 8).to(torch.bfloat16)
    if method == 'ia3':
        torch.testing.assert_close(model(inputs_embeds=x).logits, clean(inputs_embeds=x).logits, rtol=0, atol=0)
    loss = model(inputs_embeds=x, labels=x).loss
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.02).step()
    assert peft_suite.audit(model, method, CONFIG)['param_sha256'] != initial['param_sha256']
    if method == 'vera':
        # Enforce reading the checkpoint buffers, even when a fresh seed happens to regenerate them.
        model.base_model.vera_A['default'].add_(0.2)
    model.eval()
    before = model(inputs_embeds=x).logits.detach()
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    reloaded = Swift.from_pretrained(deepcopy(clean), str(tmp_path)).eval()
    if method == 'vera':
        torch.testing.assert_close(reloaded.base_model.vera_A['default'], model.base_model.vera_A['default'], rtol=0, atol=0)
    torch.testing.assert_close(reloaded(inputs_embeds=x).logits, before, rtol=0, atol=0)


def test_audit_rejects_missing_projection_and_trainable_tower():
    from experiments.ms_swift_aqa import peft_suite
    model = attached('bone', TinyOmni())
    model.thinker.audio_tower.q_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='trainable'):
        peft_suite.audit(model, 'bone', CONFIG)
    model.thinker.audio_tower.q_proj.weight.requires_grad_(False)
    attn = model.thinker.model.layers[-1].self_attn
    attn.v_proj = attn.v_proj.get_base_layer()
    with pytest.raises(ValueError, match='v_proj'):
        peft_suite.audit(model, 'bone', CONFIG)


def test_heterogeneous_vera_targets_match_native_peft_and_gradients():
    from peft import VeraConfig, get_peft_model
    torch.manual_seed(33)
    clean = TinyOmni()
    reference = get_peft_model(deepcopy(clean), VeraConfig(r=3, target_modules=BASE['target_regex']))
    candidate = attached('vera', deepcopy(clean))
    with torch.no_grad():
        for name, p in reference.named_parameters():
            if p.requires_grad:
                p.uniform_(0.1, 0.9)
    candidate.load_state_dict(reference.state_dict())
    x = torch.randn(2, 3, 8)
    grad = torch.randn_like(x)
    results = []
    for model in (reference, candidate):
        inputs = x.detach().clone().requires_grad_()
        output = model(inputs_embeds=inputs).logits
        factors = [p for p in model.parameters() if p.requires_grad]
        results.append([output, *torch.autograd.grad(output, [inputs, *factors], grad)])
    for want, got in zip(*results, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_options_prevent_changes_to_shared_protocol():
    from experiments.ms_swift_aqa import peft_suite
    altered = deepcopy(CONFIG)
    altered['method_options']['vera']['target_regex'] = '.*'
    with pytest.raises(ValueError, match='target_regex'):
        peft_suite.options('vera', altered, BASE)


def test_factorized_boft_matches_native_dense_outputs_and_gradients(monkeypatch):
    from peft.tuners.boft import layer as native
    from peft.tuners.boft import model as factory
    from experiments.ms_swift_aqa import peft_suite
    # The native operator is just a C++/CUDA implementation of batched block_diag.
    # Give the native dense reference the equivalent small CPU operation.
    monkeypatch.setattr(native, 'get_fbd_cuda', lambda: True)
    monkeypatch.setattr(native.FastBlockDiag, 'apply', lambda r: torch.stack([torch.block_diag(*v.unbind()) for v in r]))
    torch.manual_seed(17)
    linear = nn.Linear(16, 7)
    args = dict(boft_block_size=4, boft_block_num=0, boft_n_butterfly_factor=3, boft_dropout=0.)
    reference = native.Linear(deepcopy(linear), 'default', **args)
    peft_suite.install('boft')
    actual = factory.Linear(deepcopy(linear), 'default', **args)
    assert sum(b.numel() for b in actual.buffers()) <= 3 * 2 * 16
    with torch.no_grad():
        reference.boft_R['default'].normal_(std=0.15)
        reference.boft_s['default'].uniform_(0.5, 1.5)
        actual.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, 16)
    grad = torch.randn(3, 2, 7)
    results = []
    for layer in (reference, actual):
        inputs = x.detach().clone().requires_grad_()
        prediction = layer(inputs)
        results.append([prediction, *torch.autograd.grad(prediction, (inputs, layer.boft_R['default'],
                                                                       layer.boft_s['default']), grad)])
    for want, got in zip(*results, strict=True):
        torch.testing.assert_close(got, want, atol=3e-6, rtol=3e-5)


def test_sparse_fourier_matches_native_ifft2_outputs_and_gradients(monkeypatch):
    from peft.tuners.fourierft import layer as native
    from peft.tuners.fourierft import model as factory
    from experiments.ms_swift_aqa import peft_suite
    torch.manual_seed(23)
    linear = nn.Linear(11, 7)
    args = dict(n_frequency=31, scaling=30.)
    reference = native.FourierFTLinear(deepcopy(linear), 'default', **args)
    peft_suite.install('fourierft')
    actual = factory.FourierFTLinear(deepcopy(linear), 'default', **args)
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, 11)
    grad = torch.randn(3, 2, 7)
    results = []
    for layer in (reference, actual):
        if layer is actual:
            monkeypatch.setattr(torch.fft, 'ifft2', lambda *a, **kw: pytest.fail('materialized a dense Fourier update'))
            # On CUDA, index_add_ into repeated bins can change the order in
            # which floats are summed. Exercise a rectangular layout with collisions.
            def reject_atomic_add(self, dim, indices, *args, **kwargs):
                pytest.fail('used index_add_ with colliding frequency bins')
            monkeypatch.setattr(torch.Tensor, 'index_add_', reject_atomic_add)
        inputs = x.detach().clone().requires_grad_()
        prediction = layer(inputs)
        results.append([prediction, *torch.autograd.grad(prediction, (inputs, layer.fourierft_spectrum['default']), grad)])
    for want, got in zip(*results, strict=True):
        torch.testing.assert_close(got, want, atol=4e-6, rtol=3e-5)


def test_adalora_rank_allocation_clean_reload_and_identical_resume(tmp_path):
    from swift.callbacks.adalora import AdaloraCallback
    from swift.tuners import Swift
    from swift.trainers import TrainingArguments
    from transformers import Trainer
    from experiments.ms_swift_aqa import peft_suite
    torch.manual_seed(41)
    clean = TinyOmni().bfloat16()
    model = attached('adalora', deepcopy(clean))
    rows = [dict(inputs_embeds=torch.randn(3, 8).bfloat16(), labels=torch.ones(3)) for _ in range(6)]

    def train(model, folder, resume=None):
        args = TrainingArguments(output_dir=str(folder), max_steps=6, per_device_train_batch_size=1, use_cpu=True,
                                 learning_rate=0.03, lr_scheduler_type='constant', save_steps=3, report_to=[],
                                 disable_tqdm=True, save_safetensors=True, gradient_checkpointing=False,
                                 gradient_accumulation_steps=1, dataloader_num_workers=0)
        trainer = Trainer(model=model, args=args, train_dataset=rows)
        trainer.add_callback(AdaloraCallback(args, trainer))
        peft_suite.prepare_trainer(trainer, 'adalora', resume)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        return peft_suite.audit(model, 'adalora', CONFIG)

    model.eval()
    x = rows[0]['inputs_embeds'][None]
    assert model(inputs_embeds=x, labels=x).loss > model.get_base_model()(inputs_embeds=x, labels=x).loss
    final = train(model, tmp_path/'original')
    assert 0 < final['allocated_rank'] <= 14
    cp = tmp_path/'original'/'checkpoint-6'
    reloaded = Swift.from_pretrained(deepcopy(clean), str(cp)).eval()
    torch.testing.assert_close(reloaded(inputs_embeds=x).logits, model.eval()(inputs_embeds=x).logits, atol=0, rtol=0)
    mid = tmp_path/'original'/'checkpoint-3'
    resumed = Swift.from_pretrained(deepcopy(clean), str(mid), is_trainable=True)
    restarted = train(resumed, tmp_path/'resumed', mid)
    assert restarted['param_sha256'] == final['param_sha256']
    assert restarted['allocated_rank_per_module'] == final['allocated_rank_per_module']
