import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.ms_swift_aqa.paper_middle import BUILDERS
from experiments.ms_swift_aqa.paper_middle import activation_mask, after_install


@pytest.mark.parametrize('method', ['toplora', 'gralora', 'magnitude-loram', 'aurora', 'flylora'])
def test_optimizer_and_reload(method):
    torch.manual_seed(123)
    base = nn.Linear(16, 8)
    clean = copy.deepcopy(base)
    x = torch.randn(2, 5, 16, requires_grad=True)
    settings = {'top_k': 2} if method == 'flylora' else {}
    module = BUILDERS[method](base, 4, 8, settings)
    torch.testing.assert_close(module(x), clean(x), atol=3e-6, rtol=3e-6)
    opt = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=.01)
    module(x).square().sum().backward()
    assert x.grad is not None and base.weight.grad is None
    opt.step()
    module.eval()
    torch.manual_seed(654)
    fresh = BUILDERS[method](clean, 4, 8, settings).eval()
    saved = {k: t.clone() for k, t in module.state_dict().items() if not k.startswith('base_layer.')}
    assert all(k.startswith('base_layer.') for k in fresh.load_state_dict(saved, strict=False).missing_keys)
    torch.testing.assert_close(fresh(x), module(x))
    assert not torch.equal(module(x), F.linear(x, base.weight, base.bias))


def test_gralora_rank_slices_match_dense_subblocks():
    m = BUILDERS['gralora'](nn.Linear(10, 6, bias=False), 4, 8, {'blocks': 2})
    m.up.data.normal_()
    x = torch.randn(3, 4, 10)
    rows = []
    for j in range(2):
        cols = []
        for i in range(2):
            a = m.down[i, :, j * 2:(j+1)*2]
            b = m.up[j, i * 2:(i+1)*2]
            cols.append((a @ b).T)
        rows.append(torch.cat(cols, 1))
    weight = torch.cat(rows)
    torch.testing.assert_close(m(x), m.base_layer(x) + 2*F.linear(x, weight))


def test_unilora_single_global_vector_and_isometry():
    layers = nn.ModuleList([BUILDERS['unilora'](nn.Linear(8, 6), 2, 2, {}) for _ in range(3)])
    after_install(layers, {'method': 'unilora', 'targets':['0', '1', '2'], 'settings': {'vector_length': 11}})
    anchor = layers[0]
    assert [(n, p.numel()) for n, p in layers.named_parameters() if p.requires_grad] == [('0.vector', 11)]
    bank = anchor.vector * anchor.inv_sqrt_count
    expanded = torch.cat([bank[i].reshape(-1) for m in layers for i in (m.index_a, m.index_b)])
    torch.testing.assert_close(expanded.square().sum(), anchor.vector.square().sum())
    for layer in layers:
        layer(torch.randn(2, 3, 8)).sum().backward()
    assert anchor.vector.grad.abs().sum() > 0


def test_alora_repeated_invocation_padding_missing_and_cached_suffix():
    from contextvars import ContextVar
    ids = torch.tensor([[0,5,4,7,5,4,9],[0,0,4,7,5,6,9],[5,4,7,9,2,0,0]])
    mask = activation_mask(ids, ids.ne(0), [5,4])
    assert mask.int().tolist() == [[0,0,0,0,1,1,1],[0,0,0,0,0,0,0],[1,1,1,1,1,0,0]]
    layer = BUILDERS['alora'](nn.Linear(8, 6), 2, 4, {})
    layer.up.data.normal_()
    layer.layout = ContextVar('alora_test', default=(mask,))
    x = torch.randn(3, 7, 8, requires_grad=True)
    out = layer(x)
    torch.testing.assert_close(out[~mask], layer.base_layer(x)[~mask])
    assert not torch.equal(out[mask], layer.base_layer(x)[mask])
    out.sum().backward()
    assert layer.up.grad.abs().sum() > 0
    cached = torch.randn(3, 1, 8)
    torch.testing.assert_close(layer(cached)[1], layer.base_layer(cached)[1])
    assert not torch.equal(layer(cached)[0], layer.base_layer(cached)[0])


def test_flylora_padding_does_not_change_bias():
    from contextvars import ContextVar
    layer = BUILDERS['flylora'](nn.Linear(8,6), 4, 8, {'top_k':2})
    other = copy.deepcopy(layer)
    x = torch.randn(1,5,8)
    layer.layout = ContextVar('valid', default=(torch.tensor([[1,1,1,0,0]]),))
    layer(x)
    other(x[:,:3])
    torch.testing.assert_close(other.route_bias, layer.route_bias)
    fixed = layer.route_bias.clone()
    layer.eval()(x)
    torch.testing.assert_close(layer.route_bias, fixed)


@pytest.mark.parametrize('method', ['toplora','gralora','magnitude-loram','aurora','flylora','unilora','alora'])
def test_swift_tuner_omni_audio_update_and_clean_reload(method, tmp_path):
    from experiments.ms_swift_aqa.paper_plugin import tuners_map
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni, batch

    torch.manual_seed(762)
    model = tiny_omni()
    model.thinker.config.audio_token_id = 5
    clean = copy.deepcopy(model)
    settings_file = Path(__file__).resolve().parents[1]/'papers28'/'configs'/f'{method}.json'
    settings = json.loads(settings_file.read_text())['settings']
    if method == 'alora':
        settings['invocation_tokens'] = [6]  # token preceding the target in this tiny tokenizer
    if method == 'flylora':
        settings['top_k'] = 2
    if method == 'unilora':
        settings['vector_length'] = 97
    pattern = r'thinker\.model\.layers\.[34]\.(?:self_attn\.[qkvo]_proj|mlp\.(?:up|gate|down)_proj)'
    args = SimpleNamespace(target_modules=pattern, lora_rank=4, lora_alpha=4,
                           model_kwargs={'paper_settings':settings})
    tuner = tuners_map['paper_' + method]
    tuner.prepare_model(args, model)
    inputs = batch()
    inputs['input_features'] = torch.randn(1, 8, 8).bfloat16()
    inputs['feature_attention_mask'] = torch.ones(1, 8, dtype=torch.long)
    inputs['labels'] = inputs['input_ids'].clone()
    inputs['labels'][:, :-1] = -100
    assert len(model._paper_adapter_spec['targets']) == 14
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss = model.train()(**inputs).loss
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    tuner.save_pretrained(model,tmp_path)
    with torch.inference_mode():
        tuner.from_pretrained(clean,tmp_path)
        assert not any(p.requires_grad for p in clean.parameters())
        torch.testing.assert_close(clean.eval()(**inputs).logits,model.eval()(**inputs).logits,rtol=0,atol=0)
