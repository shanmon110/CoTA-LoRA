import copy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.ms_swift_aqa import paper_late


def test_bora_matches_explicit_block_update_and_reloads():
    torch.manual_seed(3)
    base = nn.Linear(8, 6).to(torch.bfloat16)
    module = paper_late.BUILDERS["bora"](base, 3, 6, {"bora_blocks": 2})
    other = paper_late.BUILDERS["bora"](copy.deepcopy(base), 3, 6, {"bora_blocks": 2})
    x = torch.randn(2, 3, 8).bfloat16()
    torch.testing.assert_close(module(x), base(x))
    with torch.no_grad():
        module.branch.up.normal_()
    branch = module.branch
    g = (branch.sigma / branch.sigma.abs().mean()).exp()
    dense = torch.cat([
        torch.cat([(branch.up[i*3:(i+1)*3] * g[i,j]) @ branch.down[:,j*4:(j+1)*4]
                   for j in range(2)], dim=1)
        for i in range(2)], dim=0)
    expected = (base(x).float() + 2 * (x.float() @ dense.T)).bfloat16()
    torch.testing.assert_close(module(x), expected)
    other.load_state_dict(module.state_dict())
    torch.testing.assert_close(other(x), module(x))
    module(x).sum().backward()
    assert branch.sigma.grad.abs().sum() > 0 and base.weight.grad is None


def test_lokra_matches_amd_sampled_training_penalty_and_scales():
    torch.manual_seed(3)
    model = paper_late.BUILDERS["lokra"](nn.Linear(6, 4), 3, 6, {})
    with torch.no_grad():
        model.branch.up.normal_()
    torch.manual_seed(99)
    random.seed(7)
    penalty = paper_late.training_penalty(model, "lokra", {"lambda_coef": 0.001})
    torch.manual_seed(99)
    terms = []
    for matrix in (model.branch.down.T, model.branch.up):
        selected = matrix[torch.randperm(matrix.shape[0])[:3]]
        terms.append(-torch.linalg.slogdet(selected @ selected.T + 1e-6 * torch.eye(3))[1])
    expected = 0.001 * sum(terms) / (3 * (math.log(3) - 1) / 2)
    torch.testing.assert_close(penalty, expected)
    penalty.backward()
    assert model.branch.down.grad.abs().sum() > 0
    assert model.branch.up.grad.abs().sum() > 0
    assert model.base_layer.weight.grad is None


def test_stable_lora_shrinks_only_until_first_norm_crossing_survives_resume():
    module = paper_late.BUILDERS["stable-lora"](nn.Linear(5, 4), 3, 6, {})
    settings = {"shrink_ratio": 0.1}

    def optimizer(layer):
        return paper_late.install_optimizer("stable-lora", layer,
                torch.optim.AdamW(layer.parameters(), lr=0), settings)

    opt = optimizer(module)
    old = module.branch.down.detach().clone()
    opt.step()
    torch.testing.assert_close(module.branch.down, 0.9 * old)
    with torch.no_grad():
        module.branch.up.fill_(100)
    opt.step()
    torch.testing.assert_close(module.branch.down, 0.9 * old)
    other = copy.deepcopy(module)
    opt2 = optimizer(other)
    opt2.load_state_dict(opt.state_dict())
    with torch.no_grad():
        other.branch.up.zero_()
    opt2.step()
    torch.testing.assert_close(other.branch.down, 0.9 * old)


def test_loft_alternates_projects_moments_and_resumes_exactly():
    torch.manual_seed(17)
    module = paper_late.BUILDERS["loft"](nn.Linear(6, 4), 3, 3, {})
    x = torch.randn(2, 6)

    def optimizer(layer):
        return paper_late.install_optimizer("loft", layer,
                torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],
                                  lr=0.002, weight_decay=0), {"eps": 1e-4, "max_grad_norm": 1.})

    def step(layer, opt):
        opt.zero_grad()
        layer(x).square().sum().backward()
        opt.step()

    opt = optimizer(module)
    a0 = module.lora_A.weight.detach().clone()
    step(module, opt)
    torch.testing.assert_close(module.lora_A.weight, a0)
    b1 = module.lora_B.weight.detach().clone()
    assert b1.abs().sum() > 0 and "row_products" in opt.state[module.lora_B.weight]
    other = copy.deepcopy(module)
    opt2 = optimizer(other)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(3):
        step(module, opt)
        step(other, opt2)
        torch.testing.assert_close(other.lora_A.weight, module.lora_A.weight, atol=0, rtol=0)
        torch.testing.assert_close(other.lora_B.weight, module.lora_B.weight, atol=0, rtol=0)
    assert not torch.equal(module.lora_A.weight, a0)


@pytest.mark.parametrize("update_a", [True,False])
def test_loft_projected_clipping_matches_upstream_dense_active_update(update_a):
    from experiments.ms_swift_aqa.paper_late_loft import projected_clip
    from experiments.ms_swift_aqa.paper_late_loft.optim_helper import compute_full_grad_from_proj, rescale_gradients
    layer = paper_late.BUILDERS["loft"](nn.Linear(15,12),4,4,{})
    params = [p for p in layer.parameters() if p.requires_grad]
    optimizer = paper_late.install_optimizer("loft", layer, torch.optim.AdamW(params,weight_decay=0.), {})
    optimizer.update_A = update_a
    for p in params:
        p.data.normal_()
        p.grad = torch.randn_like(p)
    a,b = params
    p,other = (a,b) if update_a else (b,a)
    scaled,_ = rescale_gradients(p.grad,other,update_a)
    dense = compute_full_grad_from_proj(scaled,other,update_a)
    clip_factor = (.01/(dense.norm()+1e-6)).clamp_max(1.)
    before = [p.grad.clone() for p in params]
    projected_clip(optimizer,.01)
    for p,old in zip(params,before):
        torch.testing.assert_close(p.grad,old*clip_factor)


def test_ralora_uses_gradient_entropy_and_restores_calibrated_blocks():
    from experiments.ms_swift_aqa.paper_late_ralora import block_count
    layer = paper_late.BUILDERS["ralora"](nn.Linear(12, 12), 2, 4, {})
    low_rank = torch.zeros(12, 12)
    low_rank[0, 0] = 1
    assert block_count(low_rank, 2, 3)[0] == 1
    assert block_count(torch.eye(12), 2, 3)[0] == 4
    count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    layer.configure(4)
    assert count == 2*(12+12)
    with torch.no_grad():
        layer.up.normal_()
    x = torch.randn(2, 6, 12)
    dense = torch.block_diag(*[b @ a for a,b in zip(layer.down.view(4,2,3), layer.up.view(4,3,2))])
    torch.testing.assert_close(layer(x), layer.base_layer(x) + 2*x@dense.T)
    other = paper_late.BUILDERS["ralora"](copy.deepcopy(layer.base_layer), 2, 4, {})
    # This mirrors Swift's adapter loader, which copies state tensors directly.
    assert other(x).shape == x.shape
    with torch.no_grad():
        for key,value in other.state_dict(keep_vars=True).items():
            value.copy_(layer.state_dict()[key])
    torch.testing.assert_close(other(x), layer(x))


def test_ralora_inference_loader_sees_copied_block_count():
    base = nn.Linear(8, 8)
    target = paper_late.BUILDERS["ralora"](base, 2, 4, {})
    target.configure(2)
    with torch.no_grad():
        target.up.normal_()
    x = torch.randn(1, 8)
    with torch.inference_mode():
        fresh = paper_late.BUILDERS["ralora"](copy.deepcopy(base), 2, 4, {})
        fresh(x)
        fresh.load_state_dict(target.state_dict())
        torch.testing.assert_close(fresh(x), target(x))


def test_igu_path_pruning_and_final_mask_roundtrip():
    from experiments.ms_swift_aqa.paper_late_igu import IGULinear, RankScores
    opts = {"initial_rank_multiplier": 2}
    layer = IGULinear(nn.Linear(7, 6), 2, 4, opts)
    x = torch.randn(3, 7)
    layer.path_coefficient = 0.5
    full = layer.base_layer(x)
    half = layer(x) - full
    layer.path_coefficient = 1.
    torch.testing.assert_close(half, (layer(x)-full)/8)
    scores = RankScores({"test": layer}, 0.85, 0.85)
    layer(x).sum().backward()
    scores.update()
    scores.allocate(2*(7+6+1))
    assert layer.mask.sum() == 2
    clone = IGULinear(copy.deepcopy(layer.base_layer), 2, 4, opts)
    clone.load_state_dict(layer.state_dict())
    torch.testing.assert_close(layer(x), clone(x))


def test_mixer_hard_specialist_soft_fusion_and_padding_invariance():
    from experiments.ms_swift_aqa.paper_late_mixer import MixerLinear, mixer_loss, set_batch_mask
    layer = MixerLinear(nn.Linear(8,8), 4, 4, {"num_experts":4,"router_hidden":8,"top_k":3})
    x = torch.randn(1,4,8)
    with torch.no_grad():
        for expert in layer.experts:
            expert.b.weight.normal_()
    layer.hard_domain = 2
    torch.testing.assert_close(layer(x), layer.base_layer(x)+layer.experts[2](x))
    layer.hard_domain = None
    layer.seal_experts()
    set_batch_mask(layer,torch.tensor([[1,1,0,0]]))
    actual = layer(x)
    p = layer.router(x).softmax(-1)
    expected = sum(p[...,i,None]*ex(x) for i,ex in enumerate(layer.experts))
    torch.testing.assert_close(actual, layer.base_layer(x)+expected)
    before = mixer_loss(layer, {})
    x[:,2:] += 100
    layer(x)
    after = mixer_loss(layer,{})
    torch.testing.assert_close(before,after)
    after.backward()
    assert layer.router[0].weight.grad.abs().sum()>0


def test_mixer_router_regularizer_averages_all_projections():
    from experiments.ms_swift_aqa.paper_late_mixer import MixerLinear, mixer_loss, set_batch_mask
    lay = MixerLinear(nn.Linear(8,8),4,4,{"num_experts":4,"router_hidden":8,"top_k":3})
    lay.seal_experts()
    doubled = nn.ModuleList([copy.deepcopy(lay),copy.deepcopy(lay)])
    x = torch.randn(1,4,8)
    mask = torch.ones(1,4)
    for m in [lay,*doubled]:
        set_batch_mask(m,mask)
        m(x)
    torch.testing.assert_close(mixer_loss(lay,{}),mixer_loss(doubled,{}))


def test_igu_interval_pruning_leaves_singular_values_free_to_regrow_until_final_mask():
    from experiments.ms_swift_aqa.paper_late_igu import IGULinear, RankScores
    layer = IGULinear(nn.Linear(8,8),2,2,{"initial_rank_multiplier":2})
    scores = RankScores({"p":layer},.85,.85)
    scores.allocate(2*17, final=False)
    assert layer.mask.bool().all()
    vanished = layer.diag == 0
    assert vanished.sum() == 2
    x = torch.randn(1,8)
    layer(x).sum().backward()
    assert torch.all(layer.diag.grad[vanished] != 0)
    scores.allocate(2*17, final=True)
    assert layer.mask.sum() == 2


@pytest.fixture
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_loragen_product_loss_is_factorization_invariant_and_generation_shapes(single_cpu_thread):
    from experiments.ms_swift_aqa.paper_late_loragen import product_losses, fit_generator, peft_tensors
    torch.manual_seed(3)
    # Three source tasks, two decoder layers and two GQA projection shapes.
    data = {k:torch.randn(3,2,*shape)*0.05 for k,shape in
            {"A_q":(2,8),"B_q":(8,2),"A_v":(2,8),"B_v":(4,2)}.items()}
    equivalent = {k:(value*2 if k.startswith("A") else value/2) for k,value in data.items()}
    angular,spectral = product_losses(data,equivalent)
    torch.testing.assert_close(angular,torch.tensor(0.),atol=1e-6,rtol=0)
    torch.testing.assert_close(spectral,torch.tensor(0.),atol=1e-6,rtol=0)
    descriptions = torch.randn(4,12)
    settings = {"latent_dim":8,"hidden_dim":8,"num_experts":2,"top_k":1,"ae_steps":3,
                "diffusion_steps":3,"diffusion_updates":3,"sampling_steps":2,"denoiser_layers":1,
                "ae_lr":0.001,"denoiser_lr":0.001,"batch_size":2}
    generated,diag,state = fit_generator(data,descriptions,descriptions,settings,42)
    assert diag["source_experts"] == 3
    assert diag["lae_updates"] == 3 and diag["diffusion_updates"] == 3
    assert all(torch.isfinite(tensor).all() for tensor in generated.values())
    tensors = peft_tensors(generated)
    assert len(tensors) == 8
    assert tensors['base_model.model.thinker.model.layers.1.self_attn.v_proj.lora_B.weight'].shape == (4,2)


def test_loragen_uses_published_sinusoidal_embedding_frequencies():
    from experiments.ms_swift_aqa.paper_late_gen.position import generate_original_PE
    pe = generate_original_PE(3,4)
    expected = torch.tensor([math.sin(1.),math.cos(1000**(-.25)),math.sin(1000**(-.5)),math.cos(1000**(-.75))])
    torch.testing.assert_close(pe[1],expected)


def test_loragen_publishes_one_complete_checkpoint_after_interrupted_write(tmp_path, monkeypatch):
    from peft import LoraConfig
    from experiments.ms_swift_aqa.paper_late_loragen import publish_checkpoint
    destination = tmp_path / "run"
    cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "v_proj"])
    weights = {"test.weight": torch.randn(2, 2)}
    old_save = torch.save

    def stop(*args, **kwargs):
        raise OSError("interrupted save")

    monkeypatch.setattr(torch, "save", stop)
    with pytest.raises(OSError, match="interrupted"):
        publish_checkpoint(destination, cfg, weights, {}, {})
    assert not destination.exists()
    monkeypatch.setattr(torch, "save", old_save)
    publish_checkpoint(destination, cfg, weights, {}, {})
    assert (destination/"checkpoint-generated"/"adapter_model.safetensors").is_file()
    assert (destination/"source-generator.pt").is_file()
    assert (destination/"generation.json").is_file()
    with pytest.raises(FileExistsError):
        publish_checkpoint(destination, cfg, weights, {}, {})


@pytest.mark.parametrize("tamper", ["weights", "adapter_config", "source_library", "experiment"])
def test_loragen_reuse_validates_generation_against_source_and_native_files(tmp_path, tamper):
    from peft import LoraConfig
    from experiments.ms_swift_aqa.paper_late_loragen import publish_checkpoint, validate_generation
    from experiments.ms_swift_librispeech.train import digest
    config = {"model_revision":"rev", "method_options":{"loragen":{"train_sha256":"training"}}}
    config_file = tmp_path/"config.json"
    config_file.write_text(json.dumps(config))
    library = tmp_path/"library.json"
    library.write_text(json.dumps({"train_sha256":"training"}))
    out = tmp_path/"run"
    meta = {"experiment_sha256": digest(config_file), "source_train_sha256":"training",
            "model_revision":"rev", "library_sha256":digest(library)}
    publish_checkpoint(out, LoraConfig(r=2,target_modules=["q_proj","v_proj"]), {"a":torch.ones(2)}, {}, meta)
    validate_generation(config_file, library, out)
    cp = out/"checkpoint-generated"
    files = {"weights":cp/"adapter_model.safetensors", "adapter_config":cp/"adapter_config.json",
             "source_library":library, "experiment":config_file}
    changed = files[tamper]
    changed.write_bytes(changed.read_bytes()+b" ")
    with pytest.raises(ValueError, match="provenance"):
        validate_generation(config_file, library, out)


@pytest.mark.parametrize("method", sorted(paper_late.BUILDERS))
def test_swift_paper_tuner_updates_on_omni_audio_and_reloads(method, tmp_path, single_cpu_thread):
    from experiments.ms_swift_aqa.paper_plugin import tuners_map
    from experiments.ms_swift_aqa.paper_late_mixer import set_batch_mask
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import batch, tiny_omni

    torch.manual_seed(12)
    model = tiny_omni()
    model.thinker.config.audio_token_id = 5
    clean = copy.deepcopy(model)
    overlay = Path(paper_late.__file__).parent/"papers28"/"configs"/f"{method}.json"
    settings = json.loads(overlay.read_text())["settings"]
    if method == "lora-mixer":
        settings = dict(settings, router_hidden=8)
    kwargs = SimpleNamespace(target_modules=r"thinker\.model\.layers\.[34]\.self_attn\.[qvo]_proj",
                             lora_rank=4, lora_alpha=4, model_kwargs={"paper_settings":settings})
    tuner = tuners_map["paper_" + method]
    tuner.prepare_model(kwargs, model)
    inputs = batch()
    inputs["input_features"] = torch.randn(1, 8, 8).bfloat16()
    inputs["feature_attention_mask"] = torch.ones(1, 8, dtype=torch.long)
    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=0.01, weight_decay=0.)
    if method in {"stable-lora", "loft"}:
        optimizer = paper_late.install_optimizer(method, model, optimizer, settings)
    for path in model._paper_adapter_spec["targets"]:
        module = model.get_submodule(path)
        if method == "ralora":
            module.configure(2)
        elif method == "igu-lora":
            module.mask[::2] = 0
        elif method == "lora-mixer":
            module.seal_experts()
    if method == "lora-mixer":
        set_batch_mask(model, inputs["attention_mask"])
    initial = [p.detach().clone() for p in trainables]
    model.train()
    loss = model(**inputs).loss
    if method in {"igu-lora","lora-mixer","lokra"}:
        loss = loss + paper_late.training_penalty(model, method, settings)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before, after) for before, after in zip(initial, trainables))
    tuner.save_pretrained(model, tmp_path)
    with torch.inference_mode():
        tuner.from_pretrained(clean, tmp_path)
        model.eval()
        clean.eval()
        assert not any(p.requires_grad for p in clean.parameters())
        torch.testing.assert_close(clean(**inputs).logits, model(**inputs).logits, atol=0, rtol=0)


def test_loragen_native_peft_loads_exact_generated_qv_and_preserves_other_projections(tmp_path, single_cpu_thread):
    from peft import LoraConfig
    from swift.pipelines.utils import prepare_adapter
    from torch.nn import functional as F
    from experiments.ms_swift_aqa.paper_late_loragen import peft_tensors, publish_checkpoint
    from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni
    from experiments.ms_swift_librispeech.train import digest, model_options

    torch.manual_seed(42)
    model = tiny_omni()
    layers = model.thinker.model.layers
    model.config.save_pretrained(tmp_path)
    snapshot = tmp_path/"snapshot.json"
    snapshot.write_text(json.dumps({"revision":"test", "files":{"config.json":digest(tmp_path/"config.json")}}))
    config = {"model":str(tmp_path), "model_revision":"test", "snapshot_manifest":str(snapshot),
              "adapter":{"last_n":len(layers),"rank":2,"alpha":2,"target_modules":["q_proj","v_proj"]}}
    options = model_options(config,"lora")
    generated = {name:0.1 * torch.randn(1, len(layers), *shape) for name, shape in
                 {"A_q":(2,16), "B_q":(16,2), "A_v":(2,16), "B_v":(8,2)}.items()}
    cfg = LoraConfig(r=2, lora_alpha=2, task_type="CAUSAL_LM",
                     target_modules=options["target_regex"])
    folder = tmp_path/"run"
    publish_checkpoint(folder, cfg, peft_tensors(generated), {}, {})
    args = SimpleNamespace(tuner_type=options["tuner_type"],tuner_backend=options["tuner_backend"],
                           adapters=[str(folder/"checkpoint-generated")])
    wrapped = prepare_adapter(args, model)
    x = torch.randn(1, 3, 16).bfloat16()
    for i, layer in enumerate(wrapped.get_base_model().thinker.model.layers):
        assert isinstance(layer.self_attn.k_proj, nn.Linear) and isinstance(layer.self_attn.o_proj, nn.Linear)
        for p in ("q","v"):
            proj = getattr(layer.self_attn, p + "_proj")
            a, b = generated["A_"+p][0,i], generated["B_"+p][0,i]
            torch.testing.assert_close(proj.lora_A.default.weight, a, rtol=0, atol=0)
            torch.testing.assert_close(proj.lora_B.default.weight, b, rtol=0, atol=0)
            expected = (proj.base_layer(x).float() + F.linear(F.linear(x.float(),a),b)).bfloat16()
            torch.testing.assert_close(proj(x), expected, rtol=0, atol=0)

