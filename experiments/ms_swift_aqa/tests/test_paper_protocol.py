from copy import deepcopy
from types import SimpleNamespace

from peft import get_peft_model, LoraConfig, PeftModel
import torch

from experiments.ms_swift_aqa.paper_eva import allocate
from experiments.ms_swift_aqa.paper_federated import ClientBatches, FederatedRounds, partition
from experiments.ms_swift_aqa.paper_longlora import shifted_heads, attach
from experiments.ms_swift_aqa.paper_plugin import PaperTuner, install, adapter_state
from experiments.ms_swift_librispeech.tests.test_adapter_pair import tiny_omni, batch


def test_eva_zero_rank_native_clean_restore(tmp_path):
    clean = tiny_omni()
    base = deepcopy(clean)
    model = get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=r'thinker\.model\.layers\.[01]\.self_attn\.o_proj',
                                          init_lora_weights='eva'))
    layers = [(k,m) for k,m in model.named_modules() if hasattr(m,'lora_A')]
    allocate(model,{k:torch.randn(0 if i==0 else 4,m.in_features) for i,(k,m) in enumerate(layers)})
    layer = model.get_submodule(layers[1][0])
    with torch.no_grad():
        layer.lora_B.default.weight.fill_(.2)
    model.save_pretrained(tmp_path)
    rebuilt = PeftModel.from_pretrained(clean,tmp_path).eval()
    assert not hasattr(rebuilt.get_submodule(layers[0][0]),'lora_A')
    assert rebuilt.get_submodule(layers[1][0]).lora_A.default.weight.shape[0] == 4
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(**batch()).logits,rebuilt(**batch()).logits,rtol=0,atol=0)


def test_longlora_causal_and_shifted_with_padding():
    torch.manual_seed(123)
    q,k,v = [torch.randn(2,4,13,3,requires_grad=True) for _ in range(3)]
    mask = torch.zeros(2,1,13,13)
    mask[1,:,:,:3] = -torch.inf
    value = shifted_heads(q,k,v,mask,4,0.)
    for h in range(4):
        width,offset=4, 0 if h<2 else 2
        for t in range(13):
            start = max(0, (t-offset)//width*width+offset)
            allowed = torch.arange(start,t+1)
            for b in range(2):
                chosen = allowed if b==0 else allowed[allowed>=3]
                ref = (torch.softmax(k[b,h,chosen]@q[b,h,t]/3**.5,0) @ v[b,h,chosen]
                       if chosen.numel() else torch.zeros(3))
                torch.testing.assert_close(value[b,h,t],ref)
    value.sum().backward()
    assert all(t.grad.isfinite().all() for t in (q,k,v))
    model = get_peft_model(tiny_omni(),LoraConfig(r=2,target_modules=['o_proj'])).eval()
    with torch.no_grad():
        original = model(**batch()).logits
        attach(model,4)
        assert torch.equal(original,model(**batch()).logits)
    model.train()(**batch()).loss.backward()


def test_clients_are_isolated_and_only_active_factor_averaged():
    shards = partition([{'source_id':str(i//2)} for i in range(32)],2,42)
    batches=ClientBatches(shards,2,4,42)
    seen = list(batches)
    assert sorted(i for b in seen for i in b)==list(range(32))
    assert all(set(b)<=set(shards[(i//2)%2]) for i,b in enumerate(seen))
    class Pair(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Module()
            self.a.lora_A=torch.nn.Linear(1,1,bias=False)
            self.a.lora_B=torch.nn.Linear(1,1,bias=False)
    model=Pair()
    optimizer=torch.optim.AdamW(model.parameters())
    controller=FederatedRounds(2,1)
    initial=model.a.lora_B.weight.item()
    for s,delta in enumerate([1.,3.]):
        controller.on_step_begin(None,SimpleNamespace(global_step=s),None,model,optimizer)
        assert not model.a.lora_A.weight.requires_grad and model.a.lora_B.weight.item()==initial
        with torch.no_grad(): model.a.lora_B.weight.add_(delta)
        controller.on_step_end(None,SimpleNamespace(global_step=s+1),None,model)
    torch.testing.assert_close(model.a.lora_B.weight, torch.tensor([[initial+2.]]))
    controller.on_step_begin(None,SimpleNamespace(global_step=2),None,model,optimizer)
    assert model.a.lora_A.weight.requires_grad and not model.a.lora_B.weight.requires_grad


def test_swift_paper_tuner_roundtrip_nonzero(tmp_path):
    from swift.tuner_plugin import tuners_map
    clean=tiny_omni()
    base=deepcopy(clean)
    t=tuners_map['paper_bora']
    target='thinker.model.layers.4.self_attn.o_proj'
    trained=install(base,dict(method='bora',format_version=1,model_type='qwen2_5_omni',targets=[target],
                             settings={'bora_blocks':2},rank=2,alpha=4))
    with torch.no_grad():
        for p in trained.parameters():
            if p.requires_grad: p.add_(.25)
    t.save_pretrained(trained,tmp_path)
    restored=t.from_pretrained(clean,tmp_path)
    assert not any(p.requires_grad for p in restored.parameters())
    assert all('base_layer' not in k for k in adapter_state(restored))
    assert torch.equal(trained.eval()(**batch()).logits,restored.eval()(**batch()).logits)
