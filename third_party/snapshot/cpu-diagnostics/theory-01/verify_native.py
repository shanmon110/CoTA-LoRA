"""Reload the selected same-source adapter on the real pristine model and check every matrix."""
import argparse
import json
from pathlib import Path
import re
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration
from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity

cli=argparse.ArgumentParser()
cli.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/MMAR/runs/papers28-transfer-01/theory-01/omni-control'))
args=cli.parse_args()
root=args.root
cfg=Qwen2_5OmniConfig.from_pretrained('/root/autodl-tmp/west-preflight/model')
cfg.enable_audio_output=False
base=Qwen2_5OmniForConditionalGeneration.from_pretrained('/root/autodl-tmp/west-preflight/model',config=cfg,
             torch_dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager',local_files_only=True)
before={n:p for n,p in base.named_parameters() if n in {'thinker.model.layers.0.self_attn.q_proj.weight','thinker.model.layers.0.self_attn.k_proj.weight'}}
ckpt=root/'checkpoint-constructed'
model=PeftModel.from_pretrained(base,ckpt,is_trainable=False,local_files_only=True)
model.eval()
weights=load_file(ckpt/'adapter_model.safetensors')
meta=json.loads((root/'construction.json').read_text())
assert meta['checkpoint_identity']==checkpoint_identity(ckpt,'lora')
target_names={f'base_model.model.thinker.model.layers.{i}.self_attn.{proj}' for i in range(36) for proj in ('o_proj','q_proj')}
targets={k:v for k,v in model.named_modules() if hasattr(v,'lora_A')}
assert targets.keys()==target_names
torch.set_num_threads(2)
x=torch.randn(2,2048,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(4271))
gold=torch.nn.functional.linear
for name,layer in targets.items():
    a,b=layer.lora_A.default.weight,layer.lora_B.default.weight
    assert a.dtype==b.dtype==torch.float32
    assert a.shape==(8,2048) and b.shape==(2048,8) and layer.base_layer.weight.shape==(2048,2048)
    assert torch.equal(weights[name+'.lora_A.weight'],a) and torch.equal(weights[name+'.lora_B.weight'],b)
    assert layer.scaling['default']==2
    with torch.no_grad():
        expected=(gold(x,layer.base_layer.weight,layer.base_layer.bias)+2*gold(gold(x.float(),a),b)).to(torch.bfloat16)
        torch.testing.assert_close(layer(x), expected, rtol=0,atol=0)
assert model.base_model.model.thinker.model.layers[0].self_attn.q_proj.base_layer.weight is before['thinker.model.layers.0.self_attn.q_proj.weight']
assert model.base_model.model.thinker.model.layers[0].self_attn.k_proj.weight is before['thinker.model.layers.0.self_attn.k_proj.weight']
assert sum(a.numel() for a in weights.values())==meta['trainable_parameters']
result=dict(checked_projection_count=len(targets),shape=[2048,2048],rank=8,scale=2,dense_dtype='bfloat16',adapter_dtype='float32',
            exact_loaded_factor_count=len(weights),exact_bf16_projection_forwards=len(targets),checkpoint_identity=meta['checkpoint_identity'])
print(json.dumps(result,indent=2))
(root/'native-reload.json').write_text(json.dumps(result,indent=2)+'\n')
