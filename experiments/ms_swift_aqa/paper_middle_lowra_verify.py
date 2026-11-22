"""CPU integration check on the full immutable Omni model and a sealed LowRA bundle."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
import weakref

import torch
from transformers import Qwen2_5OmniForConditionalGeneration
from swift.tuner_plugin import tuners_map
from . import paper_plugin
from .paper_middle_lowra_runtime import PackedProjection


def verify(args):
    cfg = json.loads(args.recipe.read_text())
    idx = json.loads((Path(cfg['settings']['quantized_base'])/'index.json').read_text())
    targets = sorted(idx['projections'])
    spec = dict(format_version=1, method='lowra', model_type='qwen2_5_omni',
                rank=cfg['rank'], alpha=cfg['alpha'], targets=targets, settings=cfg['settings'])
    def load():
        return Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,
            low_cpu_mem_usage=True)

    ids = torch.tensor([[33,51,41,61,75]])
    mask = torch.ones_like(ids)
    def outputs(model):
        return model.thinker(input_ids=ids, attention_mask=mask, use_cache=False).logits[:,-1,:]

    model = load()
    refs = [weakref.ref(model.get_submodule(target).weight) for target in targets]
    paper_plugin.install(model,spec)
    gc.collect()
    if any(ref() is not None for ref in refs):
        raise RuntimeError('Original dense decoder storage remains reachable')
    for path in targets:
        p = model.get_submodule(path)
        if not isinstance(p.base_layer,PackedProjection) or list(p.base_layer.parameters()):
            raise RuntimeError(f'Full weight or unexpected parameters retained at {path}')
    projections = sorted({p.rsplit('.',1)[-1] for p in targets})
    audit = paper_plugin.audit(model,'lowra',{'method_options':{'lowra':cfg['settings']},
        'adapter':{'last_n':36,'target_modules':projections}})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=.0002)
    with torch.no_grad():
        baseline = outputs(model.eval()).float()
    loss = torch.nn.functional.cross_entropy(outputs(model.train()).float(),torch.tensor([42]))
    loss.backward()
    if not all(p.grad is not None and p.grad.isfinite().all() for p in params):
        raise RuntimeError('Missing/nonfinite adapter gradients')
    opt.step()
    with torch.no_grad():
        updated = outputs(model.eval()).float()
    if torch.equal(baseline,updated) or not updated.isfinite().all():
        raise RuntimeError('Optimizer step left all logits unchanged or nonfinite')
    tuner = tuners_map['paper_lowra']
    with tempfile.TemporaryDirectory() as tmp:
        tuner.save_pretrained(model,tmp)
        del model,opt,params,loss,p
        gc.collect()
        restored = tuner.from_pretrained(load(),tmp).eval()
        with torch.no_grad():
            torch.testing.assert_close(outputs(restored).float(),updated,rtol=0,atol=0)
        checkpoint_bytes = (Path(tmp)/'adapter_model.safetensors').stat().st_size
    print(json.dumps(dict(projections=len(targets),released_dense_weights=len(refs),
                          trainable=audit['trainable_parameters'], adapter_bytes=checkpoint_bytes,
                          idx=spec['lowra_bundle_sha256'], clean_reload_logits_match='bitwise',
                          max_abs_logit_change=(baseline-updated).abs().max().item(),
                          device='cpu',backward=True,optimizer=True),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    verify(parser.parse_args())
