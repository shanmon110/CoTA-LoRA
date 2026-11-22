"""LoRAM-Rand, unaligned variant (ICLR 2025, Appendix C/J), ported to Omni GQA.

Upstream's stage 0 slices the coupled coordinates and stage 3 scatters back
into zero-filled factors. Their published scripts encode Llama-3 70B dimensions.
This implementation applies those operations to actual Qwen module dimensions.
https://github.com/junzhang-zj/LoRAM/tree/ce64e68bbb3a24d21d1a85e883f71323879acb88/loram
"""
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch import nn
from transformers import TrainerCallback

from experiments.ms_swift_librispeech.adapter_audit import audit_adapter, initialize_pair


@dataclass
class Slice:
    base: nn.Linear
    axis: int
    keep: torch.Tensor


def _linear(weight, bias=None, *, trainable=False):
    out, size = weight.shape
    layer = nn.Linear(size, out, bias=bias is not None, device='meta', dtype=weight.dtype)
    layer.weight = nn.Parameter(weight, requires_grad=trainable)
    if bias is not None:
        layer.bias = nn.Parameter(bias, requires_grad=trainable)
    return layer


def _sample(size, fraction, rng):
    count = max(1, round(size*(1-fraction)))
    return torch.randperm(size, generator=rng)[:count].sort().values


def _expand_blocks(indices, size):
    return (indices[:, None]*size + torch.arange(size)).flatten()


def _manifest(slices):
    return {name: {'axis': item.axis, 'original_shape': list(item.base.weight.shape), 'keep': item.keep.tolist()}
            for name, item in slices.items()}


@torch.no_grad()
def _prune(base, cfg):
    layers = base.thinker.model.layers
    if len(layers) <= cfg['first_intact'] + cfg['last_intact']:
        raise ValueError('LoRAM-Rand needs at least one middle block to prune')
    rng = torch.Generator(device='cpu').manual_seed(cfg['mask_seed'])
    sliced = {}
    heads = []
    for i in range(cfg['first_intact'], len(layers)-cfg['last_intact']):
        block = layers[i]
        attn = block.self_attn
        kv_keep = _sample(attn.num_key_value_heads, cfg['prune_fraction'], rng)
        q_keep = _expand_blocks(kv_keep, attn.num_key_value_groups)
        kv_rows = _expand_blocks(kv_keep, attn.head_dim)
        q_rows = _expand_blocks(q_keep, attn.head_dim)
        mlp_rows = _sample(block.mlp.up_proj.out_features, cfg['prune_fraction'], rng)
        heads.append((attn, attn.num_heads, attn.num_key_value_heads, len(q_keep), len(kv_keep)))
        attn.num_heads, attn.num_key_value_heads = len(q_keep), len(kv_keep)
        for owner, names, axis, rows in (
            ('self_attn', ('q_proj',), 0, q_rows), ('self_attn', ('k_proj','v_proj'), 0, kv_rows),
            ('self_attn', ('o_proj',), 1, q_rows), ('mlp', ('up_proj','gate_proj'), 0, mlp_rows),
            ('mlp', ('down_proj',), 1, mlp_rows)):
            parent = getattr(block, owner)
            for name in names:
                layer = getattr(parent, name)
                if not isinstance(layer, nn.Linear):
                    raise ValueError('Prune the clean base before attaching any adapter')
                indices = rows.to(layer.weight.device)
                weight = layer.weight.detach().index_select(axis, indices)
                bias = layer.bias
                if bias is not None:
                    bias = bias.detach().index_select(0, indices) if axis == 0 else bias.detach().clone()
                setattr(parent, name, _linear(weight, bias))
                path = f'thinker.model.layers.{i}.{owner}.{name}'
                sliced[path] = Slice(layer.cpu(), axis, rows)
    return sliced, heads


def _attach_metadata(model, sliced, heads):
    model._loram_slices = sliced
    model._loram_heads = heads
    return model


def options(method, config, base_options):
    from swift.tuner_plugin import tuners_map
    from swift.tuner_plugin.base import PeftTuner
    cfg = config['method_options'][method]
    if (set(cfg) != {'prune_fraction','first_intact','last_intact','mask_seed','alignment'}
            or not 0 < cfg['prune_fraction'] < 1 or cfg['alignment'] != 'none'
            or min(cfg['first_intact'], cfg['last_intact']) < 0):
        raise ValueError('Expected unaligned LoRAM-Rand and a valid structured pruning configuration')

    class SwiftLoRAM(PeftTuner):
        @staticmethod
        def prepare_model(args, model):
            sliced, heads = _prune(model, cfg)
            lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.,
                              target_modules=args.target_regex, task_type='CAUSAL_LM')
            return _attach_metadata(get_peft_model(model, lora), sliced, heads)

        @staticmethod
        def from_pretrained(model, model_id, **kwargs):
            if not kwargs.get('is_trainable', False):
                return PeftModel.from_pretrained(model, Path(model_id)/'recovered', **kwargs)
            sliced, heads = _prune(model, cfg)
            saved = json.loads((Path(model_id)/'loram-masks.json').read_text())
            if saved != _manifest(sliced):
                raise ValueError('LoRAM resume pruning masks differ from checkpoint')
            return _attach_metadata(PeftModel.from_pretrained(model, model_id, **kwargs), sliced, heads)
    tuners_map['loram_rand'] = SwiftLoRAM
    return {**base_options,'tuner_type':'loram_rand'}


def audit(model, method, config):
    args = config['adapter']
    result = audit_adapter(model, 'lora', args['last_n'], args['target_modules'])
    manifest = _manifest(model._loram_slices)
    result.update(param_sha256=result['ab_sha256'], target_module_count=len(result['target_layers'])*len(result['target_projections']),
                  pruning_mask_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  alignment='none', variant='LoRAM-Rand',
                  pruned_decoder_parameters=sum(p.numel() for p in model.get_base_model().thinker.model.parameters()
                                                if not p.requires_grad))
    return result


def initialize(model, method, config):
    cfg = config['adapter']
    initialize_pair(model,'lora',config['training']['seed'],cfg['last_n'],cfg['target_modules'])
    return audit(model,method,config)


def _pad_factor(value, axis, size, keep):
    shape = list(value.shape)
    shape[axis] = size
    return value.new_zeros(shape).index_copy_(axis, keep.to(value.device), value)


def _full_factors(module, item):
    a = module.lora_A.default.weight.detach()
    b = module.lora_B.default.weight.detach()
    if item.axis == 1:
        a = _pad_factor(a, 1, item.base.in_features, item.keep)
    else:
        b = _pad_factor(b, 0, item.base.out_features, item.keep)
    return a, b


@contextmanager
def evaluation_model(model, method):
    """Switch just the pruned projections for full-base dev scoring, preserving optimizer identities."""
    outer = model.get_base_model()
    replacements = []
    try:
        for attn, all_q, all_kv, _, _ in model._loram_heads:
            attn.num_heads, attn.num_key_value_heads = all_q, all_kv
        for name, item in model._loram_slices.items():
            parent, attr = name.rsplit('.',1)
            container = outer.get_submodule(parent)
            current = getattr(container, attr)
            device = current.weight.device
            replacements.append((container, attr, current, item))
            source = item.base.to(device)
            if hasattr(current, 'lora_A'):
                full = copy.copy(current)
                full._modules = current._modules.copy()
                full.base_layer = source
                full.in_features, full.out_features = source.in_features, source.out_features
                a,b = _full_factors(current,item)
                full.lora_A = nn.ModuleDict({'default': _linear(a)})
                full.lora_B = nn.ModuleDict({'default': _linear(b)})
            else:
                full = source
            setattr(container,attr,full)
        yield model
    finally:
        for container, attr, current, item in reversed(replacements):
            setattr(container,attr,current)
            item.base.cpu()
        for attn, _, _, pruned_q, pruned_kv in model._loram_heads:
            attn.num_heads,attn.num_key_value_heads = pruned_q,pruned_kv


def export(model, checkpoint):
    from experiments.ms_swift_librispeech.train import save_json
    checkpoint = Path(checkpoint)
    # Scatter only the trained dimensions into the full-size A/B matrices.
    state = {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    for name,item in model._loram_slices.items():
        current = model.get_base_model().get_submodule(name)
        if not hasattr(current, 'lora_A'):
            continue
        for factor, value in zip(('lora_A','lora_B'), _full_factors(current,item), strict=True):
            key = f'base_model.model.{name}.{factor}.default.weight'
            if key not in state:
                raise ValueError(f'Cannot identify LoRAM factor {key}')
            state[key] = value.cpu()
    model.save_pretrained(checkpoint/'recovered', state_dict=state, save_embedding_layers=False)
    save_json(checkpoint/'loram-masks.json', _manifest(model._loram_slices))


def prepare_trainer(trainer, method, resume=None):
    class RecoveredCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs):
            export(model, Path(args.output_dir)/f'checkpoint-{state.global_step}')
    trainer.add_callback(RecoveredCheckpoint())


def load_best(config, method, checkpoint, opts):
    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template
    keys = ('model','model_type','use_hf','attn_impl','torch_dtype','load_args','tuner_backend')
    args = InferArguments(**{k:opts[k] for k in keys}, tuner_type='lora', adapters=[str(Path(checkpoint)/'recovered')],
                          max_length=config['training']['max_length'], load_data_args=False)
    model, template = prepare_model_template(args)
    return model.eval(), template
