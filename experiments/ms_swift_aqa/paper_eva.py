"""Native PEFT EVA calibrated on Swift's masked, audio-conditioned training inputs."""
import hashlib
import time

import torch
from peft import EvaConfig, get_eva_state_dict, initialize_lora_eva_weights
from peft.tuners.lora.layer import LoraLayer
from torch.utils.data import DataLoader, Subset


def options(method, config, base_options):
    return {**base_options, 'init_weights':'eva'}


def native_layers(model):
    return {name:layer for name,layer in model.get_base_model().named_modules() if isinstance(layer, LoraLayer)}


def allocate(model, components):
    layers = native_layers(model)
    keys = {'base_model.model.'+name for name in layers}
    if keys != components.keys():
        raise ValueError('EVA calibration did not see exactly the intended projections')
    ranks = {k.removeprefix('base_model.model.'):v.shape[0] for k,v in components.items()}
    if sum(ranks.values()) != len(ranks)*model.peft_config['default'].r:
        raise ValueError('Native EVA rank budget changed')
    initialize_lora_eva_weights(model, eva_state_dict=components)
    remaining = native_layers(model)
    if set(remaining) != {k for k in ranks if ranks[k]}:
        raise ValueError('Incorrect native rank-zero removal')
    # PEFT 0.15 minimizes the target names on the already wrapped model. Removed
    # zero-rank modules must stay excluded when loading on a pristine Omni.
    model.peft_config['default'].target_modules = set(remaining)
    for layer in remaining.values():
        layer.lora_A['default'].float()
        layer.lora_B['default'].float()
    return ranks


def audit(model, method, config):
    layers = native_layers(model)
    n = len(model.get_base_model().thinker.model.layers)
    names = config['adapter']['target_modules']
    permitted = {f'thinker.model.layers.{i}.{parent}.{p}' for i in range(n-config['adapter']['last_n'],n)
                 for p in names for parent in ['mlp' if p in ('gate_proj','up_proj','down_proj') else 'self_attn']}
    if not 0 < len(layers) <= len(permitted) or set(layers)-permitted:
        raise ValueError('Unexpected EVA adapter scope')
    expected, ranks = {}, {}
    for name in sorted(permitted):
        ranks[name] = layers[name].r['default'] if name in layers else 0
        if name in layers:
            for factor in ('lora_A','lora_B'):
                key = f'base_model.model.{name}.{factor}.default.weight'
                expected[key] = getattr(layers[name],factor)['default'].weight
    trainables = {k:p for k,p in model.named_parameters() if p.requires_grad}
    if set(trainables) != set(expected) or sum(ranks.values()) != len(permitted)*config['adapter']['rank']:
        raise ValueError('Unexpected EVA trainables or rank budget')
    sha = hashlib.sha256()
    for k,p in sorted(trainables.items()):
        if p is not expected[k] or p.dtype != torch.float32:
            raise ValueError(f'Unexpected EVA tensor {k}')
        sha.update(k.encode()+b'\0'+p.detach().cpu().contiguous().numpy().tobytes())
    return dict(param_sha256=sha.hexdigest(), trainable_parameters=sum(p.numel() for p in trainables.values()),
                ranks=ranks, target_module_count=len(layers))


def initialize_training(trainer, method, config):
    settings = config['method_options'][method]
    if settings.keys() != {'samples','batch_size','rho','tau','max_forwards'}:
        raise ValueError('Unexpected EVA calibration settings')
    model = trainer.model
    template = trainer.template
    source = trainer.train_dataset
    rng = torch.Generator().manual_seed(config['training']['data_seed'])
    ids = torch.randperm(len(source),generator=rng)[:settings['samples']].tolist()
    if len(ids) != settings['samples']:
        raise ValueError('Insufficient training examples for EVA')
    loader = DataLoader(Subset(source,ids), batch_size=settings['batch_size'], collate_fn=template.data_collator)
    cfg = model.peft_config['default']
    cfg.eva_config = EvaConfig(rho=settings['rho'],tau=settings['tau'],use_label_mask=False)
    outer = model.get_base_model()
    calls = 0
    def forward(_, batch):
        nonlocal calls
        calls += 1
        if calls > settings['max_forwards']:
            raise RuntimeError('Native EVA did not reach its configured convergence criterion')
        if (batch['attention_mask'].shape != batch['input_ids'].shape
                or not batch['feature_attention_mask'].any(dim=1).all()
                or not (batch['input_ids'].eq(outer.config.thinker_config.audio_token_index) &
                        batch['attention_mask'].bool()).any(dim=1).all()):
            raise ValueError('Missing audio features or placeholders during EVA')
        # Use identical Swift audio scatter, positional IDs and padding. Collect
        # decoder projection inputs without allocating speech/vocabulary outputs.
        _, inputs = template.pre_forward_hook(outer, (), dict(batch))
        if 'inputs_embeds' not in inputs or 'position_ids' not in inputs:
            raise ValueError('Swift did not produce audio-conditioned embeddings and position IDs for EVA')
        inputs = {k:v for k,v in inputs.items() if k in ('inputs_embeds','attention_mask','position_ids')}
        return outer.thinker.model(**inputs,use_cache=False)
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    weights = get_eva_state_dict(model,loader,forward_fn=forward,gather_distributed_inputs=False,show_progress_bar=False)
    ranks = allocate(model,weights)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    info = dict(calibration_training_indices=ids, calibration_forwards=calls,
                calibration_seconds=time.perf_counter()-start,
                calibration_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                calibration_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
    # The first batch (used only to find shared inputs) also uses this training loader.
    return {**audit(model, method, config), **info}
