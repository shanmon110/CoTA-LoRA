"""Pair native PEFT and the integrated TA tuner on the same initial low-rank factors."""
import hashlib
import math

import torch


PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
MLP_PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def ordered_projections(names, *, require_o=False):
    ordered = tuple(name for name in PROJECTIONS + MLP_PROJECTIONS if name in names)
    if not ordered or len(ordered) != len(names):
        raise ValueError('Select a nonempty set of unique, supported decoder projections')
    if require_o and 'o_proj' not in ordered:
        raise ValueError('TA-LoRA requires o_proj in each selected decoder layer')
    return ordered


def band(count, last_n, first_layer=None):
    """Indices of the targeted decoder layers: the last `last_n`, or `last_n` layers starting at `first_layer`."""
    if type(last_n) is not int or not 1 <= last_n <= count:
        raise ValueError('invalid last_n')
    first = count - last_n if first_layer is None else first_layer
    if type(first) is not int or not 0 <= first <= count - last_n:
        raise ValueError('first_layer must leave room for last_n layers')
    return range(first, first + last_n)


def _factors(model, method, last_n, projections, first_layer=None):
    from peft import PeftModel

    if method in ('lora', 'rslora', 'dora') and isinstance(model, PeftModel):
        outer = model.get_base_model()
    elif method == 'ta_lora' and hasattr(model, '_ta_lora_config'):
        outer = model
    else:
        raise ValueError(f'Expected Swift native PEFT LoRA or installed TA plugin, got {method}')
    layers = outer.thinker.model.layers
    projections = ordered_projections(projections, require_o=method == 'ta_lora')
    factors = []
    for i in band(len(layers), last_n, first_layer):
        attn = layers[i].self_attn
        for name in projections:
            if method == 'ta_lora' and name == 'o_proj':
                branch = attn.ta_adapter
                transport_up = branch.transport_up
                if (transport_up is not None) != outer._ta_lora_config.get('independent_transport_up', False):
                    raise ValueError('TA contrast matrix and checkpoint options disagree')
                factors.append((i, branch.down, branch.up, branch.gate, None, transport_up))
            else:
                parent = layers[i].mlp if name in MLP_PROJECTIONS else attn
                proj = getattr(parent, name)
                if list(proj.active_adapters) != ['default'] or proj.merged or proj.disable_adapters:
                    raise ValueError('Expected a single active, unmerged PEFT default adapter')
                if bool(proj.use_dora.get('default')) != (method == 'dora'):
                    raise ValueError(f'PEFT use_dora disagrees with method {method} at layer {i} {name}')
                if method == 'rslora':
                    expected = proj.lora_alpha['default'] / math.sqrt(proj.r['default'])
                    if (not model.peft_config['default'].use_rslora
                            or not math.isclose(proj.scaling['default'], expected)):
                        raise ValueError(f'Incorrect rsLoRA scaling at layer {i} {name}')
                magnitude = proj.lora_magnitude_vector['default'].weight if method == 'dora' else None
                factors.append((i, proj.lora_A['default'].weight, proj.lora_B['default'].weight, None, magnitude, None))
    return factors


def audit_adapter(model, method, last_n=4, projections=('o_proj',), first_layer=None):
    """Reject a frozen factor, unintended trainable tower, or misplaced projection."""
    factors = _factors(model, method, last_n, projections, first_layer)
    expected = {id(t) for _, *tensors in factors for t in tensors if t is not None}
    gate_weights = []
    if method == 'ta_lora':
        has_context = model._ta_lora_config.get('transport_gate') in ('question', 'question_audio')
        layers = model.thinker.model.layers
        for i in band(len(layers), last_n, first_layer):
            branch = layers[i].self_attn.ta_adapter
            param = branch.gate_weight
            if (param is not None) != has_context:
                raise ValueError('Conditional gate weights and adapter mode disagree')
            if param is not None:
                rank = branch.down.shape[0]
                if param.shape != (rank, rank) or param.dtype != torch.float32:
                    raise ValueError('Expected a rank-by-rank FP32 conditional gate')
                expected.add(id(param))
                gate_weights.append(param)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if {id(p) for p in trainable.values()} != expected:
        raise ValueError(f'Unexpected trainable parameter set for {method}: {list(trainable)}')
    digest = hashlib.sha256()
    magnitudes = []
    contrast_matrices = []
    for _, down, up, gate, magnitude, transport_up in factors:
        if down.shape[0] != up.shape[1] or (gate is not None and gate.shape != (down.shape[0],)):
            raise ValueError('Invalid LoRA rank dimensions')
        for tensor in (down, up):
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        if any(t.dtype != torch.float32 for t in (down, up, gate) if t is not None):
            raise ValueError('Expected FP32 low-rank factors and TA gates')
        if magnitude is not None:
            if magnitude.shape != (up.shape[0],) or magnitude.dtype != torch.float32:
                raise ValueError('Expected one FP32 DoRA magnitude per output feature')
            magnitudes.append(magnitude)
        if transport_up is not None:
            if transport_up.shape != up.shape or transport_up.dtype != torch.float32:
                raise ValueError('Expected FP32 contrast matrix with same shape as the ordinary up matrix')
            contrast_matrices.append(transport_up)
    result = dict(ab_sha256=digest.hexdigest(), trainable_parameters=sum(p.numel() for p in trainable.values()),
                target_layers=sorted({i for i, *_ in factors}), target_projections=list(ordered_projections(projections)),
                trainable_names=list(trainable))
    if magnitudes:
        mag_digest = hashlib.sha256()
        for tensor in magnitudes:
            mag_digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
        result.update(magnitude_sha256=mag_digest.hexdigest(), magnitude_parameters=sum(t.numel() for t in magnitudes),
                      magnitude_dtypes=sorted({str(t.dtype) for t in magnitudes}))
    if contrast_matrices:
        contrast_digest = hashlib.sha256()
        for tensor in contrast_matrices:
            contrast_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result.update(transport_up_sha256=contrast_digest.hexdigest(),
                      transport_up_parameters=sum(t.numel() for t in contrast_matrices))
    if gate_weights:
        gate_digest = hashlib.sha256()
        for tensor in gate_weights:
            gate_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result.update(transport_gate_sha256=gate_digest.hexdigest(),
                      transport_gate_parameters=sum(t.numel() for t in gate_weights))
    return result


@torch.no_grad()
def initialize_pair(model, method, seed, last_n=4, projections=('o_proj',), first_layer=None):
    """Fresh adapters only. Draw A from PEFT's Kaiming distribution on an isolated CPU RNG."""
    before = audit_adapter(model, method, last_n, projections, first_layer)
    factors = _factors(model, method, last_n, projections, first_layer)
    if any(t.count_nonzero().item() for _, _, up, gate, _, transport_up in factors
           for t in (up, gate, transport_up) if t is not None):
        raise ValueError('Refusing to reinitialize trained adapter weights')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    for _, down, *_ in factors:
        initial = torch.empty(down.shape, device='cpu', dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=rng)
        down.copy_(initial)
    result = audit_adapter(model, method, last_n, projections, first_layer)
    result['ab_init_sha256'] = result['ab_sha256']
    if method == 'dora' and result['magnitude_sha256'] != before['magnitude_sha256']:
        raise ValueError('Shared initialization modified native DoRA magnitudes')
    if method == 'ta_lora':
        model._ta_lora_config['ab_init_sha256'] = result['ab_init_sha256']
    return result
