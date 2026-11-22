"""Native Swift/PEFT adapters on the pilot's exact Thinker text decoder projections.

The compatibility hooks are local to the current process. Checkpoints keep their
native PEFT config, tensor names, and dtypes. Install the same method before
training and before constructing a clean model for development or test scoring.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.ms_swift_librispeech.adapter_audit import MLP_PROJECTIONS, ordered_projections


DEFAULTS = {
    'ia3': {},
    'adalora': dict(adalora_init_r=12, adalora_target_r=8, adalora_tinit=8, adalora_tfinal=16,
                    adalora_deltaT=1, adalora_beta1=0.85, adalora_beta2=0.85, adalora_orth_reg_weight=0.5),
    'vera': dict(vera_rank=256, vera_projection_prng_key=0, vera_dropout=0., vera_d_initial=0.1),
    'boft': dict(boft_block_size=4, boft_block_num=0, boft_n_butterfly_factor=2, boft_dropout=0.),
    'fourierft': dict(fourier_n_frequency=2000, fourier_scaling=300.),
    # The installed Swift BoNE factory reads reft_rank for the BoNE block size.
    'bone': dict(reft_rank=8, init_weights='true'),
}


def _settings(method, config):
    if method not in DEFAULTS:
        raise ValueError(f'Unknown native AVQA method {method!r}')
    overrides = config['method_options'][method]
    unknown = set(overrides) - DEFAULTS[method].keys()
    if unknown:
        raise ValueError(f'Unexpected {method} options: {sorted(unknown)}')
    return {**DEFAULTS[method], **overrides}


def options(method, config, base_options):
    """Extend the verified base LoRA model options without changing the pilot scope."""
    settings = _settings(method, config)
    if base_options['tuner_backend'] != 'peft' or not base_options.get('target_regex'):
        raise ValueError('Expected native PEFT model options with an explicit projection regex')
    if method == 'bone' and str(settings['init_weights']).lower() != 'true':
        raise ValueError('The AVQA BoNE arm uses zero-initialized BoNE blocks')
    install(method)
    return {**base_options, 'tuner_type': method, 'use_dora': False, **settings}


def _parameter_shapes(method, layer, settings):
    din, dout = layer.in_features, layer.out_features
    if method == 'ia3':
        return {'ia3_l': (1, din) if layer.is_feedforward else (dout, 1)}
    if method == 'adalora':
        r = settings['adalora_init_r']
        return {'lora_A': (r, din), 'lora_B': (dout, r), 'lora_E': (r, 1)}
    if method == 'vera':
        return {'vera_lambda_d': (settings['vera_rank'],), 'vera_lambda_b': (dout,)}
    if method == 'bone':
        return {'bone_block': (settings['reft_rank'], dout)}
    if method == 'fourierft':
        return {'fourierft_spectrum': (settings['fourier_n_frequency'],)}
    size = settings['boft_block_size'] or din // settings['boft_block_num']
    return {'boft_R': (settings['boft_n_butterfly_factor'], din // size, size, size),
            'boft_s': (dout, 1)}


def audit(model, method, config):
    """Check every requested adapter and every trainable tensor against the decoder layout."""
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer

    settings = _settings(method, config)
    if (not isinstance(model, PeftModel) or set(model.peft_config) != {'default'}
            or model.peft_config['default'].peft_type.value.lower() != method):
        raise ValueError(f'Expected native single-adapter PEFT {method}')
    outer = model.get_base_model()
    layers = outer.thinker.model.layers
    n = config['adapter']['last_n']
    if not 1 <= n <= len(layers):
        raise ValueError(f'Invalid target decoder layer count {n}')
    projections = ordered_projections(config['adapter']['target_modules'])
    wanted = set()
    for i in range(len(layers) - n, len(layers)):
        for projection in projections:
            parent = 'mlp' if projection in MLP_PROJECTIONS else 'self_attn'
            wanted.add(f'thinker.model.layers.{i}.{parent}.{projection}')
    adapted = {name: layer for name, layer in outer.named_modules() if isinstance(layer, BaseTunerLayer)}
    if set(adapted) != wanted:
        raise ValueError(f'{method} projection mismatch: missing={sorted(wanted - adapted.keys())}, '
                         f'extra={sorted(adapted.keys() - wanted)}')
    expected = {}
    for name, layer in adapted.items():
        if method == 'ia3' and layer.is_feedforward != ('.mlp.' in name):
            raise ValueError(f'IA3 should scale the input of MLPs and the output of attention projections at {name}')
        if (list(layer.active_adapters) != ['default'] or layer.merged or layer.disable_adapters
                or not isinstance(layer.get_base_layer(), nn.Linear)):
            raise ValueError(f'Expected one active, unmerged native linear adapter at {name}')
        for factor, shape in _parameter_shapes(method, layer, settings).items():
            slots = getattr(layer, factor, {})
            if set(slots) != {'default'}:
                raise ValueError(f'Missing or additional {factor} adapters at {name}')
            param = slots['default']
            if param.shape != shape or param.dtype != torch.float32:
                raise ValueError(f'{name}.{factor} expected FP32 {shape}, got {param.dtype} {tuple(param.shape)}')
            expected[f'base_model.model.{name}.{factor}.default'] = param
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(expected) or any(trainable[k] is not p for k, p in expected.items()):
        raise ValueError(f'{method} trainable parameters differ: missing={sorted(expected.keys() - trainable.keys())}, '
                         f'extra={sorted(trainable.keys() - expected.keys())}')
    checksum = hashlib.sha256()
    for name, p in sorted(trainable.items()):
        checksum.update(str((name, tuple(p.shape), str(p.dtype))).encode() + b'\0')
        checksum.update(p.detach().cpu().contiguous().numpy().tobytes())
    result = dict(trainable_parameters=sum(p.numel() for p in trainable.values()), param_sha256=checksum.hexdigest(),
                  target_layers=list(range(len(layers) - n, len(layers))), target_projections=list(projections),
                  target_module_count=len(adapted), trainable_names=list(sorted(trainable)))
    if method == 'adalora':
        pattern = model.peft_config['default'].rank_pattern
        if pattern is not None:
            counts = {k: sum(mask) for k, mask in pattern.items()}
            if {k.removesuffix('.default').removesuffix('.lora_E') for k in counts} != wanted:
                raise ValueError('AdaLoRA allocation does not cover all target matrices')
            result.update(allocated_rank=sum(counts.values()), allocated_rank_per_module=counts)
    return result


def _install_ia3():
    from peft import IA3Config, get_peft_model
    from swift.tuner_plugin import tuners_map
    from swift.tuner_plugin.ia3 import IA3Tuner

    class DecoderIA3(IA3Tuner):
        @staticmethod
        def prepare_model(args, model):
            # The example plugin derives this pattern from model_arch.mlp, which
            # is unset on Omni. Limit both native PEFT masks to the decoder recipe.
            if not args.target_regex:
                raise ValueError('AQA IA3 requires an explicit decoder target regex')
            cfg = IA3Config(target_modules=args.target_regex, feedforward_modules=r'.*\.mlp\.[^.]+$')
            return get_peft_model(model, cfg)

    tuners_map['ia3'] = DecoderIA3


def _install_vera():
    from peft.tuners.vera.model import VeraModel
    from swift.pipelines.train import tuner
    original_targets = tuner.get_vera_target_modules

    def exact_targets(model, config):
        # Swift's legacy shape filter iterates the characters of a regex, and
        # drops heterogeneous targets. PEFT 0.15 slices shared A/B to each shape.
        if isinstance(config.target_modules, str):
            return config
        return original_targets(model, config)

    tuner.get_vera_target_modules = exact_targets
    original_injection = VeraModel._pre_injection_hook

    def prepare_buffers(self, model, config, adapter_name):
        original_injection(self, model, config, adapter_name)

        def load_projection_buffers(module, state, prefix, *args):
            # PEFT 0.15 removes ".default" from these keys on save and its
            # loader only reinserts it for the trainable vera_lambda_ prefix.
            for key in ('vera_A', 'vera_B'):
                saved = prefix + key
                if saved in state:
                    state[f'{saved}.{adapter_name}'] = state.pop(saved)

        self.register_load_state_dict_pre_hook(load_projection_buffers)

    VeraModel._pre_injection_hook = prepare_buffers


def _install_bone():
    from peft.tuners.bone import model as native
    from peft.tuners.bone.layer import BoneLinear

    class MixedPrecisionBone(BoneLinear):
        def forward(self, x, *args, **kwargs):
            if self.disable_adapters or self.merged or self.bone_fn == 'bat':
                return super().forward(x, *args, **kwargs)
            result = self.base_layer(x, *args, **kwargs)
            for name in self.active_adapters:
                if name in self.bone_block:
                    block = self.bone_block[name]
                    rank = block.shape[0]
                    padded = F.pad(x, (0, -x.shape[-1] % rank)) if x.shape[-1] % rank else x
                    grouped = padded.unflatten(-1, (-1, rank)).sum(-2)
                    result = result + grouped.to(block.dtype) @ block
            return result.to(x.dtype)

    native.BoneLinear = MixedPrecisionBone


def _install_boft():
    from peft.tuners.boft import model as native
    from peft.tuners.boft.layer import Linear, MultiplicativeDropoutLayer
    from torch.utils.checkpoint import checkpoint

    class FactorizedBOFT(Linear):
        def update_layer(self, adapter_name, boft_block_size, boft_block_num, boft_n_butterfly_factor,
                         boft_dropout, init_weights):
            # The native implementation builds a dense (factors,din,din) permutation
            # buffer and silently forces one factor without its compiled extension.
            dim, factors = self.in_features, boft_n_butterfly_factor
            if (boft_block_size > 0) == (boft_block_num > 0) or factors < 1:
                raise ValueError('Specify one positive BOFT block dimension and positive butterfly factor count')
            size = boft_block_size or dim // boft_block_num
            blocks = boft_block_num or dim // size
            if (size * blocks != dim or blocks % (2 ** (factors - 1))
                    or (factors > 1 and size % 2)):
                raise ValueError(f'BOFT cannot partition input dimension {dim} into these butterfly blocks')
            device = self.get_base_layer().weight.device
            perms = []
            for i in range(factors):
                # Each PEFT permutation puts the even half-blocks before the odd
                # half-blocks within each group; the innermost permutation is I.
                order = torch.arange(dim, device=device)
                if i:
                    groups = order.reshape(blocks // (2 ** i), -1, size // 2)
                    order = torch.cat((groups[:, ::2], groups[:, 1::2]), dim=1).flatten()
                perms.append(order)
            perm = torch.stack(perms)
            self.register_buffer('perm_' + adapter_name, perm, persistent=False)
            self.register_buffer('inverse_perm_' + adapter_name, perm.argsort(dim=-1), persistent=False)
            self.boft_dropout[adapter_name] = MultiplicativeDropoutLayer(boft_dropout) if boft_dropout else nn.Identity()
            self.boft_block_size[adapter_name] = size
            self.boft_block_num[adapter_name] = blocks
            self.boft_R[adapter_name] = nn.Parameter(torch.empty(factors, blocks, size, size))
            self.boft_s[adapter_name] = nn.Parameter(torch.empty(self.out_features, 1))
            self.reset_boft_parameters(adapter_name, init_weights)
            self._move_adapter_to_device_of_base_layer(adapter_name)
            self.set_adapter(self.active_adapters)

        def _rotate(self, x, name, apply_dropout):
            raw = self.boft_R[name]
            factors, blocks, size, _ = raw.shape
            q = self.cayley_batch(raw.flatten(0, 1)).reshape_as(raw)
            if apply_dropout:
                q = self.boft_dropout[name](q)
            shape = x.shape
            for i in reversed(range(factors)):
                # x @ (P D P.T); PEFT multiplies successive factors on the left.
                if i:
                    x = x.index_select(-1, getattr(self, 'inverse_perm_' + name)[i])
                packed = x.reshape(-1, blocks, size).transpose(0, 1)
                x = torch.bmm(packed, q[i].to(x.dtype)).transpose(0, 1).reshape(shape)
                if i:
                    x = x.index_select(-1, getattr(self, 'perm_' + name)[i])
            return x

        def get_delta_weight(self, adapter):
            weight = self.get_base_layer().weight
            eye = torch.eye(self.in_features, device=weight.device, dtype=self.boft_R[adapter].dtype)
            return self._rotate(eye, adapter, False).to(weight.dtype), self.boft_s[adapter].to(weight.dtype)

        def forward(self, x, *args, **kwargs):
            if self.disable_adapters or self.merged:
                return super().forward(x, *args, **kwargs)
            rotated, scale = x, 1.
            for name in reversed(self.active_adapters):
                if name in self.boft_R:
                    # Save the original activation instead of one permuted
                    # activation for every butterfly factor in the backward pass.
                    if self.training and torch.is_grad_enabled():
                        dropout = not isinstance(self.boft_dropout[name], nn.Identity)
                        rotated = checkpoint(self._rotate, rotated, name, True,
                                             use_reentrant=False, preserve_rng_state=dropout)
                    else:
                        rotated = self._rotate(rotated, name, False)
                    scale = scale * self.boft_s[name].flatten()
            base = self.base_layer(rotated, *args, **kwargs)
            bias = self.base_layer.bias
            signal = base if bias is None else base - bias
            # Only scale the weight contribution, preserving identity at initialization.
            return (base + signal * (scale - 1.)).to(x.dtype)

    native.Linear = FactorizedBOFT


class _FrequencyAxis(nn.Module):
    """Group sampled PEFT coordinates for a fixed-order sum at each Fourier bin."""
    def __init__(self, indices):
        super().__init__()
        order = indices.argsort(stable=True)
        bins, counts = indices[order].unique_consecutive(return_counts=True)
        for name, tensor in [('indices', indices), ('order', order), ('bins', bins), ('counts', counts)]:
            self.register_buffer(name, tensor, persistent=False)

    def inverse(self, coefficients, size):
        # Several sampled matrix entries can have the same row or column.
        # Reduce them in order, then write each distinct bin exactly once.
        # A CUDA index_add_ on the repeated bins could sum them in different orders.
        sorted_values = torch.view_as_real(coefficients.index_select(-1, self.order)).movedim(-2, 0).contiguous()
        sums = torch.segment_reduce(sorted_values, 'sum', lengths=self.counts, unsafe=True).movedim(0, -2)
        spectrum = coefficients.new_zeros((*coefficients.shape[:-1], size))
        spectrum.index_copy_(-1, self.bins, torch.view_as_complex(sums))
        return torch.fft.ifft(spectrum, dim=-1).real


class _SparseFourier(torch.autograd.Function):
    """Apply Re(IFFT2(S)) to x with S restricted to PEFT's sampled frequencies."""
    @staticmethod
    def forward(ctx, x, spectrum, rows, cols, out_features, scaling):
        xf = torch.fft.ifft(x.to(spectrum.dtype), dim=-1).index_select(-1, cols.indices)
        ctx.save_for_backward(x, spectrum)
        ctx.rows, ctx.cols = rows, cols
        ctx.scaling = scaling
        return rows.inverse(xf * spectrum, out_features) * scaling

    @staticmethod
    def backward(ctx, grad):
        x, spectrum = ctx.saved_tensors
        rows, cols = ctx.rows, ctx.cols
        gf = torch.fft.ifft(grad.to(spectrum.dtype), dim=-1).index_select(-1, rows.indices)
        dx = ds = None
        if ctx.needs_input_grad[0]:
            # Transposition exchanges the sampled row/column frequencies.
            dx = (cols.inverse(gf * spectrum, x.shape[-1]) * ctx.scaling).to(x.dtype)
        if ctx.needs_input_grad[1]:
            xf = torch.fft.ifft(x.to(spectrum.dtype), dim=-1).index_select(-1, cols.indices)
            ds = (gf * xf).real.reshape(-1, spectrum.numel()).sum(0) * ctx.scaling
        return dx, ds, None, None, None, None


def _install_fourier():
    from peft.tuners.fourierft import model as native
    from peft.tuners.fourierft.layer import FourierFTLinear

    class SparseFourierLinear(FourierFTLinear):
        def update_layer(self, adapter_name, *args, **kwargs):
            super().update_layer(adapter_name, *args, **kwargs)
            device = self.get_base_layer().weight.device
            rows, cols = self.indices[adapter_name]
            self.add_module('rows_' + adapter_name, _FrequencyAxis(rows).to(device))
            self.add_module('cols_' + adapter_name, _FrequencyAxis(cols).to(device))

        def forward(self, x, *args, **kwargs):
            if self.disable_adapters or self.merged:
                return super().forward(x, *args, **kwargs)
            result = self.base_layer(x, *args, **kwargs)
            for name in self.active_adapters:
                if name in self.fourierft_spectrum:
                    rows, cols = getattr(self, 'rows_' + name), getattr(self, 'cols_' + name)
                    result = result + _SparseFourier.apply(x, self.fourierft_spectrum[name], rows, cols,
                                                           self.out_features, self.fourierft_scaling[name])
            return result.to(x.dtype)

    native.FourierFTLinear = SparseFourierLinear


@lru_cache(maxsize=None)
def install(method):
    """Install the process-local compatibility code before Swift constructs the adapter."""
    if method not in DEFAULTS:
        raise ValueError(f'Unknown native AVQA method {method!r}')
    if method == 'ia3':
        _install_ia3()
    elif method == 'vera':
        _install_vera()
    elif method == 'bone':
        _install_bone()
    elif method == 'boft':
        _install_boft()
    elif method == 'fourierft':
        _install_fourier()


def prepare_trainer(trainer, method, resume=None):
    """Attach any method-specific optimizer/checkpoint hooks to the Swift trainer."""
    if method != 'adalora':
        return
    from swift.callbacks.adalora import AdaloraCallback
    from transformers import TrainerCallback

    model = trainer.model
    cfg = model.peft_config['default']
    allocator = model.base_model.rankallocator
    if cfg.peft_type.value != 'ADALORA' or not 0 < cfg.target_r < cfg.init_r or cfg.deltaT < 1:
        raise ValueError('AdaLoRA requires a smaller positive target rank and a positive allocation interval')
    original_save = model.save_pretrained

    def save_full_factors(*args, **kwargs):
        # PEFT's compressed save changes rank shapes. On BF16 bases it can
        # round FP32 factors on reload and cannot restore the optimizer slots.
        # The actual native factors already include the learned zero singular
        # values. Keep their full shapes and checkpoint the allocator separately.
        pattern = cfg.rank_pattern
        try:
            cfg.rank_pattern = None
            return original_save(*args, **kwargs)
        finally:
            cfg.rank_pattern = pattern

    class Allocation(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if not (0 <= cfg.tinit and 0 <= cfg.tfinal and cfg.tinit + cfg.tfinal < state.max_steps):
                raise ValueError('AdaLoRA schedule must include an allocation phase before the final steps')
            allocator.set_total_step(state.max_steps)
            if resume is None:
                return
            saved = torch.load(Path(resume) / 'adalora-allocator.pt', weights_only=True, map_location='cpu')
            if saved['global_step'] != state.global_step or saved['max_steps'] != state.max_steps:
                raise ValueError('AdaLoRA checkpoint step differs from the restored Trainer schedule')
            params = {k: p for k, p in model.get_base_model().named_parameters() if p.requires_grad}
            expected = set(params) if state.global_step < state.max_steps - cfg.tfinal else set()
            for field in ('exp_avg_ipt', 'exp_avg_unc'):
                tensors = saved[field]
                if set(tensors) != expected or any(t.shape != params[k].shape for k, t in tensors.items()):
                    raise ValueError(f'Incomplete AdaLoRA allocator state: {field}')
                setattr(allocator, field, {k: t.to(params[k]) for k, t in tensors.items()})
            # PEFT overwrites instantaneous importance before reading it, and
            # uses key existence here to decide whether the averages are fresh.
            allocator.ipt = {k: torch.zeros_like(v) for k, v in allocator.exp_avg_ipt.items()}
            cfg.rank_pattern = saved['rank_pattern']

        def on_optimizer_step(self, args, state, control, **kwargs):
            # Transformers calls this after optimizer.step() and before zero_grad().
            # state.global_step still counts completed previous steps, also on resume.
            model.base_model.update_and_allocate(state.global_step + 1)

        def on_save(self, args, state, control, **kwargs):
            saved = {'global_step': state.global_step, 'max_steps': state.max_steps, 'rank_pattern': cfg.rank_pattern}
            for field in ('exp_avg_ipt', 'exp_avg_unc'):
                saved[field] = {k: t.detach().cpu() for k, t in getattr(allocator, field).items()}
            path = Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'adalora-allocator.pt'
            tmp = path.with_suffix('.tmp')
            torch.save(saved, tmp)
            tmp.replace(path)

    trainer.remove_callback(AdaloraCallback)
    model.save_pretrained = save_full_factors
    trainer.add_callback(Allocation())
