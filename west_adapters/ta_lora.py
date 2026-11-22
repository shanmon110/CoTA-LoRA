"""TA-LoRA shared atoms and modality-conditioned attention transport."""
import torch
from torch import nn
from torch.nn import functional as F

from .o_lora import OutputAtoms, OutputLoRA


class TransportAtoms(OutputAtoms):
    def __init__(self, proj, rank, alpha, residual_precision):
        super().__init__(proj, rank, alpha, residual_precision)
        self.gate = nn.Parameter(self.down.new_zeros(rank))

    def coefficients(self, ordinary, contrast):
        coeff = super().coefficients(ordinary, contrast)
        return coeff if contrast is None else coeff + self.gate.float() * F.linear(contrast.float(), self.down.float())

    def attention_delta(self, ordinary, logits, values, mask, cache_position, layout):
        contrast = _transport(logits, values, mask, cache_position, layout)
        return self(ordinary, contrast)


class TransportLoRA(OutputLoRA):
    name = "ta-lora"
    mode = "ta"
    needs_audio_context = True
    atoms = TransportAtoms

    def atom_shapes(self, module, config):
        return {**super().atom_shapes(module, config), "gate": (config["rank"],)}


def _conditional_probs(logits, members):
    # Never softmax a row of all -inf: masking NaNs *afterwards* still poisons
    # backward. For an empty group any finite dummy row is OK; it is discarded.
    present = members.any(dim=-1, keepdim=True)
    safe = torch.where(present, logits.masked_fill(~members, -torch.inf), 0.0)
    return safe.softmax(dim=-1, dtype=torch.float32), present


def _transport(logits, values, mask, cache_position, layout):
    b, _, q, k = logits.shape
    audio, valid = layout
    if audio.size(0) != b:
        raise ValueError("audio_context batch mismatch; beam expansion is not supported")
    audio, valid = audio[:, :k].to(logits.device), valid[:, :k].to(logits.device)
    # The context is fixed before generate(); DynamicCache grows at the right.
    if k > audio.size(1):
        extension = (0, k - audio.size(1))
        audio, valid = F.pad(audio, extension, value=False), F.pad(valid, extension, value=True)
    pos = torch.arange(k - q, k, device=logits.device) if cache_position is None else cache_position
    if pos.ndim != 1 or pos.numel() != q:
        raise ValueError("expected one absolute cache position per query")
    pos = pos.to(logits.device)
    keys = torch.arange(k, device=logits.device)
    visible = valid[:, None, None, :] & (keys <= pos[:, None])
    if mask is not None:
        # v4.52.3 eager receives an inverted additive mask: 0 means visible.
        visible = visible & mask.eq(0)
    text_query = (valid & ~audio).index_select(1, pos)[:, None, :, None]

    # Use the same post-RoPE logits for both partitions, *before* global
    # softmax/dropout. Dividing small global attention masses is unstable.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p_a, has_a = _conditional_probs(logits, visible & audio[:, None, None, :])
        p_n, has_n = _conditional_probs(logits, visible & ~audio[:, None, None, :])
        delta_p = (p_a - p_n).masked_fill(~(text_query & has_a & has_n), 0)
        delta = torch.matmul(delta_p, values.float())
    return delta.transpose(1, 2).reshape(b, q, -1)
