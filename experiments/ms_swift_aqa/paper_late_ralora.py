"""RaLoRA (base variant) from the ICLR 2026 supplemental RaLoRA.py.

The supplement SHA-256 is a40c3c99d6016e8ca37074f47f545fb6350c9b37836f2d9ba8e232ec95205bdf.
"""
import math
import time

import torch
from torch import nn


def block_count(gradient, rank, max_power):
    singular = torch.linalg.svdvals(gradient.float())
    total = singular.sum()
    if not torch.isfinite(total):
        raise ValueError("RaLoRA received a nonfinite full-weight gradient")
    if total == 0:
        return 1, 0.
    p = singular / total
    effective = (-(p * (p + 1e-10).log()).sum()).exp().item()
    exponent = max(0, min(max_power, math.floor(math.log2(max(1., effective)) - math.log2(rank))))
    return 2 ** exponent, effective


class RaLoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, settings):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        opts = dict(dtype=torch.float32, device=base.weight.device)
        # Block-structured factors have exactly the same storage cost at all n.
        self.down = nn.Parameter(torch.empty(rank * base.in_features, **opts))
        self.up = nn.Parameter(torch.zeros(rank * base.out_features, **opts))
        with torch.inference_mode(False):
            block_count = torch.ones((), dtype=torch.long, device=base.weight.device)
        self.register_buffer("block_count", block_count)
        self._block_version = -1
        self.configure(1)

    @torch.no_grad()
    def configure(self, blocks):
        ni, no = self.base_layer.in_features, self.base_layer.out_features
        if blocks < 1 or blocks & (blocks - 1) or ni % blocks or no % blocks:
            raise ValueError(f"RaLoRA requires power-of-two equal blocks dividing {ni} and {no}")
        self.block_count.fill_(blocks)
        self.down.uniform_(-1 / math.sqrt(ni // blocks), 1 / math.sqrt(ni // blocks))
        self.up.zero_()

    def forward(self, x):
        base = self.base_layer(x)
        # Moving a restored model under inference_mode can also create a new
        # inference tensor. Keep the scalar versioned for deferred Swift loads.
        if torch.is_inference(self.block_count):
            with torch.inference_mode(False):
                self.block_count = self.block_count.clone()
            self._block_version = -1
        if self._block_version != self.block_count._version:
            self._blocks = int(self.block_count.item())
            self._block_version = self.block_count._version
        n = self._blocks
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = self.down.view(n, self.rank, -1)
            b = self.up.view(n, -1, self.rank)
            z = torch.einsum("...gi,gri->...gr", x.float().unflatten(-1,(n,-1)), a)
            y = torch.einsum("...gr,gor->...go", z, b).flatten(-2)
            return (base.float() + self.scale*y).to(base.dtype)


def calibrate(trainer, settings):
    model, template = trainer.model, trainer.template
    count = settings["calibration_steps"]
    width = settings["calibration_batch_size"]
    max_power = settings["erank_max_power"]
    if not all(type(x) is int and x >= 1 for x in (width, count)) or max_power < 0:
        raise ValueError("Invalid RaLoRA calibration budget")
    layers = {name: layer for name,layer in model.named_modules() if isinstance(layer, RaLoRALinear)}
    dataset = trainer.train_dataset
    if not layers or count*width > len(dataset):
        raise ValueError("No RaLoRA targets or too few training examples for calibration")
    gen = torch.Generator().manual_seed(trainer.args.data_seed)
    indices = torch.randperm(len(dataset), generator=gen)[:count*width].tolist()
    saved_trainables = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    gradients, hooks = {}, []
    before = time.perf_counter()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def collect(name):
        def offload(p):
            g = p.grad.detach().to(device="cpu", dtype=torch.float32)
            if name in gradients:
                gradients[name].add_(g)
            else:
                gradients[name] = g
            p.grad = None
        return offload

    try:
        model.train()
        model.requires_grad_(False)
        for name, layer in layers.items():
            p = layer.base_layer.weight.requires_grad_(True)
            hooks.append(p.register_post_accumulate_grad_hook(collect(name)))
        template.register_post_encode_hook([model])
        for start in range(0, len(indices), width):
            batch = trainer.data_collator([dataset[i] for i in indices[start:start+width]])
            inputs = trainer._prepare_inputs(batch)
            with template.forward_context(model, inputs), trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(model, inputs)
            loss.backward()
            del loss, inputs, batch
    finally:
        for hook in hooks:
            hook.remove()
        template.remove_post_encode_hook()
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        for param in saved_trainables:
            param.requires_grad_(True)
        model.train(was_training)

    if gradients.keys() != layers.keys():
        raise ValueError("Missing RaLoRA full-weight gradients")
    allocation = {}
    for name, layer in layers.items():
        grad = gradients.pop(name).div_(count).to(layer.down.device)
        n, effective = block_count(grad, layer.rank, max_power)
        del grad
        layer.configure(n)
        allocation[name] = {"blocks": n, "effective_gradient_rank": effective}
    return {"calibration_train_indices": indices, "calibration_steps": count,
            "calibration_seconds": time.perf_counter()-before, "allocation": allocation,
            "calibration_peak_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "calibration_peak_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
