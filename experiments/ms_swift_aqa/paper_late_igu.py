"""IGU-LoRA pruning and final fitting with an immutable rank mask.

Adapted from Cuixuans/igulora, 3f5ba10a6646c81c557e4669b14b072f76822085,
iglora/{lora_module,run_pruning_sft,rank_allocator,run_final_sft}.py.
The release uses one sampled path coefficient per batch on all three SVD
factors; its forward therefore multiplies the update by the coefficient cubed.
"""
import random
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .paper_late_stages import stage_loss, stage_session, train_indices


class IGULinear(nn.Module):
    def __init__(self, base, rank, alpha, settings):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.final_rank = rank
        initial = rank * settings.get("initial_rank_multiplier", 2)
        if type(initial) is not int or initial < rank or initial > min(base.in_features,base.out_features):
            raise ValueError("Invalid IGU candidate rank")
        self.scale = alpha / (initial + 1e-5)
        kw = dict(dtype=torch.float32, device=base.weight.device)
        self.down = nn.Parameter(torch.empty(initial, base.in_features, **kw))
        self.up = nn.Parameter(torch.empty(base.out_features, initial, **kw))
        self.diag = nn.Parameter(torch.empty(initial, **kw))
        self.register_buffer("mask", torch.ones(initial, **kw))
        self.dropout = nn.Dropout(settings.get("dropout", 0.))
        self.path_coefficient = 1.
        self.reset_factors()

    @torch.no_grad()
    def reset_factors(self):
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=0.02)
        self.diag.fill_(0.1)

    def forward(self,x):
        base = self.base_layer(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = F.linear(self.dropout(x.float()), self.down) * (self.diag*self.mask)
            delta = F.linear(z, self.up) * (self.scale*self.path_coefficient**3)
            return (base.float()+delta).to(base.dtype)


def orthogonality(layers):
    terms = []
    for layer in layers:
        with torch.autocast(device_type=layer.down.device.type, enabled=False):
            for gram in (layer.down @ layer.down.T, layer.up.T @ layer.up):
                terms.append(torch.linalg.vector_norm(gram - torch.diag_embed(torch.diagonal(gram))))
    return torch.stack(terms).mean()


class RankScores:
    def __init__(self, layers, beta1, beta2):
        self.layers = layers
        self.avg = {}
        self.unc = {}
        self.beta1, self.beta2 = beta1, beta2
        for name, layer in layers.items():
            for key in ("down", "up", "diag"):
                p = getattr(layer,key)
                self.avg[name,key] = torch.zeros_like(p)
                self.unc[name,key] = torch.zeros_like(p)

    @torch.no_grad()
    def update(self):
        for name, layer in self.layers.items():
            for key in ("down","up","diag"):
                p = getattr(layer,key)
                sensitivity = (p*p.grad).abs()
                mean = self.avg[name,key].lerp_(sensitivity,1-self.beta1)
                self.unc[name,key].lerp_((sensitivity-mean).abs(),1-self.beta2)

    @torch.no_grad()
    def allocate(self, budget, *, final=True):
        scored = []
        for name, layer in self.layers.items():
            def score(key):
                return (self.avg[name,key]+1e-6)/(self.unc[name,key]+1e-6)
            combined = score("diag")+score("down").mean(1)+score("up").mean(0)
            # Generalize the authors' Llama MLP/attention multiplier to real
            # Omni GQA and MLP widths. Each selected slot has this exact cost.
            cost = layer.down.size(1)+layer.up.size(0)+1
            scored.extend((s,name,i,cost) for i,s in enumerate(combined.tolist()))
        selected = {name: [] for name in self.layers}
        remaining = budget
        for _,name,i,cost in sorted(scored, reverse=True):
            if cost <= remaining:
                selected[name].append(i)
                remaining -= cost
        for name, layer in self.layers.items():
            keep = torch.zeros_like(layer.diag, dtype=torch.bool)
            keep[selected[name]] = True
            # The original rank allocator zeroes E at a pruning boundary.
            # Intermediate zeroes can regrow; only final SFT freezes a mask.
            layer.diag.masked_fill_(~keep, 0.)
            if final:
                layer.mask.copy_(keep)
        return budget-remaining


@contextmanager
def _pruning_phase(model, layers, accumulation):
    # Authors divide CE by accumulation but add one full orthogonality term
    # per microbatch while pruning. Final SFT uses CE+orth at equal weight.
    model._igu_pruning_accumulation = accumulation
    try:
        yield
    finally:
        del model._igu_pruning_accumulation
        for layer in layers.values():
            layer.path_coefficient = 1.


def prune(trainer, settings):
    model = trainer.model
    layers = {name:layer for name,layer in model.named_modules() if isinstance(layer,IGULinear)}
    if not layers:
        raise ValueError("IGU-LoRA found no candidate projections")
    n_steps, accum = settings["pruning_steps"], settings["stage_accumulation"]
    interval, warmup = settings["prune_interval"], settings["prune_warmup"]
    if not (0 <= warmup < n_steps and interval >= 1 and accum >= 1):
        raise ValueError("Invalid IGU-LoRA pruning schedule")
    indices = train_indices(trainer,n_steps*accum)
    rng = random.Random(trainer.args.seed)
    nodes = settings["path_points"]
    if nodes < 2:
        raise ValueError("IGU-LoRA needs more than one integration node")
    scores = RankScores(layers, settings["beta1"], settings["beta2"])
    cost = lambda l: l.down.size(1)+l.up.size(0)+1
    initial = sum(cost(layer)*layer.diag.numel() for layer in layers.values())
    target = sum(cost(layer)*layer.final_rank for layer in layers.values())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=trainer.args.learning_rate,
                                  weight_decay=trainer.args.weight_decay)
    with stage_session(trainer) as metrics, _pruning_phase(model,layers,accum):
        for step in range(1,n_steps+1):
            optimizer.zero_grad(set_to_none=True)
            for i in indices[(step-1)*accum:step*accum]:
                coefficient = rng.randint(1,nodes)/nodes
                for layer in layers.values():
                    layer.path_coefficient = coefficient
                # trainer.compute_loss includes the authors' orthogonality loss.
                (stage_loss(trainer, [i])/accum).backward()
            if trainer.args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(parameters,trainer.args.max_grad_norm)
            optimizer.step()
            scores.update()
            if step > warmup and (step % interval == 0 or step == n_steps):
                f = (n_steps-step)/(n_steps-warmup)
                used = scores.allocate(round(target + (initial-target)*f**3), final=(step == n_steps))
    # The public final SFT script starts from new factors and the selected mask.
    for layer in layers.values():
        layer.reset_factors()
    ranks = {name:int(layer.mask.sum().item()) for name,layer in layers.items()}
    return {**metrics,"pruning_train_indices":indices,"pruning_optimizer_steps":n_steps,
            "target_active_parameters":target, "final_active_parameters":used, "active_ranks":ranks}
