"""ICLR LoRA-Mixer joint expert preparation and serial attention-projection routing.

Based on the authors' supplementary llama3-train.py (archive SHA-256
2b87e32bd63ab60fb13a4b52294f953f9dba172dae9146a240cd60a3fc9dc486).
Uses Eq. 5 means over real tokens instead of letting padded length scale RSL.
"""
import hashlib
import json
import math
from pathlib import Path
import random

from safetensors.torch import save_file
import torch
from torch import nn
from torch.nn import functional as F

from .paper_late_stages import stage_loss, stage_session


class Expert(nn.Module):
    def __init__(self,base,rank):
        super().__init__()
        kw = dict(bias=False, device=base.weight.device, dtype=torch.float32)
        self.a = nn.Linear(base.in_features,rank,**kw)
        self.b = nn.Linear(rank,base.out_features,**kw)
        nn.init.zeros_(self.b.weight)
        self.register_buffer("initial_a",torch.zeros_like(self.a.weight))
        self.register_buffer("initial_b",torch.zeros_like(self.b.weight))

    def forward(self,x):
        return self.b(self.a(x))


class MixerLinear(nn.Module):
    def __init__(self,base,rank,alpha,settings):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        n = settings["num_experts"]
        self.top_k = settings["top_k"]
        if rank % n or not 1 <= self.top_k <= n or alpha != rank:
            raise ValueError("LoRA-Mixer requires divisible total rank, valid top-k, and unit scaling")
        self.experts = nn.ModuleList(Expert(base,rank//n) for _ in range(n))
        h = settings["router_hidden"]
        kw = dict(dtype=torch.float32,device=base.weight.device)
        self.router = nn.Sequential(nn.Linear(base.in_features,h,**kw),nn.GELU(),nn.Linear(h,n,**kw))
        nn.init.xavier_normal_(self.router[0].weight,gain=math.sqrt(2))
        nn.init.zeros_(self.router[0].bias)
        nn.init.xavier_normal_(self.router[2].weight)
        self.hard_domain = None
        self.valid_tokens = None
        self.routing_terms = None

    @torch.no_grad()
    def seal_experts(self):
        for expert in self.experts:
            expert.initial_a.copy_(expert.a.weight)
            expert.initial_b.copy_(expert.b.weight)

    def forward(self,x):
        base = self.base_layer(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.hard_domain is not None:
                delta = self.experts[self.hard_domain](x)
            else:
                p = self.router(x).softmax(-1)
                # The authors use soft fusion on train and inference. Hard
                # top-k contributes to the empirical expert loads in the loss.
                delta = sum(ex(x)*p[...,i,None] for i,ex in enumerate(self.experts))
                if self.training:
                    if self.valid_tokens is None or self.valid_tokens.shape != x.shape[:-1]:
                        raise ValueError("LoRA-Mixer needs the aligned training attention mask")
                    valid = self.valid_tokens.to(device=x.device).reshape(-1,1)
                    probs = p.reshape(-1,p.shape[-1])
                    num = valid.sum().clamp_min(1)
                    importance = (probs*valid).sum(0)/num
                    top = F.one_hot(probs.topk(self.top_k).indices, len(self.experts)).float().mean(1)
                    load = (top*valid).sum(0)/num
                    entropy = -(probs*probs.clamp_min(1e-9).log()*valid).sum()/num
                    self.routing_terms = ((importance*load).sum(), entropy)
            return (base.float()+delta).to(base.dtype)


def set_batch_mask(model,mask):
    if mask is None or mask.ndim != 2:
        raise ValueError("LoRA-Mixer requires a two-dimensional collator attention mask")
    for m in model.modules():
        if isinstance(m,MixerLinear):
            m.valid_tokens = mask


def mixer_loss(model,settings):
    layers = [m for m in model.modules() if isinstance(m,MixerLinear)]
    if not layers:
        raise ValueError("LoRA-Mixer has no routed projections")
    if layers[0].hard_domain is not None or not model.training:
        return layers[0].base_layer.weight.new_zeros((),dtype=torch.float32)
    aux, anchors = [], []
    for m in layers:
        if m.routing_terms is None:
            raise ValueError("Missing RSL statistics from forward")
        balance, entropy = m.routing_terms
        m.routing_terms = None
        aux.append(settings.get("balance_alpha",0.1)*balance-settings.get("entropy_gamma",0.1)*entropy)
        for ex in m.experts:
            anchors.extend([(ex.a.weight-ex.initial_a).square().sum(),(ex.b.weight-ex.initial_b).square().sum()])
    return (settings.get("balance_loss_weight",0.1)*torch.stack(aux).mean()
            + settings.get("preserve_weight",1e-5)*torch.stack(anchors).sum())


def _audio_key(row):
    audio, = row["audios"]
    if isinstance(audio,dict):
        audio = audio["path"]
    return (Path(audio).name,row["messages"][0]["content"],row["messages"][-1]["content"])


def prepare_experts(trainer,config):
    settings = config["method_options"]["lora-mixer"]
    domains = settings["domains"]
    if len(domains) != settings["num_experts"] or len(set(domains)) != len(domains):
        raise ValueError("Define a distinct training domain for every expert")
    path = Path(settings["source_manifest"])
    out = Path(trainer.args.output_dir)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != json.loads((out/"protocol.json").read_text())["train_sha256"]:
        raise ValueError("Expert supervision must use the same source training manifest as SFT")
    source = {}
    withheld_clips = set()
    heldout_types = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = _audio_key(row)
        value = row["question_type"]
        record = (value, row["source_id"])
        if key in source and source[key] != record:
            raise ValueError("Source question maps to two expert domains")
        source[key] = record
        if value not in domains:
            withheld_clips.add(row["source_id"])
            heldout_types.add(value)
    raw = trainer.train_dataset.dataset
    buckets = [[] for _ in domains]
    held_out = []
    for i in range(len(raw)):
        domain,clip = source[_audio_key(raw[i])]
        if domain in domains and clip not in withheld_clips:
            buckets[domains.index(domain)].append(i)
        else:
            held_out.append(i)
    per = settings["examples_per_expert"]
    accum = settings["stage_accumulation"]
    if accum < 1 or per < 1 or per % accum or any(len(b)<per for b in buckets):
        raise ValueError("Invalid data budget for domain-specific LoRA warm-up")
    rng = random.Random(trainer.args.data_seed)
    chosen = [rng.sample(b,per) for b in buckets]
    batches = [(d,ids[i:i+accum]) for d,ids in enumerate(chosen) for i in range(0,per,accum)]
    rng.shuffle(batches)
    model = trainer.model
    layers = {name:m for name,m in model.named_modules() if isinstance(m,MixerLinear)}
    if not layers: raise ValueError("Cannot prepare an empty expert library")
    params = [p for m in layers.values() for ex in m.experts for p in ex.parameters()]
    optimizer = torch.optim.AdamW(params,lr=trainer.args.learning_rate,weight_decay=trainer.args.weight_decay)
    with stage_session(trainer) as measurements:
        try:
            for domain,ids in batches:
                for m in layers.values(): m.hard_domain = domain
                optimizer.zero_grad(set_to_none=True)
                for i in ids:
                    (stage_loss(trainer,[i])/len(ids)).backward()
                nn.utils.clip_grad_norm_(params, trainer.args.max_grad_norm)
                optimizer.step()
        finally:
            for m in layers.values(): m.hard_domain = None
    for m in layers.values(): m.seal_experts()
    library = out/"source-experts"
    library.mkdir(exist_ok=True)
    entries = []
    for d,ids in enumerate(chosen):
        values = {}
        for name,m in layers.items():
            if name.rsplit(".",1)[-1] in ("q_proj","v_proj"):
                values[name+".a"] = m.experts[d].a.weight.detach().cpu().contiguous()
                values[name+".b"] = m.experts[d].b.weight.detach().cpu().contiguous()
        if not values:
            raise ValueError("No Q/V factors in the expert library")
        destination = library/f"expert-{d}.safetensors"
        save_file(values,str(destination))
        entries.append({"domain":domains[d],"description":settings["descriptions"][d],
                        "train_indices":ids,"file":destination.name,
                        "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()})
    metadata = {"model_revision":config["model_revision"],"train_sha256":sha,"experts":entries,
                "excluded_question_types": sorted(heldout_types),
                "excluded_raw_sources": sorted(withheld_clips),
                "rank":next(iter(layers.values())).experts[0].a.out_features,"factor_scale":1.}
    (library/"library.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return {**measurements,"expert_library":str(library/"library.json"),"warmup_steps":len(batches),
            "specialist_parameters_per_active_domain":sum(p.numel() for m in layers.values() for p in m.experts[0].parameters()),
            "specialist_trainable_parameters_total":sum(p.numel() for p in params),
            "source_domain_counts":{domain:len(b) for domain,b in zip(domains,buckets)},
            "source_examples_per_expert":per, "outside_expert_domains":len(held_out)}
