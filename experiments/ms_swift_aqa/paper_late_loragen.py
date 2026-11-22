"""Generate one Q/V PEFT adapter from an AVQA-only Qwen-Omni expert library."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .paper_late_gen.autoencoder import LoRAEncoder, StructureAware_LoRADecoder_MoE
from .paper_late_gen.transformer import LoRATransformer
from .paper_late_gen.diffusion import GaussianDiffusion1D


def _product_inner(a,b,c,d):
    # <BA,DC>_F via r x r products.
    return ((b.transpose(-1,-2) @ d) * (a @ c.transpose(-1,-2))).sum((-2,-1))


def _singular(a,b):
    # Singular values of BA equal those of the r x r triangular product.
    ra = torch.linalg.qr(a.transpose(-1,-2),mode="reduced").R
    rb = torch.linalg.qr(b,mode="reduced").R
    return torch.linalg.svdvals(rb @ ra.transpose(-1,-2))


def product_losses(target, prediction):
    angular, spectral = [], []
    for module in ("q","v"):
        a,b = target["A_"+module], target["B_"+module]
        c,d = prediction["A_"+module], prediction["B_"+module]
        norm = (_product_inner(a,b,a,b)*_product_inner(c,d,c,d)).clamp_min(1e-24).sqrt()
        angular.append((1-_product_inner(a,b,c,d)/norm).mean())
        with torch.no_grad():
            s = _singular(a,b)
            fraction = s.square()/s.square().sum(-1,keepdim=True).clamp_min(1e-16)
            keep = (fraction.cumsum(-1)-fraction < 0.85)
            weight = (s*keep)/(s*keep).sum(-1,keepdim=True).clamp_min(1e-12)
        spectral.append((weight*(s-_singular(c,d)).abs()).sum(-1).mean())
    return torch.stack(angular).mean(), torch.stack(spectral).mean()


def _decode(decoder,z):
    y = decoder(z)["decoder_only"]
    return {f"{f}_{p}":y[f"lora_{p}{f.lower()}"] for f in "AB" for p in "qv"}


def _step(loss,opt):
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError("Nonfinite LoRAGen objective")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def fit_generator(data,descriptions,auxiliary,settings,seed):
    """Fit solely to source experts. Last description is a prespecified new task."""
    torch.manual_seed(seed)
    n,layers,rank,dim_in = data["A_q"].shape
    qout,vout = data["B_q"].shape[-2],data["B_v"].shape[-2]
    device = descriptions.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    h,zdim = settings["hidden_dim"],settings["latent_dim"]
    if descriptions.shape[0] != n+1 or auxiliary.shape != descriptions.shape or n < 2:
        raise ValueError("Expected a description per source expert and one held-out description")
    encoder = LoRAEncoder(dim_in,rank,zdim,h,True,dim_in,qout,vout).to(device)
    decoder = StructureAware_LoRADecoder_MoE(zdim,rank,dim_in,hidden_dim=h,num_layers=layers,
                    num_experts=settings["num_experts"],top_k=settings["top_k"],noisy_gating=False,
                    shared_expert_pool=True,max_blocks=4*layers,dec_only=True,
                    dec_d_in=dim_in,dec_dq_out=qout,dec_dv_out=vout).to(device)
    # Upstream constructs unused T5 heads even for decoder-only backbones.
    for key in ("encoder_head","decoder_attn_head","cross_attn_head"):
        setattr(decoder,key,nn.Identity())
    ae_params = [*encoder.parameters(),*decoder.parameters()]
    aopt = torch.optim.AdamW(ae_params,lr=settings["ae_lr"],weight_decay=0.)
    bs = min(n,settings["batch_size"])
    start = time.perf_counter()
    for _ in range(settings["ae_steps"]):
        ids = torch.randperm(n,device=device)[:bs]
        src = {k:t[ids] for k,t in data.items()}
        mu,logvar = encoder(src).chunk(2,dim=-1)
        logvar = logvar.clamp(-30.,20.)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
        generated = _decode(decoder,z)
        direction,spectrum = product_losses(src,generated)
        kl = -0.5*(1+logvar-mu.square()-logvar.exp()).mean()
        loss = direction+2*spectrum+1e-6*kl+1e-5*decoder.moe_aux_loss()
        _step(loss,aopt)
    del aopt
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        latents = encoder(data).chunk(2,-1)[0]
        reconstruction,_ = product_losses(data,_decode(decoder,latents))
        scale = latents.abs().amax().clamp_min(1.)
        training_seq = (latents/scale).transpose(1,2).contiguous()
    vae_memory = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        vae_memory = (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
    vae_seconds = time.perf_counter()-start
    denoiser = LoRATransformer(zdim,zdim,zdim,descriptions.shape[1],auxiliary.shape[1],
                settings["denoiser_layers"],layernum=layers*4,pe="original").to(device)
    diffusion = GaussianDiffusion1D(denoiser,seq_length=layers*4,timesteps=settings["diffusion_steps"],
                    sampling_timesteps=settings["sampling_steps"],loss_type="l2",
                    objective="pred_v",beta_schedule="linear").to(device)
    dopt = torch.optim.Adam(diffusion.parameters(),lr=settings["denoiser_lr"])
    dstart = time.perf_counter()
    for _ in range(settings["diffusion_updates"]):
        ids = torch.randperm(n,device=device)[:bs]
        loss = diffusion(training_seq[ids],descriptions[ids],auxiliary[ids])
        _step(loss,dopt)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diffusion_seconds = time.perf_counter()-dstart
    diffusion.eval()

    @torch.no_grad()
    def generate(desc,aux):
        # Use exactly the same random path for the true/permuted controls.
        torch.manual_seed(seed+1)
        z = diffusion.sample(desc,aux,batch_size=len(desc)).transpose(1,2)*scale
        return _decode(decoder,z)

    with torch.no_grad():
        matched = generate(descriptions[:n],auxiliary[:n])
        permuted = generate(descriptions[:n].roll(1,0),auxiliary[:n].roll(1,0))
        matched_loss,_ = product_losses(data,matched)
        permuted_loss,_ = product_losses(data,permuted)
        generated = generate(descriptions[n:],auxiliary[n:])
        if not all(torch.isfinite(t).all() for t in generated.values()):
            raise FloatingPointError("Nonfinite generated adapter")
    diagnostics = {"source_experts":n, "lae_updates":settings["ae_steps"],
                   "diffusion_updates":settings["diffusion_updates"],
                   "source_reconstruction_product_cosine":1-float(reconstruction),
                   "source_generated_product_cosine":1-float(matched_loss),
                   "source_permuted_description_product_cosine":1-float(permuted_loss),
                   "lae_trainable_parameters":sum(p.numel() for p in ae_params),
                   "diffusion_trainable_parameters":sum(p.numel() for p in diffusion.parameters()),
                   "lae_seconds":vae_seconds,"diffusion_seconds":diffusion_seconds}
    if vae_memory is not None:
        diagnostics["lae_peak_allocated_bytes"], diagnostics["lae_peak_reserved_bytes"] = vae_memory
        diagnostics["diffusion_and_sampling_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        diagnostics["diffusion_and_sampling_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    state = {"encoder":encoder.state_dict(),"decoder":decoder.state_dict(),"diffusion":diffusion.state_dict(),
             "scale":scale,"descriptions":descriptions,"auxiliary":auxiliary}
    return generated, diagnostics, state


def peft_tensors(generated):
    result = {}
    for name,values in generated.items():
        f,p = name.split("_")
        if values.shape[0] != 1:
            raise ValueError("Only one prespecified adapter may be exported")
        for i,value in enumerate(values[0]):
            key = f"base_model.model.thinker.model.layers.{i}.self_attn.{p}_proj.lora_{f}.weight"
            result[key] = value.detach().cpu().float().contiguous()
    return result


def _load_library(metadata,path,device):
    packed = {}
    for record in metadata["experts"]:
        file = path.parent/record["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Specialist changed: {file}")
        state = load_file(str(file))
        expected = {f"thinker.model.layers.{i}.self_attn.{p}_proj.{f.lower()}"
                    for i in range(36) for p in "qv" for f in "AB"}
        if set(state) != expected:
            raise ValueError("LoRAGen library must contain all 36 Q/V projection factors")
        for p in "qv":
            for f in "AB":
                values = [state[f"thinker.model.layers.{i}.self_attn.{p}_proj.{f.lower()}"] for i in range(36)]
                packed.setdefault(f+"_"+p,[]).append(torch.stack(values))
    return {k:torch.stack(v).to(device=device,dtype=torch.float32) for k,v in packed.items()}


@torch.no_grad()
def _descriptions(config, texts):
    from .evaluate_frozen import load_frozen
    model,template = load_frozen(config)
    vectors = []
    for text in texts:
        tokens = template.tokenizer(text,return_tensors="pt").to(model.device)
        out = model.thinker.model(input_ids=tokens["input_ids"],attention_mask=tokens["attention_mask"],
                                  use_cache=False,return_dict=True)
        vectors.append(out.last_hidden_state.float().mean(1)[0])
    return torch.stack(vectors)


def publish_checkpoint(output, peft_config, weights, generator_state, report):
    """Expose the adapter and its provenance together, even across a retry."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh generator checkpoint destination: {output}")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".loragen-") as tmp:
        staging = Path(tmp)
        cp = staging/"checkpoint-generated"
        peft_config.save_pretrained(str(cp))
        weights_path = cp/"adapter_model.safetensors"
        save_file(weights, str(weights_path), metadata={"format":"pt"})
        torch.save(generator_state, staging/"source-generator.pt")
        report = dict(report, adapter_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                      adapter_config_sha256=hashlib.sha256((cp/"adapter_config.json").read_bytes()).hexdigest())
        (staging/"generation.json").write_text(json.dumps(report, indent=2)+"\n")
        staging.replace(output)


def validate_generation(config_path, library, output):
    """Bind the experiment, frozen specialist library and native adapter before target scoring."""
    config = json.loads(config_path.read_text())
    report = json.loads((output/"generation.json").read_text())
    cp = output/"checkpoint-generated"
    pairs = {"library_sha256":library, "experiment_sha256":config_path,
             "adapter_sha256":cp/"adapter_model.safetensors", "adapter_config_sha256":cp/"adapter_config.json"}
    if (report["model_revision"] != config["model_revision"]
            or report["source_train_sha256"] != config["method_options"]["loragen"]["train_sha256"]
            or any(report[k] != hashlib.sha256(path.read_bytes()).hexdigest() for k,path in pairs.items())):
        raise ValueError("LoRAGen provenance changed since generation")
    return report


def run(config_path,library,output):
    from peft import LoraConfig
    from experiments.ms_swift_librispeech.train import model_options
    config = json.loads(config_path.read_text())
    settings = config["method_options"]["loragen"]
    metadata = json.loads(library.read_text())
    source_protocol = json.loads((library.parent.parent/"protocol.json").read_text())
    if (source_protocol["method"] != "lora-mixer" or metadata["model_revision"] != config["model_revision"]
            or source_protocol["train_sha256"] != metadata["train_sha256"]
            or metadata["train_sha256"] != settings["train_sha256"] or metadata["factor_scale"] != 1.):
        raise ValueError("LoRAGen and source specialists must share the registered source train hash and base")
    domains = {e["domain"] for e in metadata["experts"]}
    heldout = set(settings["heldout_question_types"])
    if (not heldout or not heldout.issubset(metadata["excluded_question_types"])
            or domains & heldout or not metadata["excluded_raw_sources"]):
        raise ValueError("The generated task must be held out of all specialist preparation")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh generator checkpoint destination: {output}")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    texts = [e["description"] for e in metadata["experts"]] + [settings["source_description"]]
    descriptions = _descriptions(config,texts)
    embedding_peak_allocated = torch.cuda.max_memory_allocated()
    embedding_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.empty_cache()
    data = _load_library(metadata,library,descriptions.device)
    seed = config["training"]["seed"]
    generated,diagnostics,state = fit_generator(data,descriptions,torch.ones_like(descriptions),settings,seed)
    state["settings"], state["seed"] = settings,seed
    weights = peft_tensors(generated)
    rank = metadata["rank"]
    if config["adapter"]["rank"] != rank or config["adapter"]["alpha"] != rank:
        raise ValueError("Generated rank/unit scaling must agree with specialist library and evaluation config")
    if set(config["adapter"]["target_modules"]) != {"q_proj","v_proj"}:
        raise ValueError("LoRAGen decoder-only output must target exactly Q/V")
    opts = model_options(config,"lora")
    peft_cfg = LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,bias="none",task_type="CAUSAL_LM",
                          target_modules=opts["target_regex"],base_model_name_or_path=config["model"],
                          revision=config["model_revision"],inference_mode=True)
    torch.cuda.synchronize()
    init = json.loads((library.parent.parent/"initialization.json").read_text())
    elapsed = time.perf_counter()-start
    n = sum(t.numel() for t in weights.values())
    report = {**diagnostics,"source_domains":sorted(domains),"source_task_description":texts[-1],
              "experiment_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "heldout_question_types":sorted(heldout),"source_train_sha256":metadata["train_sha256"],
              "library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),
              "model_revision":config["model_revision"],"generated_adapter_parameters":n,
              "trainable_parameters":n,"generator_seconds_including_embedding":elapsed,
              "specialist_steps":init["warmup_steps"],"specialist_seconds":init["stage_seconds"],
              "specialist_parameters_per_active_domain":init["specialist_parameters_per_active_domain"],
              "specialist_trainable_parameters_total":init["specialist_trainable_parameters_total"],
              "specialist_peak_allocated_bytes":init["stage_peak_allocated"],
              "specialist_peak_reserved_bytes":init["stage_peak_reserved"],
              "embedding_peak_allocated_bytes":embedding_peak_allocated,"embedding_peak_reserved_bytes":embedding_peak_reserved,
              "optimizer_updates":init["warmup_steps"]+diagnostics["lae_updates"]+diagnostics["diffusion_updates"],
              "training_seconds":elapsed+init["stage_seconds"],
              "peak_allocated_bytes":max(torch.cuda.max_memory_allocated(),init["stage_peak_allocated"],
                                         diagnostics["lae_peak_allocated_bytes"],embedding_peak_allocated),
              "peak_reserved_bytes":max(torch.cuda.max_memory_reserved(),init["stage_peak_reserved"],
                                        diagnostics["lae_peak_reserved_bytes"],embedding_peak_reserved),
              "selection":"fixed source descriptor and seed before target evaluation"}
    publish_checkpoint(output, peft_cfg, weights, state, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config","library","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    run(args.config,args.library,args.output)
