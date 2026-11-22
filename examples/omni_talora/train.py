"""Train one independent adapter. Run from the WEST root with python -m."""

import argparse
import json
import math
from pathlib import Path

from .runtime import load_base, local_snapshot_sha256, move_inputs
from .budget import Budget, add_budget_args
from dataclasses import asdict


def validate_output(directory):
    path = Path(directory)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose an empty or new adapter directory: {path}")


def train_updates(thinker, loader, optimizer, *, device, audio_token_id, max_steps,
                  accumulation_steps, one_epoch=False, after_update=None, start_step=0, iterator=None):
    """Run updates from start_step to max_steps, optionally continuing a live iterator."""
    import torch
    from .attention import audio_context

    if min(max_steps, accumulation_steps) < 1 or not len(loader):
        raise ValueError("Positive step counts and a nonempty loader are required.")
    thinker.train()
    params = [p for p in thinker.parameters() if p.requires_grad]
    dtype = next(thinker.parameters()).dtype
    history = []
    if iterator is None:
        iterator = iter(loader)
    for step in range(start_step, max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_tokens = 0
        total_loss = torch.zeros((), device=device)
        window = accumulation_steps
        if one_epoch:
            window = min(window, len(loader) - step * accumulation_steps)
        if window <= 0:
            raise ValueError("Requested updates exceed one epoch")
        for _ in range(window):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_inputs(batch, device, dtype)
            n = int(batch["labels"][:, 1:].ne(-100).sum().item())
            if not n:
                raise ValueError("A microbatch has no supervised next tokens.")
            # A non-reentrant checkpoint may replay attention in backward.
            # Keep the utterance's audio flags in scope until backward ends.
            with audio_context(batch["input_ids"], batch["attention_mask"], audio_token_id):
                loss = thinker(**batch, use_cache=False).loss.float()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite loss at update {step + 1}")
                (loss * n).backward()
            total_tokens += n
            total_loss += loss.detach() * n
        for p in params:
            if p.grad is not None:
                p.grad.div_(total_tokens)
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        mean = (total_loss / total_tokens).item()
        history.append(mean)
        print(json.dumps({"step": step + 1, "supervised_tokens": total_tokens, "loss": mean}), flush=True)
        if after_update is not None:
            after_update(step + 1)
    return history


def prepare_training(a, input_budget, adapter_options=None):
    """Load the bound base/processor and install the same adapter for either training frontend."""
    from functools import partial
    from .diagnostics import SmokeLog
    from transformers import set_seed
    from .adapters import install_from_config
    from west_adapters import resolve_layout, finalize_ranks
    from .data import OmniCollator, load_manifest

    set_seed(a.seed)
    data = load_manifest(a.manifest, a.task)
    source = a.model_name_or_path
    base_digest = None
    if Path(source).is_dir():
        source = str(Path(source).resolve())
        base_digest = local_snapshot_sha256(source)
    if a.base_sha256 is not None and a.base_sha256 != base_digest:
        raise ValueError("Requested local snapshot digest does not match the base and processor")
    smoke = SmokeLog(a.smoke_log, a.device)
    with smoke.stage("load"):
        model, processor = load_base(source, a.revision, a.device)
    collate = OmniCollator(processor, a.task, budget=input_budget)
    if a.smoke_log:
        collate = partial(smoke.collate, collate)
    options = vars(a) if adapter_options is None else adapter_options
    method, config, projs = resolve_layout(model, options)
    rank = config["rank"]
    config.update(model_name_or_path=source, revision=getattr(model.config, "_commit_hash", None),
                  task=a.task)
    if base_digest:
        config["base_sha256"] = base_digest
    if any(value is not None for value in asdict(input_budget).values()):
        config["input_budget"] = asdict(input_budget)
    print(json.dumps({"base_sha256": base_digest, "revision": config["revision"]}), flush=True)
    directions = None
    if method.uses_directions:
        with smoke.stage("calibration"):
            directions = method.calibrate(model, config, data, collate, a.batch_size, a.device)
    ranks = finalize_ranks(config, projs, directions)
    with smoke.stage("adapter_setup"):
        install_from_config(model, config, directions=directions)
    del directions
    params = [v for v in model.parameters() if v.requires_grad]
    print(json.dumps({"task": a.task, "mode": method.mode, "examples": len(data),
                      "requested_rank": options["rank"], "resolved_rank": rank, "target_ranks": ranks,
                      "alpha": config["alpha"], "effective_scale": config["alpha"] / rank,
                      "residual_precision": config["residual_precision"],
                      "trainable_parameters": sum(v.numel() for v in params)}), flush=True)
    return model, processor, data, collate, config, smoke


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--revision", help="Hub revision; resolved commit is saved for evaluation.")
    p.add_argument("--base-sha256", help="Require this content digest for the local Omni snapshot.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["asr", "ser"], required=True)
    p.add_argument("--mode", choices=["ta", "lora", "toplora", "bora", "dora", "eva"], default="ta")
    p.add_argument("--target-set", choices=["o", "qv"], default="o")
    p.add_argument("--match-ta-budget", action="store_true",
                   help="For Q/V-LoRA, choose the nearest uniform rank to TA at --rank.")
    p.add_argument("--bora-blocks", type=int, default=2)
    p.add_argument("--eva-samples", type=int, default=128, help="Maximum number of training clips for calibration.")
    p.add_argument("--eva-rho", type=float, default=2., help="Maximum layer rank multiplier for EVA.")
    p.add_argument("--eva-seed", type=int, default=42)
    p.add_argument("--eva-token-mask", choices=["labels", "nonpad"], default="labels",
                   help="EVA SVD rows: supervised answer/EOS tokens, or all non-padding tokens.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.)
    p.add_argument("--last-n", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=8)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--max-steps", type=int, default=100)
    schedule.add_argument("--one-epoch", action="store_true", help="Consume every training record once.")
    p.add_argument("--save-midpoint", action="store_true",
                   help="Also save the adapter after half the optimizer updates for dev selection.")
    p.add_argument("--dev-manifest", help="Enable online early stopping on full ASR dev WER.")
    p.add_argument("--dev-interval", type=int, default=892)
    p.add_argument("--dev-patience", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Use non-reentrant activation checkpointing during SFT.")
    p.add_argument("--smoke-log", help="Append per-stage memory peaks and input lengths to JSONL.")
    add_budget_args(p)
    a = p.parse_args(argv)
    input_budget = Budget.from_args(a)
    if a.dev_manifest and (a.task != "asr" or a.save_midpoint or min(a.dev_interval, a.dev_patience) < 1
                           or Path(a.dev_manifest).resolve() == Path(a.manifest).resolve()):
        p.error("Online validation requires distinct ASR data, positive interval/patience, and no --save-midpoint")
    if min(a.rank, a.last_n, a.batch_size, a.accumulation_steps, a.max_steps) <= 0:
        p.error("rank, last-n, batch-size and step counts must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in (a.lr, a.alpha)):
        p.error("lr and alpha must be finite and positive")
    if a.target_set == "qv" and a.mode != "lora":
        p.error("--target-set qv requires --mode lora")
    if a.match_ta_budget and a.target_set != "qv":
        p.error("--match-ta-budget requires --target-set qv")
    if a.mode == "bora" and a.bora_blocks < 1:
        p.error("--bora-blocks must be positive")
    if a.mode == "eva" and (a.eva_samples < 1 or not math.isfinite(a.eva_rho) or a.eva_rho < 1):
        p.error("EVA requires positive --eva-samples and finite --eva-rho >= 1")
    validate_output(a.output)
    import torch
    from torch.utils.data import DataLoader
    from .checkpoint import save_adapter

    model, processor, data, collate, config, smoke = prepare_training(a, input_budget)
    if a.gradient_checkpointing:
        checkpoint_kwargs = {"use_reentrant": False}
        from west_adapters.registry import checkpoint_method
        if checkpoint_method(config).needs_audio_context:
            from .attention import checkpoint_audio_context
            checkpoint_kwargs["context_fn"] = checkpoint_audio_context
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
    params = [v for v in model.parameters() if v.requires_grad]
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if a.one_epoch:
        a.max_steps = math.ceil(len(loader) / a.accumulation_steps)
    updates = a.max_steps
    if a.save_midpoint and updates < 2:
        raise ValueError("Midpoint selection requires at least two updates")

    def save_midpoint(step):
        if step == updates // 2:
            save_adapter(model, Path(a.output) / f"step-{step}", config)

    kwargs = dict(device=a.device, audio_token_id=model.config.audio_token_id,
                  accumulation_steps=a.accumulation_steps, one_epoch=a.one_epoch)
    if a.dev_manifest:
        from .data import load_manifest
        from .evaluate import evaluate
        from .diagnostics import SmokeLog
        from .checkpoint import load_adapter

        dev = load_manifest(a.dev_manifest, "asr")
        train_files = {row["audio"] for row in data}
        if any(row["audio"] in train_files for row in dev):
            raise ValueError("Dev audio overlaps training data")
        output = Path(a.output)
        iterator = iter(loader)
        best_errors, best_step, misses = math.inf, None, 0
        start = 0
        while start < updates:
            end = min(start + a.dev_interval, updates)
            with smoke.stage("train"):
                train_updates(model, loader, opt, start_step=start, max_steps=end, iterator=iterator, **kwargs)
            candidate = output / f"step-{end}"
            save_adapter(model, candidate, config)
            dev_log = SmokeLog(str(output / f"dev-{end}.jsonl"), a.device)
            with dev_log.stage("evaluate"):
                score = evaluate(model, processor, dev, task="asr", device=a.device,
                                 batch_size=a.batch_size, max_new_tokens=128, budget=input_budget, smoke=dev_log)
            if score["word_errors"] < best_errors:
                best_errors, best_step, misses = score["word_errors"], end, 0
            else:
                misses += 1
            decision = dict(validation_step=end, score=score, best_step=best_step,
                            non_improving_checks=misses, early_stopped=misses >= a.dev_patience and end < updates)
            with (output / "dev-history.jsonl").open("a") as log:
                print(json.dumps(decision), file=log)
            print(json.dumps(decision), flush=True)
            if misses >= a.dev_patience:
                break
            start = end
        load_adapter(model, output / f"step-{best_step}")
    else:
        with smoke.stage("train"):
            train_updates(model, loader, opt, max_steps=updates, **kwargs,
                          after_update=save_midpoint if a.save_midpoint else None)
    save_adapter(model, a.output, config)
    with (Path(a.output) / "train_args.json").open("x", encoding="utf-8") as f:
        json.dump(vars(a), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
