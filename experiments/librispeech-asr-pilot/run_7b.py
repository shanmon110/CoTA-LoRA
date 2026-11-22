"""LibriSpeech ASR training, development-set selection and held-out decoding."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.librispeech.asr.common import MANIFEST_HASHES, METHODS, ROOT, batches, save, sha
from examples.librispeech.asr.local.prepare_data import ACCUM, BATCH, COUNTS, DEFAULT_CONFIG, prepare_splits, read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?", help="Legacy positional output directory")
    p.add_argument("--dir", type=Path, help="Experiment directory")
    p.add_argument("--stage", choices=("data", "train", "decode", "all"), default="all")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-dir", type=Path, help="Prepared recipe data directory (plan/ and manifests/)")
    p.add_argument("--corpus", type=Path, help="Index extracted LibriSpeech subsets into a new --data-dir")
    p.add_argument("--early-stopping", action="store_true", help="Validate online at quarter-epoch intervals.")
    args = p.parse_args()
    conf = json.loads(args.config.read_text())
    plan = ROOT / conf["plan"]
    test_dir = ROOT / conf["test_manifests"]
    base = ROOT / conf["model"]
    base_hash = conf["base_sha256"]
    if args.corpus and (args.stage != "data" or not args.data_dir):
        p.error("--corpus requires --stage data and a new --data-dir")
    expected_hashes = MANIFEST_HASHES
    if args.data_dir:
        plan = args.data_dir.resolve() / "plan"
        test_dir = args.data_dir.resolve() / "manifests"
    if args.corpus:
        from examples.omni_talora import prepare_manifest, preflight

        args.data_dir.mkdir(parents=True, exist_ok=False)
        for split, subset in (("train", "train-clean-100"), ("dev", "dev-clean"),
                              ("test-clean", "test-clean"), ("test-other", "test-other")):
            prepare_manifest.main(["--task", "asr", "--root", str(args.corpus / subset),
                                   "--output", str(test_dir / (split + ".jsonl"))])
        preflight.main(["--task", "asr", "--train-manifest", str(test_dir / "train.jsonl"),
                        "--dev-manifest", str(test_dir / "dev.jsonl"), "--output-dir", str(plan),
                        "--batch-size", str(BATCH), "--model-name-or-path", str(base), "--prefilter"])
    if args.data_dir and args.stage != "decode":
        summary = json.loads((plan / "summary.json").read_text())["splits"]
        expected_hashes = {name: summary[name]["filtered_sha256"] for name in ("train", "dev")}
    if args.stage == "data":
        _, hashes, _, limits = prepare_splits(plan, test_dir, expected_hashes)
        print(json.dumps(dict(counts=COUNTS, manifest_sha256=hashes, input_budget=limits), indent=2))
        return
    if bool(args.dir) == bool(args.run_dir):
        p.error("Supply one experiment directory with --dir or a positional argument")
    out = (args.dir or args.run_dir).resolve()
    state_file = out / "status.json"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    if args.stage == "decode":
        status = json.loads(state_file.read_text())
        chosen = json.loads((out / "dev-selection.json").read_text())
        if (status["state"] != "trained" or set(chosen) != set(METHODS)
                or any(any(status["results"][m][k] != chosen[m][k] for k in
                       ("selected_step", "selected_adapter", "dev", "adapter_sha256")) for m in METHODS)):
            raise ValueError("Decoding requires a completed training stage with all seven selections")
        paths = {name: test_dir / (name + ".jsonl") for name in ("test-clean", "test-other")}
        hashes = status["manifest_sha256"]
        ids = {name: [r["id"] for r in read_rows(path)] for name, path in paths.items()}
        status.update(state="running", pid=os.getpid())
    else:
        out.mkdir(parents=True, exist_ok=False)
        status = dict(state="running", phase="prepare", results={}, pid=os.getpid())
    save(state_file, status)
    try:
        if args.stage != "decode":
            paths, hashes, ids, limits = prepare_splits(plan, test_dir, expected_hashes)
            steps = math.ceil(COUNTS["train"] / (BATCH * ACCUM))
            mid = steps // 2
            anchors = [mid, steps]
            status.update(model=str(base), base_sha256=base_hash, manifest_sha256=hashes, input_budget=limits,
                          batch_size=BATCH, accumulation_steps=ACCUM, epochs=1, candidates=anchors,
                          split_counts=COUNTS, methods=list(METHODS), seed=42, max_new_tokens=128,
                          selection="lowest corpus dev WER; earlier step on tie")
            if args.early_stopping:
                interval = steps // 4
                anchors = list(range(interval, steps, interval)) + [steps]
                status.update(candidates=anchors, dev_interval=interval, patience=1,
                              selection="full dev WER every quarter epoch; stop on first tie or increase; earliest best wins")
                from torch import Generator
                from torch.utils.data import DataLoader
                indices = DataLoader(range(len(ids["train"])), batch_size=BATCH, shuffle=True,
                                     generator=Generator().manual_seed(42))
                expected_order = [ids["train"][i] for batch in indices for i in batch.tolist()]
            save(state_file, status)
            budget_args = [v for name, val in limits.items() for v in ("--"+name.replace("_", "-"), str(val))]
            (out / "device.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
        order_hash = None
        word_counts = {}

        def run(method, tag, split, options, training=False):
            status.update(phase=tag, method=method)
            save(state_file, status)
            if sha(paths[split]) != hashes[split]:
                raise ValueError(f"{split} manifest changed")
            prefix = out / method / tag
            events_path = prefix.with_suffix(".jsonl")
            module = "train" if training else "evaluate"
            cmd = [sys.executable, "-u", "-m", "examples.omni_talora." + module,
                   "--device", "cuda:0", "--batch-size", str(BATCH), "--manifest", str(paths[split]),
                   "--smoke-log", str(events_path), *options]
            with (out / "commands.txt").open("a") as commands:
                print(shlex.join(cmd), file=commands)
            print(method, tag, flush=True)
            with prefix.with_suffix(".log").open("x") as log, prefix.with_suffix(".stderr").open("x") as err:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=err, check=True)
            events = read_rows(events_path)
            if not any(e["event"] == "stage_end" and e["stage"] == module and e["status"] == "ok" for e in events):
                raise RuntimeError(f"Incomplete {tag} events")
            return prefix.with_suffix(".log"), batches(events, module)

        def evaluate(method, split, candidate, step):
            result = status["results"][method]
            if sha(candidate / "adapter.safetensors") != result["adapter_sha256"][str(step)]:
                raise ValueError("Candidate weights changed")
            log, order = run(method, f"{split}-step-{step}", split,
                             ["--adapter", str(candidate), "--max-new-tokens", "128"])
            score = json.loads(log.read_text())
            if (ids[split] != order or score["utterances"] != len(order)
                    or score["reference_words"] <= 0 or score["max_new_tokens"] != 128
                    or score["reference_words"] != word_counts.setdefault(split, score["reference_words"])):
                raise ValueError(f"Incomplete or inconsistent {split} evaluation")
            return score

        for method, extra in (METHODS.items() if args.stage != "decode" else ()):
            directory = out / method
            directory.mkdir()
            adapter = directory / "adapter"
            selection_args = (["--dev-manifest", str(paths["dev"]), "--dev-interval", str(interval),
                               "--dev-patience", "1"] if args.early_stopping else ["--save-midpoint"])
            if sha(paths["dev"]) != hashes["dev"]:
                raise ValueError("Development manifest changed")
            opts = ["--task", "asr", "--model-name-or-path", str(base), "--base-sha256", base_hash,
                    "--one-epoch", *selection_args, "--accumulation-steps", str(ACCUM), "--seed", "42",
                    "--gradient-checkpointing", "--rank", "8", "--last-n", "4", "--lr", "0.0002",
                    "--alpha", "10" if method == "qv-lora" else "16", "--output", str(adapter),
                    *budget_args, *extra]
            log, order = run(method, "train", "train", opts, training=True)
            actual_hash = hashlib.sha256("\n".join(order).encode()).hexdigest()
            if not args.early_stopping and (len(order) != len(ids["train"]) or set(order) != set(ids["train"])
                    or order_hash not in (None, actual_hash)):
                raise ValueError("Training epoch or order differs between methods")
            order_hash = actual_hash
            records = read_rows(log)
            train, = [r for r in records if "effective_scale" in r]
            losses = [r for r in records if "step" in r]
            finished = len(losses) if args.early_stopping else steps
            if ([r["step"] for r in losses] != list(range(1, finished + 1))
                    or not all(math.isfinite(r["loss"]) for r in losses)
                    or train["effective_scale"] != 2 or train["resolved_rank"] != (5 if method == "qv-lora" else 8)):
                raise ValueError("Training protocol mismatch")
            if args.early_stopping:
                if (finished not in anchors or order != expected_order[:min(len(expected_order), finished * BATCH * ACCUM)]):
                    raise ValueError("Training did not consume the common seeded order up to the stopping point")
                history = read_rows(adapter / "dev-history.jsonl")
                if [h["validation_step"] for h in history] != [i for i in anchors if i <= finished]:
                    raise ValueError("Missing validation checkpoint")
                if sha(paths["dev"]) != hashes["dev"]:
                    raise ValueError("Development manifest changed during training")
                dev_scores, weights = {}, {}
                best = None
                for h in history:
                    i, score = h["validation_step"], h["score"]
                    dev_order = batches(read_rows(adapter / f"dev-{i}.jsonl"), "evaluate")
                    if (dev_order != ids["dev"] or score["utterances"] != len(dev_order)
                            or score["max_new_tokens"] != 128 or score["reference_words"] <= 0
                            or score["reference_words"] != word_counts.setdefault("dev", score["reference_words"])):
                        raise ValueError("Incomplete online dev evaluation")
                    dev_scores[str(i)] = score
                    weights[str(i)] = sha(adapter / f"step-{i}" / "adapter.safetensors")
                    improved = best is None or score["word_errors"] < dev_scores[str(best)]["word_errors"]
                    if improved:
                        best = i
                    if (h["best_step"] != best or h["early_stopped"] != (not improved and i < steps)
                            or (not improved and i != finished)):
                        raise ValueError("Early-stopping decision disagrees with dev scores")
                if finished < steps and not history[-1]["early_stopped"]:
                    raise ValueError("Training ended before a stopping decision")
                status["results"][method] = dict(train=train, train_order_sha256=actual_hash, trained_steps=finished,
                    trained_examples=len(order), dev=dev_scores, adapter_sha256=weights, selected_step=best,
                    selected_adapter=str(adapter / f"step-{best}"), early_stopped=history[-1]["early_stopped"])
                save(state_file, status)
                continue
            candidates = {mid: adapter / f"step-{mid}", steps: adapter}
            result = dict(train=train, train_order_sha256=order_hash, dev={},
                          adapter_sha256={str(i): sha(d / "adapter.safetensors") for i,d in candidates.items()})
            status["results"][method] = result
            for i, candidate in candidates.items():
                result["dev"][str(i)] = evaluate(method, "dev", candidate, i)
                save(state_file, status)
            best = min(anchors, key=lambda i: (result["dev"][str(i)]["word_errors"], i))
            result.update(selected_step=best, selected_adapter=str(candidates[best]))
            save(state_file, status)
        if args.stage != "decode":
            save(out / "dev-selection.json", {m: {k: r[k] for k in ("selected_step", "selected_adapter", "dev", "adapter_sha256")}
                                              for m,r in status["results"].items()})
        if args.stage == "train":
            status.update(state="trained", phase=None, method=None)
            return
        for method, result in status["results"].items():
            result["test"] = {}
            for split in ("test-clean", "test-other"):
                result["test"][split] = evaluate(method, split, Path(result["selected_adapter"]), result["selected_step"])
                save(state_file, status)
        status.update(state="completed", phase=None, method=None)
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        save(state_file, status)


if __name__ == "__main__":
    main()
