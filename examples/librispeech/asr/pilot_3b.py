"""Run the seven independent, single-seed LibriSpeech ASR pilot arms on one GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from .common import MANIFEST_HASHES, METHODS, PLAN, ROOT, batches, save, sha

BASE = Path("/root/autodl-tmp/west-preflight/model")
BASE_HASH = "b09075962102a1d421f3d891bec565d2c40eedd0a8c5dc0668d749b9a6b00238"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="New output directory; existing runs are preserved.")
    args = p.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    common = ["--device", "cuda:0", "--batch-size", "1"]
    fixed = ["--task", "asr", "--model-name-or-path", str(BASE), "--base-sha256", BASE_HASH,
             "--last-n", "4", "--rank", "8", "--gradient-checkpointing", "--seed", "42",
             "--lr", "0.0002", "--max-steps", "100", "--accumulation-steps", "8",
             "--max-audio-seconds", "32.645", "--max-batch-frames", "3265",
             "--max-sequence-length", "883"]
    status = {"run_dir": str(out), "model": str(BASE), "base_sha256": BASE_HASH,
              "manifest_sha256": MANIFEST_HASHES, "results": {}}
    reference_order = None
    expected_ids = [json.loads(row)["id"] for row in (PLAN / "dev.jsonl").read_text().splitlines()]
    reference_words = None
    try:
        for phase, module, split in (("train", "train", "train"), ("evaluate", "evaluate", "dev")):
            manifest = PLAN / (split + ".jsonl")
            for tag, extra in METHODS.items():
                status.update(state="running", phase=phase, method=tag)
                save(out / "status.json", status)
                assert sha(manifest) == MANIFEST_HASHES[split], f"{split} manifest changed"
                directory = out / tag
                directory.mkdir(exist_ok=True)
                adapter = directory / "adapter"
                metrics = directory / (phase + "-events.jsonl")
                options = ([*fixed, *extra, "--alpha", "10" if tag == "qv-lora" else "16",
                            "--output", str(adapter)] if phase == "train" else
                           ["--adapter", str(adapter), "--max-new-tokens", "128"])
                cmd = [sys.executable, "-m", "examples.omni_talora." + module, *common, *options,
                       "--manifest", str(manifest), "--smoke-log", str(metrics)]
                with (out / "commands.txt").open("a") as commands:
                    print(shlex.join(cmd), file=commands)
                stdout = directory / (phase + ".log")
                print(f"{tag} {phase}", flush=True)
                with stdout.open("x") as log, (directory / (phase + ".stderr")).open("x") as err:
                    subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=err, check=True)
                events = [json.loads(row) for row in metrics.read_text().splitlines()]
                peaks = {e["stage"]: {k: e[k] for k in ("peak_allocated_bytes","peak_reserved_bytes","seconds")}
                         for e in events if e["event"] == "stage_end" and e["status"] == "ok"}
                assert phase in peaks
                order = batches(events, phase)
                if phase == "train":
                    steps = [json.loads(row) for row in stdout.read_text().splitlines() if row.startswith("{")]
                    summary, = [s for s in steps if "trainable_parameters" in s]
                    updates = [s for s in steps if "step" in s]
                    assert [s["step"] for s in updates] == list(range(1,101))
                    assert all(math.isfinite(s["loss"]) for s in updates)
                    assert len(order) == 800 and (reference_order is None or reference_order == order)
                    reference_order = order
                    assert summary["effective_scale"] == 2 and summary["residual_precision"] == "fp32_add_then_cast"
                    assert summary["resolved_rank"] == (5 if tag == "qv-lora" else 8)
                    assert sha(adapter / "adapter.safetensors")
                    status["results"][tag] = dict(train=summary, train_peaks=peaks,
                        adapter_sha256=sha(adapter / "adapter.safetensors"),
                        order_sha256=hashlib.sha256("\n".join(order).encode()).hexdigest(),
                        last_train_loss=updates[-1]["loss"])
                else:
                    score = json.loads(stdout.read_text())
                    assert order == expected_ids and len(order) == score["utterances"] == 2703
                    assert score["max_new_tokens"] == 128 and score["reference_words"] > 0
                    assert reference_words in (None, score["reference_words"])
                    reference_words = score["reference_words"]
                    status["results"][tag].update(evaluation=score, eval_peaks=peaks)
                save(out / "status.json", status)
        status.update(state="completed", phase=None, method=None)
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        save(out / "status.json", status)


if __name__ == "__main__":
    main()
