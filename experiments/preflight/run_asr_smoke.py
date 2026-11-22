"""Reproduce the limited LibriSpeech GPU acceptance runs from the repository root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path("/root/autodl-tmp/west-preflight/model")
BASE_SHA256 = "b09075962102a1d421f3d891bec565d2c40eedd0a8c5dc0668d749b9a6b00238"
GENERATION_LIMIT = 128


def guarded_evaluation():
    """Audit actual CLI calls in a clean process against the held-out inputs only."""
    from unittest.mock import patch
    from examples.omni_talora import data, evaluate, eva

    del sys.argv[1]
    allowed = Path(sys.argv[sys.argv.index("--manifest") + 1]).resolve()
    rows = [json.loads(line) for line in allowed.read_text().splitlines()]
    audio = {(allowed.parent / row["audio"]).resolve() for row in rows}
    original_read, original_decode = data.load_manifest, data._load_audio

    def read(path, *args, **kwargs):
        assert Path(path).resolve() == allowed, f"Unexpected evaluation manifest: {path}"
        return original_read(path, *args, **kwargs)

    def decode(path, *args, **kwargs):
        assert Path(path).resolve() in audio, f"Unexpected evaluation audio: {path}"
        return original_decode(path, *args, **kwargs)

    with (patch.object(data, "load_manifest", read), patch.object(data, "_load_audio", decode),
          patch.object(eva, "calibrate_eva", side_effect=AssertionError("EVA refit during evaluation"))):
        evaluate.main()
    print("EVA reload guard passed (held-out manifest and audio only)", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path, help="New directory for adapters.")
    p.add_argument("--log-dir", required=True, type=Path, help="New directory for evidence.")
    p.add_argument("--methods", nargs="+", choices=["lora", "ta", "eva"],
                   default=["lora", "ta", "eva"], help="Select a subset for an isolated smoke rerun.")
    a = p.parse_args()
    selection = json.loads((HERE / "asr-selection.json").read_text())
    a.run_dir.mkdir(parents=True, exist_ok=False)
    a.log_dir.mkdir(parents=True, exist_ok=False)
    limits = [arg for k, v in selection["budget"].items()
              for arg in ("--" + k.replace("_", "-"), str(v))]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    devs = selection["splits"]["dev"]["representatives"]
    train = selection["splits"]["train"]["representatives"]
    common = ["--device", "cuda:0", "--batch-size", "1"]
    (a.log_dir / "device.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
    with (a.log_dir / "commands.txt").open("x") as commands:
        def run(tag, entry, args, item):
            manifest = Path(item["manifest"])
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == item["sha256"]
            cmd = [sys.executable, *entry, *args, *common, "--manifest", str(manifest),
                   "--smoke-log", str(a.log_dir / (tag + ".jsonl"))]
            print(shlex.join(cmd), file=commands, flush=True)
            print(tag, flush=True)
            with (a.log_dir / (tag + ".log")).open("x") as out:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)

        for mode in a.methods:
            for i, item in enumerate([selection["eva"]] if mode == "eva" else train):
                tag = f"{mode}-{i}"
                adapter = a.run_dir / tag
                run(tag + "-train", ["-m", "examples.omni_talora.train"],
                    ["--task", "asr", "--mode", mode, "--model-name-or-path", str(BASE),
                     "--base-sha256", BASE_SHA256, "--rank", "8", "--alpha", "16", "--last-n", "4",
                     "--accumulation-steps", "1", "--max-steps", "2", "--seed", "42",
                     "--output", str(adapter), "--gradient-checkpointing", *limits], item)
                from safetensors.torch import load_file
                state = load_file(adapter / "adapter.safetensors", device="cpu")
                changes = {k: int(v.count_nonzero()) for k, v in state.items()
                           if k.endswith((".up", ".gate"))}
                if not any(n for k, n in changes.items() if k.endswith(".up")):
                    raise RuntimeError(f"{tag}: all zero-initialized up matrices still zero after two updates")
                with (a.log_dir / "saved-adapters.jsonl").open("a") as saved:
                    print(json.dumps(dict(tag=tag, adapter=str(adapter), nonzero=changes,
                                          sha256=hashlib.sha256((adapter / "adapter.safetensors")
                                                                .read_bytes()).hexdigest())), file=saved)
                # Every saved adapter is opened in a fresh process; the final adapter
                # also covers any additional distinct extrema in the held-out split.
                if mode == "eva":
                    indices = range(len(devs))
                elif i == len(train) - 1:
                    indices = range(min(i, len(devs) - 1), len(devs))
                else:
                    indices = [min(i, len(devs) - 1)]
                for j in indices:
                    entry = (["-m", "experiments.preflight.run_asr_smoke", "evaluate-eva"]
                             if mode == "eva" else ["-m", "examples.omni_talora.evaluate"])
                    run(f"{tag}-dev{j}", entry,
                        ["--adapter", str(adapter), "--max-new-tokens", str(GENERATION_LIMIT)], devs[j])


if __name__ == "__main__":
    if sys.argv[1:2] == ["evaluate-eva"]:
        guarded_evaluation()
    else:
        main()
