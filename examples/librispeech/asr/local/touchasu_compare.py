"""Continue an existing native WEST LoRA run with a serial TouchASU comparison."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from examples.librispeech.asr.common import ROOT, save, sha

RECIPE = ROOT / "examples/aishell/asr"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "conf/touchasu_comparison.json"
BASELINE = "peft-lora"


def read(path):
    return json.loads(path.read_text())


def verify_checkpoints(checkpoints, steps):
    states = [(checkpoints, steps[-1])]
    states.extend((checkpoints / f"checkpoint-{step}", step) for step in steps)
    for state_dir, step in states:
        if read(state_dir / "trainer_state.json")["global_step"] != step:
            raise ValueError(f"Unexpected global_step for {state_dir}, expected {step}")


def baseline_completed(baseline, steps):
    exit_code = baseline / "exit.code"
    if not exit_code.exists():
        return False
    if exit_code.read_text().strip() != "0":
        raise RuntimeError(f"Baseline exit code {exit_code.read_text().strip()}. See {baseline}/train.log")
    verify_checkpoints(baseline / "checkpoints", steps)
    return True


def compare_scores(peft, ta):
    for key in ("generation_sha256", "manifest_sha256", "reference_words", "utterances"):
        if peft[key] != ta[key]:
            raise ValueError(f"Incomparable scores: {key} differs")
    for metric in (peft, ta):
        if not math.isclose(metric["wer"], metric["word_errors"] / metric["reference_words"]):
            raise ValueError("WER differs from the error and reference counts")
    diff = ta["word_errors"] - peft["word_errors"]
    return {"lower_test_wer": ("tie" if diff == 0 else "ta-lora" if diff < 0 else BASELINE),
            "ta_minus_peft_percentage_points": 100 * diff / peft["reference_words"]}


def initialize(output, baseline, data, config, pid):
    spec = read(config)
    inputs = read(baseline / "inputs.json")
    strategy = (config.parent / spec["training_config"]).resolve()
    if sha(strategy) != inputs["strategy_sha256"]:
        raise ValueError("Shared training profile differs from the live baseline")
    train = baseline / "data/train.jsonl"
    if sha(train) != inputs["manifest_sha256"]:
        raise ValueError("Baseline training manifest changed")
    training = read(strategy)
    steps = spec["selection_steps"]
    if (sorted(set(steps)) != steps or min(steps) <= 0 or steps[-1] != training["max_steps"]
            or any(s % training["save_steps"] for s in steps)):
        raise ValueError("Select common saved steps including the training endpoint")
    # Keep the same updates; only checkpoint writing frequency changes to fit a single data disk.
    training["save_steps"] = math.gcd(*steps)
    model = read(baseline / "touch_asu.json")
    if model["model_type"] != "touch_asu" or not model["lora_config"] or "method" in model["lora_config"]:
        raise ValueError("Expected a native PEFT LoRA predecessor")
    adapters = [read(config.parent / p) for p in spec["adapters"]]
    names = [a["method"] for a in adapters]
    if not names or names[0] != "ta-lora" or len(set(names)) != len(names):
        raise ValueError("TA-LoRA must be first and each configured method must be unique")
    model.update(lora_config=None, adapter_attention_backend=spec["adapter_attention_backend"])
    files = {"data/train.jsonl": train, "generation.json": config.parent / spec["generation_config"]}
    split_metadata = read(data / "splits.json")
    if split_metadata["train-clean-100"]["sha256"] != inputs["manifest_sha256"]:
        raise ValueError("Evaluation splits were prepared for another training manifest")
    manifest_hashes = {s: sha(data / f"{s}.jsonl") for s in ("dev-clean", "test-clean")}
    if any(split_metadata[s]["sha256"] != digest for s, digest in manifest_hashes.items()):
        raise ValueError("Evaluation manifests differ from split metadata")
    for key in ("deepspeed", "accelerator_config"):
        files[key + ".json"] = RECIPE / training[key]
        training[key] = str(output / (key + ".json"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "data").mkdir()
    (output / "adapters").mkdir()
    for name, source in files.items():
        (output / name).write_bytes(source.read_bytes())
    for name, adapter in zip(names, adapters):
        save(output / "adapters" / f"{name}.json", adapter)
    save(output / "training.json", training)
    save(output / "model.json", model)
    hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    plan = dict(baseline=str(baseline), baseline_pid=pid, data=str(data), steps=steps,
                methods=names, hashes=hashes, baseline_model_sha256=sha(baseline / "touch_asu.json"),
                manifests=manifest_hashes)
    save(output / "plan.json", plan)


def check_inputs(output, plan):
    baseline = Path(plan["baseline"])
    if (sha(baseline / "touch_asu.json") != plan["baseline_model_sha256"] or
            sha(baseline / "data/train.jsonl") != plan["hashes"]["data/train.jsonl"]):
        raise ValueError("Baseline inputs changed")
    for path, digest in plan["hashes"].items():
        if sha(output / path) != digest:
            raise ValueError(f"Comparison input changed: {path}")
    for split, digest in plan["manifests"].items():
        if sha(Path(plan["data"]) / f"{split}.jsonl") != digest:
            raise ValueError(f"Evaluation manifest changed: {split}")


def run_stage(output, state, arm, stage, command):
    directory = output / arm
    directory.mkdir(exist_ok=True)
    state.update(phase=stage, method=arm)
    save(output / "status.json", state)
    with (directory / (stage + ".log")).open("a") as log:
        print(json.dumps([str(s) for s in command]), file=log, flush=True)
        subprocess.run(command, cwd=RECIPE, stdout=log, stderr=subprocess.STDOUT, check=True)


def submit_arm(output, plan, state, name):
    check_inputs(output, plan)
    ckpts = output / name / "checkpoints"
    if not (ckpts / "trainer_state.json").exists():
        run_stage(output, state, name, "train", [
            "bash", "run.sh", "--stage", "train", "--model-conf", output / "model.json",
            "--training-config", output / "training.json", "--adapter-config", output / "adapters" / f"{name}.json",
            "--data", output / "data", "--dir", ckpts])
    verify_checkpoints(ckpts, plan["steps"])


def score_arm(output, plan, state, name, stage):
    check_inputs(output, plan)
    arm = output / name
    ckpts = (Path(plan["baseline"]) if name == BASELINE else arm) / "checkpoints"
    verify_checkpoints(ckpts, plan["steps"])
    args = ["--checkpoints", ckpts, "--steps", *map(str, plan["steps"]), "--data", plan["data"],
            "--output", arm / "eval", "--generation", output / "generation.json"]
    run_stage(output, state, name, stage, [
        sys.executable, "-m", "examples.librispeech.asr.local.touchasu_wer", stage, *args])
    if stage == "dev":
        return
    selection = read(arm / "eval/dev-selection.json")
    metric = read(arm / "eval/test-clean/metrics.json")
    if (metric["step"] != selection["selected_step"] or selection["selected_step"] not in plan["steps"]
            or metric["manifest_sha256"] != plan["manifests"]["test-clean"]
            or selection["manifest_sha256"] != plan["manifests"]["dev-clean"]
            or any(s["generation_sha256"] != plan["hashes"]["generation.json"] for s in (selection, metric))):
        raise ValueError(f"Evaluation does not match the selected checkpoint/protocol: {arm}")
    state["results"][name] = {"dev": selection["score"], "test": metric}
    save(output / "status.json", state)


def record_results(output, state):
    results = state["results"]
    pair = compare_scores(results[BASELINE]["test"], results["ta-lora"]["test"])
    report = {"ordinary_lora_vs_ta": pair, "methods": results}
    if "o-lora" in results:
        matched = compare_scores(results["o-lora"]["test"], results["ta-lora"]["test"])
        winner = matched["lower_test_wer"]
        report["matched_o_vs_ta"] = {
            "lower_test_wer": "o-lora" if winner == BASELINE else winner,
            "ta_minus_o_percentage_points": matched["ta_minus_peft_percentage_points"]}
    save(output / "comparison.json", report)


def execute(output, gpu_check=False):
    plan = read(output / "plan.json")
    state = {"phase": "waiting_for_baseline", "method": BASELINE, "results": {}}
    save(output / "status.json", state)
    try:
        baseline = Path(plan["baseline"])
        while not baseline_completed(baseline, plan["steps"]):
            pid = plan["baseline_pid"]
            if pid and not Path(f"/proc/{pid}").exists():
                time.sleep(3)  # Allow the supervising shell to publish exit.code.
                if not baseline_completed(baseline, plan["steps"]):
                    raise RuntimeError(f"Baseline PID {pid} exited without recording success")
                break
            time.sleep(30)
        if gpu_check:
            os.environ["WEST_TEST_FLASH_CUDA"] = "1"
            check = ROOT / "test/test_touch_asu_packed_attention.py"
            run_stage(output, state, "preflight", "gpu", [
                sys.executable, "-m", "pytest", "-q",
                f"{check}::test_flash_touchasu_packed_replay_and_decode[cuda]"])
        for name in plan["methods"]:
            submit_arm(output, plan, state, name)
            group = (BASELINE, name) if name == "ta-lora" else (name,)
            for stage in ("dev", "test"):
                for arm in group:
                    score_arm(output, plan, state, arm, stage)
            record_results(output, state)
        state.update(phase="complete", method=None)
    except BaseException as exc:
        state.update(failed_stage=state["phase"], phase="failed", error=str(exc))
        raise
    finally:
        save(output / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    init = subs.add_parser("init")
    init.add_argument("--baseline", type=Path, required=True)
    init.add_argument("--baseline-pid", type=int)
    init.add_argument("--data", type=Path, required=True)
    init.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = subs.add_parser("run")
    run.add_argument("--gpu-check", action="store_true", help="Exercise native BF16 packed TA on the released GPU")
    for sub in (init, run):
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "init":
        initialize(output, args.baseline.resolve(), args.data.resolve(), args.config.resolve(), args.baseline_pid)
    else:
        with (output / ".queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])
            execute(output, args.gpu_check)


if __name__ == "__main__":
    main()
