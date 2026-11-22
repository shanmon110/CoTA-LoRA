"""Shared LibriSpeech split identities, adapter methods and result helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPARISON = Path(__file__).resolve().parent / "conf/comparison.json"
PLAN = ROOT / "experiments/preflight/asr-plan"
MANIFEST_HASHES = {
    "train": "e33bd6e2b6c75cbcd6706fb95949843624f7d26dfd14c723763a5d61c90c8411",
    "dev": "eea0acf0f504eb8ca2a095591d2887b6b9653fc6a4a45ae389de5bca3a003b04",
}
METHODS = {
    "o-lora": ["--mode", "lora"],
    "ta-lora": ["--mode", "ta"],
    "qv-lora": ["--mode", "lora", "--target-set", "qv", "--match-ta-budget"],
    "toplora": ["--mode", "toplora"],
    "bora": ["--mode", "bora", "--bora-blocks", "2"],
    "dora": ["--mode", "dora"],
    "eva": ["--mode", "eva", "--eva-samples", "128", "--eva-seed", "42",
            "--eva-rho", "2", "--eva-token-mask", "labels"],
}


def training_strategy(model_config, comparison_path, comparison, override):
    strategy = json.loads((comparison_path.parent / comparison["training_config"]).read_text())
    strategy.update(model_config.get("training_args", {}))
    if override:
        strategy.update(json.loads(override.read_text()))
    return strategy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, result):
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    stage.replace(path)


def batches(events, stage):
    return [s["id"] for e in events if e["stage"] == stage and e["event"] == "batch"
            for s in e["samples"]]
