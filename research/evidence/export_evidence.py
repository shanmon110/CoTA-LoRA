#!/usr/bin/env python3
"""Create this allowlisted, aggregate-only evidence archive from local runs.

Sources are read-only. The script writes only beside itself. It requires the
original local run directories, and is not a training/evaluation entry point.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

DEST = Path(__file__).resolve().parent
ROOTS = {
    "ta-conditional-01": Path("/root/autodl-tmp/MMAR/runs/ta-conditional-01"),
    "ta-conditional-transfer-01": Path("/root/autodl-tmp/MMAR/runs/ta-conditional-transfer-01"),
    "papers28-transfer-01": Path("/root/autodl-tmp/MMAR/runs/papers28-transfer-01"),
    "ordinary-lora-pilot": Path("/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/lora"),
}
entries = []
omissions = []
failures = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def secret_check(raw, name):
    text = raw.decode("utf-8")
    patterns = [r"hf_[A-Za-z0-9]{20,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                r"github_pat_[A-Za-z0-9_]{20,}", r"sk-[A-Za-z0-9_-]{20,}",
                r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"]
    if any(re.search(p, text) for p in patterns):
        raise ValueError(f"credential-shaped value detected in {name}; value suppressed")


def raw_copy(group, rel):
    source = ROOTS[group] / rel
    if not source.is_file():
        failures.append({"source": str(source), "reason": "missing allowlisted source"})
        return
    data = source.read_bytes()
    secret_check(data, str(source))
    target = DEST / "raw" / group / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    entries.append({"path": str(target.relative_to(DEST)), "kind": "byte_copy",
                    "source": str(source), "source_bytes": len(data),
                    "source_sha256": digest(data), "bytes": len(data), "sha256": digest(data)})


def derived(group, rel, target_rel, value, excluded, note):
    source = ROOTS[group] / rel
    data = source.read_bytes()
    output = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    secret_check(output, target_rel)
    target = DEST / "derived" / group / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(output)
    entries.append({"path": str(target.relative_to(DEST)), "kind": "derived_aggregate",
                    "source": str(source), "source_bytes": len(data),
                    "source_sha256": digest(data), "bytes": len(output), "sha256": digest(output),
                    "excluded_fields": excluded, "transformation": note})


def read(group, rel):
    return json.loads((ROOTS[group] / rel).read_text())


TOP = {
    "ta-conditional-01": ["REPORT.md", "protocol.json", "provenance.json", "provenance-phase-one.json",
        "phase-two.json", "phase-three.json", "phase-two-trigger.json", "phase-three-trigger.json",
        "gate-lr-followup.json", "fastgate-optimizer-audit.json", "status.json",
        "pair-selection-green.log", "pair-selection-red.log"],
    "ta-conditional-transfer-01": ["REPORT.md", "protocol.json", "status.json", "mmau_scores.json",
        "mmau_overlap_audit.json", "ser-verified.json", "ser-same-words.json",
        "verification.log", "mmau_official_scoring.log"],
    "papers28-transfer-01": ["REPORT.md", "SPECIAL_PROTOCOLS.md", "protocol.json", "scores.json",
        "disposition.json", "paired-target-scores.json", "split-audio-audit.json"],
    "ordinary-lora-pilot": ["protocol.json", "result.json", "logging.jsonl"],
}

for group, rels in TOP.items():
    for rel in rels:
        raw_copy(group, rel)
    root = ROOTS[group]
    for dirname in ["configs", "results"]:
        for path in sorted((root / dirname).glob("*.json")):
            raw_copy(group, path.relative_to(root))
    # Aggregate evaluation metrics and original per-step scalar training logs.
    for path in sorted(root.rglob("metrics.json")):
        rel = path.relative_to(root)
        if rel.parts[0] not in {"sources", "data", "mini-swift-speech"}:
            raw_copy(group, rel)
    for arm in sorted((root / "train").glob("*")):
        if not arm.is_dir():
            continue
        for name in ["protocol.json", "result.json", "logging.jsonl", "loraplus-optimizer.json", "generation.json"]:
            if (arm / name).is_file():
                raw_copy(group, (arm / name).relative_to(root))
    # Initialization contains a first-example supervision record; remove it.
    init_paths = sorted((root / "train").glob("*/initialization.json"))
    if group == "ordinary-lora-pilot":
        init_paths.append(root / "initialization.json")
    for path in init_paths:
        rel = path.relative_to(root)
        value = read(group, rel)
        removed = [k for k in ["supervision"] if k in value]
        for key in removed:
            del value[key]
        derived(group, rel, rel, value, removed,
                "Remove first-example supervision metadata; preserve all other initialization audit fields.")
    # Regression/failure logs are kept; stdout training/sample dumps are omitted.
    for path in sorted((root / "logs").glob("*")):
        if not path.is_file() or path.stat().st_size > 20000:
            continue
        name = path.name
        keep = (name.startswith(("cpu-", "final-replay-", "gate-lr-", "qv-regression-",
                                  "protocol-cpu", "known-target-test", "known-target-publication-",
                                  "source-snapshot-", "randlora-resume-"))
                or name in {"conditional-suite.log", "verification-final.log", "paper-middle-cpu.log",
                            "paper-late-cpu.log", "paper-completion-cpu.log", "lowra-cuda.log",
                            "lowra-cpu-openmp.log", "lowra-3b-cpu-reload.log", "middle-author-upstream-parity.log",
                            "middle-swift-omni-audio.log", "special-report-regression.log",
                            "split-audio-audit.log", "original-disposition-f546c3dd93793ac1.json"})
        if keep:
            raw_copy(group, path.relative_to(root))

# Gate probes: only layer-level or domain/pair-kind aggregate diagnostics.
for rel in ["gate-probe.json", "gate-probe-fastgate.json"]:
    value = read("ta-conditional-01", rel)
    for key in ["raw", "paired"]:
        value.pop(key, None)
    value["probe"].pop("selected", None)
    derived("ta-conditional-01", rel, rel, value, ["raw", "paired", "probe.selected"],
            "Remove per-example layer traces, per-pair identifiers, and selected sample metadata; retain aggregates.")

# analysis.json consists of aggregate statistics. Preserve original bytes and
# readable field-level JSONs; its 1.1 MB size is a documented size exception.
group = "ta-conditional-01"
source = ROOTS[group] / "analysis.json"
data = source.read_bytes()
secret_check(data, str(source))
raw_copy(group, "analysis.json")
value = json.loads(data)
for key, part in value.items():
    derived(group, "analysis.json", f"analysis/{key}.json", {key: part}, [],
            f"Exact JSON value for top-level field {key!r}; formatting changed, no field removed.")

# SER audit: no sample names, per-file hashes, text prompts, annotations, or
# dialogue identifiers. The score files already contain confusion matrices.
group = "ta-conditional-transfer-01"
value = read(group, "ser-audit.json")
excluded = ["speakers", "original_labels", "audio_checks", "prior_session5_supervised"]
value = {k: v for k, v in value.items() if k not in excluded}
derived(group, "ser-audit.json", "ser-audit-summary.json", value, excluded,
        "Keep dataset-level counts, durations, protocol labels and overlap aggregates; remove sample-level evidence.")
value = read(group, "ser-processor-audit.json")
excluded = ["longest_prompt", "longest_audio"]
value = {k: v for k, v in value.items() if k not in excluded}
derived(group, "ser-processor-audit.json", "ser-processor-audit-summary.json", value, excluded,
        "Remove individual longest-example IDs and associated example metadata; retain processor audit counts.")

group = "papers28-transfer-01"
for rel in ["theory-01/result.json", "theory-01/fnn.log", "theory-01/tfn.log",
            "theory-01/omni-control/construction.json", "theory-01/omni-control/native-reload.json",
            "theory-01/omni-control/result.json", "lowra-base/protocol.json"]:
    raw_copy(group, rel)
snapshot = "sources/83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a"
for name in ["environment.json", "capture.json"]:
    raw_copy(group, f"{snapshot}/{name}")

# Browsable ordinary-LoRA training curve from the scalar log, without examples.
group = "ordinary-lora-pilot"
rows = [json.loads(line) for line in (ROOTS[group] / "logging.jsonl").read_text().splitlines() if line]
train_rows = [row for row in rows if "loss" in row]
curve = []
for row in rows:
    if "eval_accuracy" not in row:
        continue
    step = int(row["global_step/max_steps"].split("/")[0])
    window = [record["loss"] for record in train_rows
              if step - 32 < int(record["global_step/max_steps"].split("/")[0]) <= step]
    curve.append({"step": step, "mean_loss_previous_32_steps": statistics.mean(window),
                  "source_dev_accuracy": row["eval_accuracy"],
                  "source_dev_correct_of_256": round(row["eval_accuracy"] * 256)})
derived(group, "logging.jsonl", "training-curve-summary.json",
        {"points": curve, "notes": ["Training loss is averaged from logged scalar values over the preceding 32 updates.",
          "AVQA development accuracy only; no target-set trajectory or development cross-entropy was logged."]},
        ["per_update_scalars", "duplicate_final_log_history"],
        "Group scalar losses into the 32-update windows ending at each development evaluation; preserve dev accuracy.")

# Explicitly record exclusions without enumerating individual dataset filenames.
omissions = [
    {"patterns": ["*.safetensors", "*.pt", "*.bin", "optimizer*", "scheduler*", "*.wav", "*.mp3"],
     "reason": "Model/optimizer weights and media are outside the public evidence export."},
    {"patterns": ["data/**", "predictions.jsonl", "hypotheses.jsonl", "official-input.json*", "pairs.json",
                  "client-partitions.json", "mini-swift-speech/**", "lowra-base/index.json"],
     "reason": "Exclude dataset contents, original sample lists, option text, per-sample metadata and prediction records."},
    {"patterns": ["args.json", "logs/train-*.log", "logs/mini-*.log", "driver*.log"],
     "reason": "Bulky argument/stdout dumps may embed credentials, templates or example supervision; publish explicit configs/protocols/scalar logging.jsonl instead."},
    {"patterns": ["gate-probe.json: raw, paired, probe.selected", "gate-probe-fastgate.json: raw, paired, probe.selected",
                  "initialization.json: supervision", "ser-audit.json: speakers, original_labels, audio_checks, prior_session5_supervised",
                  "ser-processor-audit.json: longest_prompt, longest_audio"],
     "reason": "Raw files are not published; aggregate derivatives and original source SHA-256 are recorded."},
    {"patterns": ["sources/** except selected environment.json and capture.json", "*.py", "*.sh", "*-partial.*"],
     "reason": "This evidence subset does not duplicate source archives, launchers or superseded partial summaries; selected failure logs remain."},
]

entries.sort(key=lambda item: item["path"])
manifest = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Public aggregate evidence subset. Original experimental files are read-only.",
    "local_absolute_paths": "Historical provenance only; use source_roots to map published counterparts. Paths do not make external data/weights available.",
    "source_roots": {k: str(v) for k, v in ROOTS.items()},
    "statistics": {"files": len(entries), "bytes": sum(item["bytes"] for item in entries),
                   "byte_copies": sum(item["kind"] == "byte_copy" for item in entries),
                   "derived_aggregates": sum(item["kind"] == "derived_aggregate" for item in entries),
                   "files_over_1000000_bytes": [item["path"] for item in entries if item["bytes"] > 1000000]},
    "files": entries, "exclusions": omissions, "failures": failures,
}
(DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest["statistics"], indent=2))
print("failures:", len(failures))
