"""Record and verify the stages of the single-seed WEST LibriSpeech comparison."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil

from examples.librispeech.asr.common import COMPARISON, MANIFEST_HASHES, ROOT, batches, save, sha, training_strategy
from examples.librispeech.asr.local.prepare_data import prepare_splits, read_rows
from west_adapters import resolve_lora_config
from examples.omni_talora.evaluate import _words

GENERATION = Path(__file__).resolve().parents[1] / "conf/generation_config.json"
TEST_SPLITS = ("test-clean", "test-other")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_comparison(conf, comparison, adapter_override, training_override):
    spec = json.loads(comparison.read_text())
    training = training_strategy(conf, comparison, spec, training_override)
    paths = [adapter_override] if adapter_override else [comparison.parent / item for item in spec["adapters"]]
    methods = {}
    for path in paths:
        plugin = json.loads(path.read_text())
        resolve_lora_config(plugin)
        name = plugin["method"]
        require(name not in methods, f"Duplicate comparison method: {name}")
        methods[name] = plugin
    require(methods, "Select at least one adapter in the comparison config")
    return training, methods


def init(run, config, data_dir, generation=GENERATION, comparison=COMPARISON, adapter=None, training=None):
    conf = json.loads(config.read_text())
    require(conf.get("model_type") == "qwen2_5_omni_adapter", "Expected a WEST Omni adapter model config")
    source_config_sha = sha(config)
    conf["model"] = str((ROOT / conf["model"]).resolve(strict=True))
    if data_dir == "-":
        plan = ROOT / conf["plan"]
        test_dir = ROOT / conf["test_manifests"]
        expected_hashes = MANIFEST_HASHES
    else:
        plan = Path(data_dir) / "plan"
        test_dir = Path(data_dir) / "manifests"
        splits = json.loads((plan / "summary.json").read_text())["splits"]
        expected_hashes = {k: splits[k]["filtered_sha256"] for k in ("train", "dev")}
    strategy, methods = read_comparison(conf, comparison, adapter, training)
    batch, eval_batch = strategy["per_device_train_batch_size"], strategy["per_device_eval_batch_size"]
    accum = strategy["gradient_accumulation_steps"]
    interval = strategy["eval_steps"]
    for value in (batch, eval_batch, accum, interval, strategy["dev_patience"]):
        require(type(value) is int and value > 0, "Expected positive integer batch, accumulation and validation settings")
    paths, source_hashes, ids, limits = prepare_splits(plan, test_dir, expected_hashes, max(batch, eval_batch))
    decode = json.loads(generation.read_text())
    require(not (set(decode) - {"do_sample", "num_beams", "max_new_tokens"})
            and decode.get("do_sample", False) is False and decode.get("num_beams", 1) == 1
            and type(decode.get("max_new_tokens")) is int and decode["max_new_tokens"] > 0,
            "Expected shared greedy ASR decoding with a positive token limit")
    if "max_new_tokens" in strategy:
        require(strategy["max_new_tokens"] == decode["max_new_tokens"], "Dev and test generation limits differ")
    strategy["max_new_tokens"] = decode["max_new_tokens"]
    for key, value in limits.items():
        strategy.setdefault(key, value)
    require(all(strategy[key] >= v for key, v in limits.items()), "Input limits exclude planned examples")
    run.mkdir(parents=True, exist_ok=False)
    (run / "data").mkdir()
    save(run / "model_config.json", conf)
    shutil.copyfile(generation, run / "generation_config.json")
    save(run / "training_config.json", strategy)
    (run / "adapters").mkdir()
    adapter_hashes = {}
    for name, config_data in methods.items():
        snapshot = run / "adapters" / (name + ".json")
        save(snapshot, config_data)
        adapter_hashes[name] = sha(snapshot)
    hashes = {}
    for split, path in paths.items():
        output = run / "data" / (split + ".jsonl")
        with output.open("x") as dst:
            for r in read_rows(path):
                wav = (path.parent / r["audio"]).resolve(strict=True)
                print(json.dumps(dict(id=r["id"], speaker=r["speaker"], wav=str(wav), txt=r["text"]),
                                 ensure_ascii=False), file=dst)
        require(sha(path) == source_hashes[split], f"{split} changed during manifest conversion")
        hashes[split] = sha(output)
    steps = strategy.get("max_steps", -1)
    if steps <= 0:
        steps = math.ceil(math.ceil(len(ids["train"]) / (batch * accum)) * strategy["num_train_epochs"])
    require(steps > 0, "Training schedule must include at least one update")
    state = dict(state="ready", phase=None, method=None, results={}, methods=list(methods),
                 adapters=methods, adapter_sha256=adapter_hashes, training_sha256=sha(run / "training_config.json"),
                 base_sha256=conf["base_sha256"], config_sha256=sha(run / "model_config.json"),
                 source_config_sha256=source_config_sha, generation_sha256=sha(run / "generation_config.json"),
                 manifest_sha256=hashes, source_manifest_sha256=source_hashes,
                 split_counts={k: len(v) for k, v in ids.items()},
                 input_budget={key: strategy[key] for key in limits},
                 seed=strategy["seed"], batch_size=batch, eval_batch_size=eval_batch, accumulation_steps=accum,
                 epochs=strategy["num_train_epochs"],
                 max_steps=steps, dev_interval=interval, patience=strategy["dev_patience"], max_new_tokens=decode["max_new_tokens"],
                 candidates=list(range(interval, steps, interval)) + [steps])
    save(run / "status.json", state)


def manifest(run, state, split):
    path = run / "data" / (split + ".jsonl")
    require(sha(path) == state["manifest_sha256"][split], f"{split} manifest changed")
    return path


def start(run, state, method):
    remaining = [m for m in state["methods"] if m not in state["results"]]
    require(state["state"] == "ready" and remaining and remaining[0] == method,
            f"Unexpected method or training already started: {method}")
    for split in ("train", "dev"):
        manifest(run, state, split)
    require(sha(run / "model_config.json") == state["config_sha256"], "Model config changed")
    require(sha(run / "training_config.json") == state["training_sha256"] and
            sha(run / "adapters" / (method + ".json")) == state["adapter_sha256"][method],
            "Training or adapter config changed")
    (run / method).mkdir()
    state.update(state="running", method=method, phase="train")


def trace(path, stage):
    events = read_rows(path)
    ends = [e for e in events if e.get("stage") == stage and e.get("event") == "stage_end"]
    require(ends and all(e["status"] == "ok" for e in ends), f"Incomplete {stage}: {path}")
    return batches(events, stage)


def validate_score(score, expected, state):
    words = sum(len(_words(r["txt"])) for r in expected)
    require(words > 0 and score["reference_words"] == words and score["utterances"] == len(expected)
            and score["max_new_tokens"] == state["max_new_tokens"]
            and math.isclose(score["wer"], score["word_errors"] / words),
            "ASR score differs from the expected evaluation protocol")


def finish(run, state, method):
    from torch import Generator
    from torch.utils.data import DataLoader

    require(state["state"] == "running" and state["method"] == method, "Unexpected training completion")
    d = run / method
    records = [json.loads(line) for line in (d / "train.log").read_text().splitlines()
               if line.startswith("{")]
    summary, = [r for r in records if "trainable_parameters" in r]
    updates = [r for r in records if "loss" in r and "step" in r]
    n = len(updates)
    require(n in state["candidates"] and [r["step"] for r in updates] == list(range(1, n + 1))
            and all(math.isfinite(r["loss"]) for r in updates), "Missing or nonfinite training updates")
    plugin = resolve_lora_config(state["adapters"][method])
    rank = plugin["rank"]
    if plugin.get("match_ta_budget"):
        match, = [row for row in records if "qv_rank" in row]
        unit = match["qv_parameters"] // match["qv_rank"]
        lo = max(1, match["ta_parameters"] // unit)
        rank = min((lo, lo + 1), key=lambda r: abs(r * unit - match["ta_parameters"]))
    require(summary["resolved_rank"] == rank and math.isclose(summary["effective_scale"], plugin["alpha"] / rank),
            "Adapter rank or scale differs from its configuration")
    train = read_rows(manifest(run, state, "train"))
    batch, accum = state["batch_size"], state["accumulation_steps"]
    loader = DataLoader(range(len(train)), batch_size=batch, shuffle=True,
                        generator=Generator().manual_seed(state["seed"]))
    pending = []
    updates_left = n
    per_epoch = math.ceil(len(loader) / accum)
    while updates_left:
        epoch_order = [train[i]["id"] for indices in loader for i in indices.tolist()]
        used = min(per_epoch, updates_left)
        pending.extend(epoch_order[:used * batch * accum])
        updates_left -= used
    order = trace(d / "train.jsonl", "train")
    require(order == pending, "Training order or number of samples changed")
    dev = read_rows(manifest(run, state, "dev"))
    dev_ids = [r["id"] for r in dev]
    adapter = d / "adapter"
    history = read_rows(adapter / "dev-history.jsonl")
    require([r["validation_step"] for r in history] == [i for i in state["candidates"] if i <= n],
            "Missing development-set checks")
    best, best_errors, misses = None, math.inf, 0
    scores = {}
    for record in history:
        step = record["validation_step"]
        score = record["score"]
        validate_score(score, dev, state)
        require(trace(adapter / f"dev-{step}.jsonl", "evaluate") == dev_ids, "Incomplete dev decode")
        improved = score["word_errors"] < best_errors
        if improved:
            best, best_errors = step, score["word_errors"]
        misses = 0 if improved else misses + 1
        stopped = misses >= state["patience"] and step < state["max_steps"]
        require(record["best_step"] == best and record["early_stopped"] == stopped
                and (not stopped or step == n), "Early-stopping decision differs from dev scores")
        scores[str(step)] = score
    require(n == state["max_steps"] or history[-1]["early_stopped"], "Unexpected end of training")
    selected = adapter / f"step-{best}"
    selected_config = json.loads((selected / "adapter_config.json").read_text())
    require((selected_config["mode"], selected_config.get("target_set", "o"), selected_config["last_n"],
             selected_config["rank"], selected_config["alpha"]) ==
            (plugin["mode"], plugin["target_set"], plugin["last_n"], rank, plugin["alpha"]),
            "Selected checkpoint differs from the configured method or layout")
    selection = dict(selected_step=best, selected_adapter=str(selected),
                     weights_sha256=sha(selected / "adapter.safetensors"),
                     config_sha256=sha(selected / "adapter_config.json"))
    state["results"][method] = dict(train=summary, trained_steps=n, trained_examples=len(order),
                                    dev=scores, early_stopped=history[-1]["early_stopped"], selection=selection)
    state.update(state="ready", phase=None, method=None)


def seal(run, state):
    require(state["state"] == "ready" and set(state["results"]) == set(state["methods"]),
            "All configured dev selections are required before testing")
    require(sha(run / "generation_config.json") == state["generation_sha256"], "Generation config changed")
    choices = {m: state["results"][m]["selection"] for m in state["methods"]}
    path = run / "dev-selection.json"
    with path.open("x") as f:
        json.dump(choices, f, indent=2)
    state.update(state="trained", selection_sha256=sha(path))


def checkpoint(run, state, method, split):
    require(state["state"] in ("trained", "testing", "completed"), "Testing requires sealed dev selections")
    manifest(run, state, split)
    require(sha(run / "generation_config.json") == state["generation_sha256"], "Generation config changed")
    sealed = run / "dev-selection.json"
    require(sha(sealed) == state["selection_sha256"], "Sealed development-set selection changed")
    choice = json.loads(sealed.read_text())[method]
    path = Path(choice["selected_adapter"])
    require(sha(path / "adapter.safetensors") == choice["weights_sha256"]
            and sha(path / "adapter_config.json") == choice["config_sha256"], "Selected adapter changed")
    return path


def score_test(run, state, method, split):
    checkpoint(run, state, method, split)
    d = run / method
    expected = read_rows(manifest(run, state, split))
    ids = [r["id"] for r in expected]
    score = json.loads((d / f"{split}.json").read_text())
    validate_score(score, expected, state)
    require(trace(d / f"{split}.jsonl", "evaluate") == ids, "Incomplete test decode")
    for kind in ("hyp", "ref"):
        rows = read_rows(d / f"{split}.{kind}.jsonl")
        require([r["id"] for r in rows] == ids, f"Misaligned test {kind}")
    overall = re.search(r"^Overall ->.*N=(\d+) C=\d+ S=(\d+) D=(\d+) I=(\d+)\s*$",
                        (d / f"{split}.wer").read_text(), flags=re.M)
    require(overall is not None, "Missing WEST word error totals")
    words, sub, deleted, inserted = map(int, overall.groups())
    require((words, sub + deleted + inserted) == (score["reference_words"], score["word_errors"]),
            "WEST and ASR decoder word error counts disagree")
    state["results"][method].setdefault("test", {})[split] = score
    state.update(state="testing", phase="test", method=method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("init", "methods", "eval-batch", "budget", "start", "finish", "seal", "checkpoint", "score", "complete", "fail"):
        cmd = subs.add_parser(name)
        cmd.add_argument("run", type=Path)
        if name == "init":
            cmd.add_argument("config", type=Path)
            cmd.add_argument("data_dir")
            cmd.add_argument("generation", type=Path, nargs="?", default=GENERATION)
            cmd.add_argument("--comparison", type=Path, default=COMPARISON)
            cmd.add_argument("--adapter", type=Path)
            cmd.add_argument("--training", type=Path)
        if name in ("start", "finish", "checkpoint", "score"):
            cmd.add_argument("method")
        if name in ("checkpoint", "score"):
            cmd.add_argument("split", choices=TEST_SPLITS)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "init":
        init(run, args.config, args.data_dir, args.generation, args.comparison, args.adapter, args.training)
        return
    with (run / ".protocol.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        dispatch(run, args)


def dispatch(run, args):
    path = run / "status.json"
    state = json.loads(path.read_text())
    if hasattr(args, "method"):
        require(args.method in state["methods"], f"Unknown method in this experiment: {args.method}")
    if args.action == "methods":
        print(*state["methods"])
        return
    if args.action == "eval-batch":
        print(state.get("eval_batch_size", state["batch_size"]))
        return
    if args.action == "budget":
        print(*(state["input_budget"][k] for k in ("max_audio_seconds", "max_batch_frames", "max_sequence_length")))
        return
    if args.action == "checkpoint":
        selected = checkpoint(run, state, args.method, args.split)
        require(state["state"] in ("trained", "testing") and state.get("phase") != "decode"
                and args.split not in state["results"][args.method].get("test", {}),
                "Decode already claimed or completed")
        state.update(state="testing", phase="decode", method=args.method, split=args.split)
        save(path, state)
        print(selected)
        return
    if args.action == "fail":
        if state["state"] in ("trained", "completed"):
            return
        state["state"] = "failed"
    elif args.action == "start":
        start(run, state, args.method)
    elif args.action == "finish":
        finish(run, state, args.method)
    elif args.action == "seal":
        seal(run, state)
    elif args.action == "score":
        require((state["state"], state["phase"], state["method"], state["split"]) ==
                ("testing", "decode", args.method, args.split), "No matching active decode")
        score_test(run, state, args.method, args.split)
    elif args.action == "complete":
        require(state["state"] == "testing" and all(set(state["results"][m].get("test", {})) == set(TEST_SPLITS)
                                                    for m in state["methods"]), "Both test splits must finish for every method")
        state.update(state="completed", phase=None, method=None, split=None)
    save(path, state)


if __name__ == "__main__":
    main()
