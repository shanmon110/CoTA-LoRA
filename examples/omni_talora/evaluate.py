"""Evaluate one adapter on its recorded task; run as a module from the WEST root."""

import argparse
import json
import re
import unicodedata
from dataclasses import asdict

from .budget import Budget, add_budget_args


CHOICES = "ABCD"
NORMALIZATION = (
    "NFKC + casefold; keep letters/digits and internal apostrophes, "
    "normalize curly apostrophes; other characters separate words; no number expansion"
)


def _words(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def _edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        previous, row = row, [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j-1] + (word != other)))
    return row[-1]


def _check_pairs(references, predictions):
    if len(references) == 0 or len(references) != len(predictions):
        raise ValueError("references and predictions must have the same nonzero length")


def asr_metrics(references, predictions):
    """Corpus WER = sum of per-utterance word edit distances / total reference words."""
    _check_pairs(references, predictions)
    errors, total = 0, 0
    for reference, prediction in zip(references, predictions):
        ref = _words(reference)
        total += len(ref)
        errors += _edit_distance(ref, _words(prediction))
    if not total:
        raise ValueError("WER is undefined: zero normalized words in reference corpus")
    return dict(wer=errors / total, word_errors=errors, reference_words=total,
                utterances=len(references), normalization=NORMALIZATION)


def ser_metrics(references, predictions):
    """Four-class unweighted average recall (undefined if a class has no references)."""
    _check_pairs(references, predictions)
    counts = {choice: dict(support=0, correct=0) for choice in CHOICES}
    for reference, prediction in zip(references, predictions):
        if reference not in counts or prediction not in counts:
            raise ValueError(f"unknown SER class in {(reference, prediction)!r}")
        counts[reference]["support"] += 1
        counts[reference]["correct"] += int(reference == prediction)
    for stats in counts.values():
        n = stats["support"]
        stats["recall"] = stats["correct"] / n if n else None
    recalls = [s["recall"] for s in counts.values()]
    return dict(uar=None if None in recalls else sum(recalls) / len(recalls),
                accuracy=sum(s["correct"] for s in counts.values()) / len(references),
                utterances=len(references), per_class=counts)


def candidate_ids(tokenizer, prompt):
    """Require exactly one distinct answer token after the actual assistant header."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for choice in CHOICES:
        full = tokenizer.encode(prompt + choice, add_special_tokens=False)
        if full[:-1] != prefix or not full or full[-1] in tokenizer.all_special_ids:
            raise ValueError(f"SER choice {choice!r} is not one ordinary token after prompt")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("SER choices do not have distinct token IDs")
    return ids


def select_choices(logits, attention_mask, ids):
    """Pick the highest conditional probability among the four candidates only."""
    import torch

    positions = torch.arange(attention_mask.shape[1], device=logits.device)
    last = torch.where(attention_mask.bool(), positions, -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("cannot score an empty SER prompt")
    next_logits = logits[torch.arange(len(last), device=logits.device), last]
    scores = next_logits.gather(1, torch.tensor(ids, device=logits.device))
    return [CHOICES[index] for index in scores.argmax(dim=1).tolist()]


def decode_continuations(tokenizer, generated, padded_prompt_width):
    # GenerationMixin returns [padded input_ids | appended ids], even for short rows.
    return tokenizer.batch_decode(generated[:, padded_prompt_width:], skip_special_tokens=True)


def count_capped_continuations(generated, padded_prompt_width, max_new_tokens, eos_token_id):
    continuation = generated[:, padded_prompt_width:]
    if continuation.shape[1] != max_new_tokens:
        return 0
    return int(continuation.ne(eos_token_id).all(dim=1).sum().item())


def evaluate(thinker, processor, records, *, task, device, batch_size, max_new_tokens,
             smoke=None, budget=None, on_predictions=None):
    import torch

    from .attention import audio_context
    from .data import OmniCollator, answer_for
    from .runtime import move_inputs

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and generation limit must be positive")
    collator = OmniCollator(processor, task, budget=budget)
    thinker.eval()
    references, predictions = [], []
    capped = 0
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            inputs = smoke.collate(collator.prompts, rows) if smoke else collator.prompts(rows)
            batch = move_inputs(inputs, device, thinker.audio_tower.dtype)
            ids, mask = batch["input_ids"], batch["attention_mask"]
            with audio_context(ids, mask, thinker.config.audio_token_id):
                if task == "asr":
                    if not mask[:, -1].all():
                        raise ValueError("batched generation requires left-padded prompts")
                    # Appended cache positions are text, never fresh audio slots.
                    # Keep the assistant's end-of-turn token available to stop.
                    suppressed = set(thinker.generation_config.suppress_tokens or ())
                    suppressed.update(processor.tokenizer.all_special_ids)
                    suppressed.add(thinker.config.audio_token_id)
                    suppressed.discard(processor.tokenizer.eos_token_id)
                    generated = thinker.generate(
                        **batch, do_sample=False, num_beams=1, num_return_sequences=1,
                        use_cache=True, return_dict_in_generate=False,
                        max_new_tokens=max_new_tokens, eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id, suppress_tokens=sorted(suppressed),
                    )
                    capped += count_capped_continuations(
                        generated, ids.shape[1], max_new_tokens, processor.tokenizer.eos_token_id)
                    answers = decode_continuations(processor.tokenizer, generated, ids.shape[1])
                elif task == "ser":
                    choices = [candidate_ids(processor.tokenizer, collator.prompt_text(row))
                               for row in rows]
                    answers = select_choices(
                        thinker(**batch, use_cache=False, return_dict=True).logits, mask, choices)
                else:
                    raise ValueError(f"unknown task: {task!r}")
            predictions.extend(answers)
            references.extend(answer_for(row, task) for row in rows)
    if task == "asr":
        score = dict(**asr_metrics(references, predictions), max_new_tokens=max_new_tokens,
                    max_new_tokens_without_eos_count=capped,
                    max_new_tokens_without_eos_ratio=capped / len(predictions))
    else:
        score = ser_metrics(references, predictions)
    if on_predictions is not None:
        on_predictions(records, references, predictions)
    return score


def evaluate_checkpoint(adapter, manifest, *, device=None, batch_size=1, max_new_tokens=128,
                        smoke_log=None, budget=None, on_predictions=None):
    import torch
    from .adapters import install_from_config
    from .checkpoint import load_adapter, read_config
    from .data import load_manifest
    from .runtime import load_base, local_snapshot_sha256
    from .diagnostics import SmokeLog

    config = read_config(adapter)
    limits = asdict(budget or Budget())
    if "input_budget" in config:
        recorded = config["input_budget"]
        if any(v is not None and v != recorded.get(k) for k, v in limits.items()):
            raise ValueError("Evaluation input budget differs from the saved training budget")
        limits = recorded
    budget = Budget(**limits)
    records = load_manifest(manifest, config["task"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    smoke = SmokeLog(smoke_log, device)
    with smoke.stage("load"):
        if ("base_sha256" in config and
                local_snapshot_sha256(config["model_name_or_path"]) != config["base_sha256"]):
            raise ValueError("Local base or processor content differs from the training snapshot")
        thinker, processor = load_base(config["model_name_or_path"], config["revision"], device)
    with smoke.stage("adapter_setup"):
        install_from_config(thinker, config)
        if load_adapter(thinker, adapter) != config:
            raise ValueError("adapter config changed while loading")
    with smoke.stage("evaluate"):
        result = evaluate(thinker, processor, records, task=config["task"], device=device,
                          batch_size=batch_size, max_new_tokens=max_new_tokens, smoke=smoke,
                          budget=budget, on_predictions=on_predictions)
    return {"task": config["task"], **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="directory with adapter weights and config")
    parser.add_argument("--manifest", required=True, help="held-out JSONL manifest")
    parser.add_argument("--device", help="torch device (default: CUDA if available, else CPU)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="ASR greedy decode limit")
    parser.add_argument("--smoke-log", help="Append per-stage memory peaks and input lengths to JSONL.")
    add_budget_args(parser)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")

    result = evaluate_checkpoint(args.adapter, args.manifest, device=args.device, batch_size=args.batch_size,
                                 max_new_tokens=args.max_new_tokens, smoke_log=args.smoke_log,
                                 budget=Budget.from_args(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
