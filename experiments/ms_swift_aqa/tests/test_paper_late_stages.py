"""Exercise the source-preparation boundary with an actual miniature Omni."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from experiments.ms_swift_aqa import paper_late
from experiments.ms_swift_aqa.paper_late_mixer import set_batch_mask
from experiments.ms_swift_aqa.paper_plugin import tuners_map
from experiments.ms_swift_librispeech.tests.test_adapter_pair import batch, tiny_omni


class SourceDataset:
    def __init__(self, rows):
        self.dataset = list(reversed(rows))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return i


@pytest.mark.parametrize("method", ["ralora", "igu-lora", "lora-mixer"])
def test_calibration_or_expert_stage_runs_and_restores_source_only_model(method, tmp_path):
    torch.manual_seed(31)
    model = tiny_omni()
    model.thinker.config.audio_token_id = 5
    clean = deepcopy(model)
    source = tmp_path/"train.jsonl"
    domains = ["Which", "Come From", "Happening", "Where"]
    rows = [{"audios":[f"audio-{i}.wav"],
             "messages":[{"role":"user","content":f"question-{i}"},{"role":"assistant","content":"option"}],
             "source_id":str(i), "question_type":domain}
            for i, domain in enumerate(domains*2+["Why"])]
    rows[-1]["source_id"] = rows[0]["source_id"]
    source.write_text("".join(json.dumps(row)+"\n" for row in rows))
    (tmp_path/"protocol.json").write_text(json.dumps({"train_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}))
    opts_file = Path(paper_late.__file__).parent/"papers28"/"configs"/f"{method}.json"
    settings = json.loads(opts_file.read_text())["settings"]
    settings.update(calibration_steps=1, calibration_batch_size=1, stage_accumulation=1,
                    pruning_steps=3, prune_interval=2, prune_warmup=1,
                    examples_per_expert=1, router_hidden=8, source_manifest=str(source))
    if method == "igu-lora":
        settings["stage_accumulation"] = 2
    config = {"model_revision":"tiny-omni", "method_options":{method:settings}}
    tuner = tuners_map["paper_" + method]
    options = SimpleNamespace(target_modules=r"thinker\.model\.layers\.4\.self_attn\.[qvo]_proj",
                              lora_rank=4, lora_alpha=4, model_kwargs={"paper_settings":settings})
    tuner.prepare_model(options, model)
    model.eval()
    stages = []
    pruning_regularizers = []

    def collate(indices):
        b = batch()
        b["input_features"] = torch.randn(1,8,8,generator=torch.Generator().manual_seed(indices[0])).bfloat16()
        b["feature_attention_mask"] = torch.ones(1,8,dtype=torch.long)
        return b

    def compute(m, inputs):
        if method == "lora-mixer":
            set_batch_mask(m, inputs["attention_mask"])
        if method == "igu-lora":
            from experiments.ms_swift_aqa.paper_late_igu import IGULinear, orthogonality
            usual = orthogonality(mod for mod in m.modules() if isinstance(mod,IGULinear))
            pruning_regularizers.append((paper_late.training_penalty(m,method,settings)/usual).item())
        loss = m(**inputs).loss
        return loss if method == "ralora" else loss+paper_late.training_penalty(m,method,settings)

    trainer = SimpleNamespace(model=model, train_dataset=SourceDataset(rows), data_collator=collate,
        args=SimpleNamespace(seed=42, data_seed=42, learning_rate=0.002, weight_decay=0.,
                             max_grad_norm=1., output_dir=str(tmp_path)),
        template=SimpleNamespace(register_post_encode_hook=lambda _:stages.append("on"),
                                 remove_post_encode_hook=lambda :stages.append("off"),
                                 forward_context=lambda *_:nullcontext()),
        _prepare_inputs=lambda x:x, compute_loss_context_manager=nullcontext, compute_loss=compute)
    result = paper_late.initialize_training(trainer, method, config)
    assert stages == ["on","off"] and not model.training
    assert all(p.grad is None for p in model.parameters())
    for path in model._paper_adapter_spec["targets"]:
        module = model.get_submodule(path)
        torch.testing.assert_close(module.base_layer.weight, clean.get_submodule(path).weight, rtol=0, atol=0)
    if method == "ralora":
        assert result["calibration_steps"] == 1 and len(result["allocation"]) == 3
        assert all(info["effective_gradient_rank"] > 0 for info in result["allocation"].values())
    elif method == "igu-lora":
        assert result["pruning_optimizer_steps"] == 3 and len(result["pruning_train_indices"]) == 6
        assert pruning_regularizers == [2.]*6 and not hasattr(model,"_igu_pruning_accumulation")
        assert 0 < result["final_active_parameters"] <= result["target_active_parameters"]
    else:
        library = Path(result["expert_library"])
        metadata = json.loads(library.read_text())
        assert metadata["excluded_raw_sources"] == ["0"]
        assert metadata["excluded_question_types"] == ["Why"]
        assert result["warmup_steps"] == 4
        for entry in metadata["experts"]:
            row, = [trainer.train_dataset.dataset[i] for i in entry["train_indices"]]
            assert row["question_type"] == entry["domain"] and row["source_id"] != "0"
            file = library.parent/entry["file"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"]
            factors = load_file(str(file))
            assert any(x.abs().sum() > 0 for name, x in factors.items() if name.endswith(".b"))
    folder = tmp_path/"checkpoint"
    tuner.save_pretrained(model, folder)
    with torch.inference_mode():
        tuner.from_pretrained(clean, folder)
        clean.eval()
        inputs = collate([1])
        torch.testing.assert_close(model(**inputs).logits, clean(**inputs).logits, atol=0, rtol=0)
