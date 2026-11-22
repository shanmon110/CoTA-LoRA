"""CPU tests: no pretrained weights, audio corpus or CUDA required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CONFIG = dict(model_name_or_path="local/model", revision=None, task="asr",
              rank=2, alpha=4.0, last_n=1, mode="ta")


class EvaluationTests(unittest.TestCase):
    def test_wer_counts_edits_across_utterances(self):
        from examples.omni_talora.evaluate import asr_metrics

        result = asr_metrics(["We’ll GO—now!", "OK"], ["we'll go", "bad extra"])
        self.assertEqual(result["reference_words"], 4)
        self.assertEqual(result["word_errors"], 3)
        self.assertEqual(result["wer"], 0.75)
        with self.assertRaises(ValueError):
            asr_metrics(["..."], ["hello"])
        with self.assertRaises(ValueError):
            asr_metrics(["hi"], [])

    def test_uar_gives_each_emotion_equal_weight(self):
        from examples.omni_talora.evaluate import ser_metrics

        result = ser_metrics(list("AAABCD"), list("AAABAA"))
        self.assertEqual(result["accuracy"], 4 / 6)
        self.assertEqual(result["uar"], 0.5)
        self.assertEqual(result["per_class"]["C"],
                         {"support": 1, "correct": 0, "recall": 0.0})
        self.assertIsNone(ser_metrics(["A"], ["A"])["uar"])

    def test_ser_rejects_ambiguous_or_retokenized_choices(self):
        from examples.omni_talora.evaluate import candidate_ids

        class Tokenizer:
            all_special_ids = []

            def encode(self, text, add_special_tokens):
                return list(map(ord, text))

        processor = Tokenizer()
        self.assertEqual(candidate_ids(processor, "x"), [65, 66, 67, 68])
        for invalid in ([121, 68], [120, 65], [120, 68, 10]):
            processor.encode = lambda text, **kw: (
                invalid if text == "xD" else list(map(ord, text)))
            with self.assertRaises(ValueError):
                candidate_ids(processor, "x")

    def test_ser_reads_only_next_token_at_last_nonpad(self):
        import torch
        from examples.omni_talora.evaluate import select_choices

        logits = torch.zeros(2, 4, 12)
        logits[:, :, 11] = 100  # best token outside the choices
        logits[0, 3, 6] = 5
        logits[1, 2, 2] = 3
        logits[1, 3, 4] = 10  # right padding, cannot answer from here
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(select_choices(logits, mask, [[1, 2, 6, 4]] * 2), ["C", "B"])

    def test_asr_removes_padded_prompt_width_before_decoding(self):
        import torch
        from examples.omni_talora.evaluate import decode_continuations

        class Tokenizer:
            def batch_decode(self, batch, skip_special_tokens):
                return [" ".join(str(x) for x in row.tolist()) for row in batch]

        generated = torch.tensor([[0, 0, 2, 7, 3], [2, 3, 4, 5, 9]])
        self.assertEqual(decode_continuations(Tokenizer(), generated, 3), ["7 3", "5 9"])

    def test_eval_uses_text_only_decode_and_closed_set_scoring(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        if __package__:
            from .test_omni_ta_integration import tiny_config
        else:  # pytest adds the test directory, not a package, to sys.path
            from test_omni_ta_integration import tiny_config
        from examples.omni_talora.attention import install_adapter
        from examples.omni_talora.data import OmniCollator
        from examples.omni_talora.evaluate import evaluate

        # Use the real Thinker/cache; replace only file/processor I/O and put
        # a known ordering on logits: audio placeholder > B > all others.
        model = Qwen2_5OmniThinkerForConditionalGeneration(tiny_config()).eval()
        install_adapter(model, rank=2, alpha=4, last_n=1, mode="ta")
        def controlled_logits(module, inputs, logits):
            result = torch.zeros_like(logits)
            result[..., 5], result[..., 9] = 100., 10.
            return result
        model.lm_head.register_forward_hook(controlled_logits)
        class Tokenizer:
            eos_token, eos_token_id, pad_token_id = "<end>", 2, 0
            all_special_ids = [0, 2, 5]
            def convert_tokens_to_ids(self, token):
                return 5
            def encode(self, text, **kwargs):
                return [8 + "ABCD".index(t) if t in "ABCD" else 1 for t in text]
            def batch_decode(self, batch, skip_special_tokens):
                return [" ".join("word" for t in row if t == 9) for row in batch]
        proc = SimpleNamespace(tokenizer=Tokenizer(), audio_token="<audio>",
                               feature_extractor=SimpleNamespace(sampling_rate=16000))
        prompt = torch.tensor([[1, 4, 5, 5, 6, 8]])
        batch = dict(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                     input_features=torch.zeros(1, 8, 8),
                     feature_attention_mask=torch.ones(1, 8, dtype=torch.long))
        with patch.object(OmniCollator, "prompts", return_value=batch), \
             patch.object(OmniCollator, "prompt_text", return_value="X"):
            options = dict(device="cpu", max_new_tokens=2, batch_size=1)
            asr = evaluate(model, proc, [{"text": "word word"}], task="asr", **options)
            ser = evaluate(model, proc, [{"emotion": "happy"}], task="ser", **options)
        self.assertEqual(asr["wer"], 0.)
        self.assertEqual(asr["max_new_tokens_without_eos_count"], 1)
        self.assertEqual(asr["max_new_tokens_without_eos_ratio"], 1.)
        self.assertNotIn("max_new_tokens_without_eos_ratio", ser)
        self.assertEqual(ser["accuracy"], 1.)

    def test_help_does_not_import_gpu_or_transformers(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "examples.omni_talora.evaluate", "--help"],
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--manifest", result.stdout)


class CheckpointConfigTests(unittest.TestCase):
    def test_read_rejects_incomplete_or_invalid_semantics(self):
        from examples.omni_talora.checkpoint import read_config

        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "adapter_config.json"
            for data in ("[]", "{}", "{", json.dumps(dict(CONFIG, revision="main"))):
                filename.write_text(data)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    read_config(tmp)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = torch.nn.Module()
        self.model.base = torch.nn.Linear(2, 2).requires_grad_(False)
        self.model.ta_adapter = torch.nn.ParameterDict({
            "down": torch.nn.Parameter(torch.ones(2, 2)),
            "up": torch.nn.Parameter(torch.ones(2, 2) * 2),
            "gate": torch.nn.Parameter(torch.tensor([0., 0.])),
        })
        self.model.ta_adapter.scale = 2.0

    def test_publish_does_not_overwrite_or_merge_existing_runs(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from examples.omni_talora.checkpoint import save_adapter, read_config

        start = Barrier(2)
        def try_save(task):
            start.wait()
            try:
                save_adapter(self.model, self.directory, dict(CONFIG, task=task))
            except FileExistsError:
                return None
            return task
        with ThreadPoolExecutor(2) as executor:
            outcomes = list(executor.map(try_save, ("ser", "asr")))
        self.assertEqual(outcomes.count(None), 1)
        self.assertIn(read_config(self.directory)["task"], outcomes)
        for name in ("adapter_config.json", "adapter.safetensors"):
            partial = self.directory / (name + ".partial")
            partial.mkdir()
            (partial / name).write_bytes(b"previous run")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                save_adapter(self.model, partial, CONFIG)
            self.assertEqual((partial / name).read_bytes(), b"previous run")
            self.assertEqual(list(partial.iterdir()), [partial / name])

    def test_metadata_must_describe_installed_adapter(self):
        from examples.omni_talora.checkpoint import save_adapter, load_adapter

        for change in ({"last_n":2}, {"rank":5}, {"alpha":8}, {"mode":"lora"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                save_adapter(self.model, self.directory, dict(CONFIG, **change))
        save_adapter(self.model, self.directory, CONFIG)
        self.model.ta_adapter.scale = 8.
        with self.assertRaises(ValueError):
            load_adapter(self.model, self.directory)

    def test_roundtrip_saves_only_adapter_and_never_changes_base(self):
        import torch
        from safetensors.torch import load_file
        from examples.omni_talora.checkpoint import load_adapter, save_adapter

        save_adapter(self.model, self.directory, CONFIG)
        keys = load_file(str(self.directory / "adapter.safetensors"))
        self.assertEqual(set(keys), {"ta_adapter.down", "ta_adapter.up", "ta_adapter.gate"})
        with torch.no_grad():
            self.model.ta_adapter.down.zero_()
            self.model.base.weight.fill_(42)
        self.assertEqual(load_adapter(self.model, self.directory), CONFIG)
        torch.testing.assert_close(self.model.ta_adapter.down, torch.ones(2, 2))
        torch.testing.assert_close(self.model.base.weight, torch.full((2, 2), 42.0))

    def test_reject_bad_training_state_and_config(self):
        from examples.omni_talora.checkpoint import save_adapter

        for extra in ({"rank":0}, {"alpha":float("nan")}, {"last_n":False}, {"mode":"unknown"},
                      {"revision":"main"}, {"model_name_or_path":""}, {"task":"qa"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                save_adapter(self.model, self.directory, dict(CONFIG, **extra))
        self.model.base.requires_grad_(True)
        with self.assertRaises(ValueError):
            save_adapter(self.model, self.directory, CONFIG)

    def test_reject_bad_tensor_set_before_copying_anything(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        from examples.omni_talora.checkpoint import load_adapter, save_adapter

        save_adapter(self.model, self.directory, CONFIG)
        archive = str(self.directory / "adapter.safetensors")
        with safe_open(archive, framework="pt") as file:
            metadata = file.metadata()
        original = load_file(archive)
        with torch.no_grad():
            self.model.ta_adapter.down.zero_()
        for change in ("extra", "missing", "shape", "dtype", "nan"):
            state = dict(original)
            if change == "extra":
                state["base.weight"] = torch.ones(2, 2)
            elif change == "missing":
                state.pop("ta_adapter.up")
            elif change == "shape":
                state["ta_adapter.up"] = torch.ones(3)
            elif change == "dtype":
                state["ta_adapter.up"] = torch.ones(2, 2, dtype=torch.float64)
            else:
                state["ta_adapter.up"] = torch.full((2, 2), float("nan"))
            save_file(state, archive, metadata=metadata)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_adapter(self.model, self.directory)
            self.assertEqual(self.model.ta_adapter.down.count_nonzero().item(), 0)

    def test_refuses_mismatched_json_and_weight_pair(self):
        from examples.omni_talora.checkpoint import load_adapter, save_adapter

        save_adapter(self.model, self.directory, CONFIG)
        (self.directory / "adapter_config.json").write_text(
            json.dumps(dict(CONFIG, alpha=999)))
        with self.assertRaises(ValueError):
            load_adapter(self.model, self.directory)
