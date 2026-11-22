"""Version-specific boundary with the official Transformers Omni checkpoint."""

from pathlib import Path
import hashlib
import json


def move_inputs(batch, device, dtype):
    from west_adapters.base import move_inputs as to_device
    return to_device(batch, device, dtype)


def local_snapshot_sha256(directory):
    """Bind local Omni weights and processor inputs, following Hub snapshot symlinks."""
    root = Path(directory)
    names = {"config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "added_tokens.json", "special_tokens_map.json", "preprocessor_config.json",
             "processor_config.json", "video_preprocessor_config.json", "chat_template.json",
             "chat_template.jinja"}
    files = [root / name for name in names if (root / name).is_file()]
    files.extend(root.glob("*.safetensors"))
    files.extend((root / "chat_templates").glob("*.jinja"))
    if not (root / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"Expected local model config and safetensors in {root}")
    entries = []
    for path in sorted(files):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append((path.relative_to(root).as_posix(), digest))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def load_base(model_name_or_path, revision, device):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5OmniConfig, Qwen2_5OmniThinkerForConditionalGeneration

    if transformers.__version__ != "4.52.3":
        raise RuntimeError("This eager attention recipe requires transformers==4.52.3.")
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu only for small checks.")
        with torch.cuda.device(dev):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise RuntimeError(f"Device {dev} needs native BF16 support for this recipe.")
        dtype = torch.bfloat16
    elif dev.type == "cpu":
        dtype = torch.float32
    else:
        raise ValueError("Supported devices: cpu or cuda[:index]")
    # Read the nested config explicitly. Passing a full Omni config to the
    # Thinker can silently construct the default (7B) text configuration.
    full = Qwen2_5OmniConfig.from_pretrained(model_name_or_path, revision=revision)
    resolved = getattr(full, "_commit_hash", None) or revision
    model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name_or_path, config=full.thinker_config, revision=resolved,
        attn_implementation="eager", torch_dtype=dtype, low_cpu_mem_usage=True,
        use_safetensors=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete Thinker weights: missing={info['missing_keys']}, "
                           f"mismatched={info['mismatched_keys']}, errors={info.get('error_msgs')}")
    # The Talker/token2wav weights from the full checkpoint are intentionally unused.
    processor = AutoProcessor.from_pretrained(model_name_or_path, revision=resolved)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.convert_tokens_to_ids(processor.audio_token) != model.config.audio_token_id:
        raise ValueError("Thinker and processor disagree on the audio placeholder token.")
    model.config._commit_hash = None if Path(model_name_or_path).is_dir() else resolved
    return model.to(dev), processor
