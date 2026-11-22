# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from contextlib import nullcontext
from typing import Optional

import torch
import wenet
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationMixin, PreTrainedModel)

from west.utils.utils import freeze_module

from .configuration_touch_asu import TouchASUConfig


class ProjectorCov1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, config.projector_hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class TouchASU(PreTrainedModel, GenerationMixin):
    """ LLM based Automatic Speech Understanding
    """
    model_type = 'touch_asu'
    config_class = TouchASUConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TouchASUConfig):
        super().__init__(config)
        self._tagged = isinstance(config.lora_config, dict) and "method" in config.lora_config
        self.needs_adapter_calibration = False
        self._calibrating = False
        self._needs_audio_context = False
        llm_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            config=llm_config,
            torch_dtype='auto',
            attn_implementation=config.adapter_attention_backend if self._tagged else "flash_attention_2",
        )
        self.encoder = wenet.load_model(config.wenet_model_name_or_path)
        # remove decoder and ctc, keep encoder only
        del self.encoder.decoder
        del self.encoder.ctc
        encoder_dim = self.encoder.encoder.output_size()
        config.hidden_size = llm_config.hidden_size  # for deepseed training
        self.projector = ProjectorCov1d(config, encoder_dim,
                                        llm_config.hidden_size)
        total_params = sum(p.numel() for p in self.projector.parameters())
        print('Projector total params: {:.2f}M'.format(total_params / 1024 /
                                                       1024))
        if self._tagged:
            self._init_adapter(config)
        elif config.lora_config is not None:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(**config.lora_config)
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        self.freeze_encoder()
        if config.lora_config is None:
            self.freeze_llm()

    def _init_adapter(self, config):
        from west_adapters import finalize_ranks, resolve_layout, resolve_lora_config

        self._adapter_options = resolve_lora_config(config.lora_config)
        method, layout, projections = resolve_layout(self.llm, self._adapter_options)
        self._needs_audio_context = method.needs_audio_context
        saved = getattr(config, "resolved_adapter_config", None)
        if saved is not None:
            # A fitted sample count may be below the requested cap.
            for key, value in layout.items():
                if key == "eva_samples" and 0 < saved.get(key, 0) <= value:
                    continue
                if saved.get(key) != value:
                    raise ValueError(f"Saved adapter conflicts with lora_config at {key}")
            self._install_adapter(saved)
        elif method.uses_directions:
            self.needs_adapter_calibration = True
            self.freeze_llm()
        else:
            finalize_ranks(layout, projections)
            self._install_adapter(layout)

    def _install_adapter(self, layout, directions=None):
        from west_adapters import install_from_config
        from .adapter_touch_asu import install_attention

        install_from_config(self.llm, layout, directions=directions,
                            attention_installer=install_attention)
        self.config.resolved_adapter_config = layout
        self.needs_adapter_calibration = False

    def calibrate_adapter(self, train_dataset):
        from torch import distributed as dist

        if not self.needs_adapter_calibration or train_dataset.extractor.inference:
            raise ValueError("EVA requires a fresh adapter and a training dataset")
        if not dist.is_initialized() or dist.get_world_size() == 1:
            self._fit_adapter(train_dataset)
            return

        # SVD on one worker determines the topology before DDP wraps the model.
        # Publish failures too, so peers do not wait for tensors that never exist.
        metadata = [None, None]
        if dist.get_rank() == 0:
            try:
                self._fit_adapter(train_dataset)
                metadata[0] = self.config.resolved_adapter_config
            except Exception as exc:
                metadata[1] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(metadata, src=0, device=self.device)
        if metadata[1]:
            raise RuntimeError(f"EVA calibration on rank 0 failed: {metadata[1]}")
        if dist.get_rank() != 0:
            self._install_adapter(metadata[0])
        with torch.no_grad():
            for param in self.llm.parameters():
                if param.requires_grad:
                    dist.broadcast(param, src=0)

    def _fit_adapter(self, train_dataset):
        from west_adapters import finalize_ranks, resolve_layout
        from .adapter_touch_asu import calibration_batches

        method, layout, projections = resolve_layout(self.llm, self._adapter_options)
        count = 0

        def batches():
            nonlocal count
            for batch in calibration_batches(train_dataset, layout):
                count += 1
                yield batch

        self._calibrating = True
        try:
            directions = method.calibrate_batches(self.llm, layout, batches(), self.device,
                                                  forward_model=self)
        finally:
            self._calibrating = False
        layout["eva_samples"] = count
        finalize_ranks(layout, projections, directions)
        self._install_adapter(layout, directions)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if self._needs_audio_context:
            from west_adapters.context import checkpoint_audio_context

            gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})
            gradient_checkpointing_kwargs.update(use_reentrant=False,
                                                 context_fn=checkpoint_audio_context)
        return super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def tie_weights(self):
        return self.llm.tie_weights()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_speech_embeddings(self, audio_features, audio_features_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_features_lengths)
        speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
        speech_proj = self.projector(speech_emb)
        speech_proj_lens = mask.squeeze(1).sum(1) // self.projector.k
        return speech_proj, speech_proj_lens

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        return_audio_mask=False,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_features_lengths)
        inputs_embeds = text_emb
        audio = torch.zeros_like(input_ids, dtype=torch.bool) if return_audio_mask else None
        for i in range(audio_features.size(0)):
            if not has_audio[i]:
                continue
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            inputs_embeds[b, s:e, :] = speech_emb[i, :speech_emb_lens[i], :]
            if audio is not None:
                audio[b, s:e] = True
        if return_audio_mask:
            return inputs_embeds, audio
        return inputs_embeds

    def _decoder_inputs(self, input_ids, attention_mask, position_ids, audio_offsets,
                        audio_features, audio_features_lengths, batch_idx, has_audio,
                        generation=False, layout_repeats=1):
        if self.needs_adapter_calibration and not self._calibrating:
            raise RuntimeError("Fresh EVA requires training calibration before use")
        use_layout = self._needs_audio_context
        mixed = self.compute_mix_embedding(
            input_ids, audio_offsets, audio_features, audio_features_lengths,
            batch_idx, has_audio, return_audio_mask=use_layout)
        embeds, audio = mixed if use_layout else (mixed, None)
        valid = attention_mask
        if self._tagged and generation and valid is not None and not valid[:, -1].all():
            # WEST collates right-padded prompts. HF generates from the last
            # column, so move padding to the left for all seven methods.
            order = valid.int().argsort(dim=-1, stable=True)
            embeds = embeds.gather(1, order[:, :, None].expand_as(embeds))
            valid = attention_mask = valid.gather(1, order)
            if use_layout:
                audio = audio.gather(1, order)
        context = nullcontext()
        if use_layout:
            from west_adapters.context import audio_mask_context

            valid = torch.ones_like(audio) if valid is None else valid
            if layout_repeats > 1:
                # HF expands each prompt into adjacent beams and only reorders
                # beams within that same prompt. Appended tokens remain text.
                audio = audio.repeat_interleave(layout_repeats, dim=0)
                valid = valid.repeat_interleave(layout_repeats, dim=0)
            context = audio_mask_context(audio, valid)
        if (self._tagged and self.config.adapter_attention_backend == "eager"
                and position_ids is not None and attention_mask is None):
            from .adapter_touch_asu import packed_attention_mask

            config = self.llm.config
            window = config.sliding_window if getattr(config, "use_sliding_window", False) else None
            attention_mask = packed_attention_mask(position_ids, embeds.dtype, window)
        return embeds, attention_mask, context

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        inputs_embeds, attention_mask, context = self._decoder_inputs(
            input_ids, attention_mask, position_ids, audio_offsets, audio_features,
            audio_features_lengths, batch_idx, has_audio)
        with context:
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            labels=labels, position_ids=position_ids, **kwargs)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        generation_config = kwargs.pop("generation_config", self.generation_config)
        repeats = 1
        if self._needs_audio_context:
            repeats = max(kwargs.get("num_beams", generation_config.num_beams),
                          kwargs.get("num_return_sequences", generation_config.num_return_sequences))
        inputs_embeds, attention_mask, context = self._decoder_inputs(
            input_ids, attention_mask, None, audio_offsets, audio_features,
            audio_features_lengths, batch_idx, has_audio, generation=True, layout_repeats=repeats)
        with context:
            return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     generation_config=generation_config, **kwargs)

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_module(self.encoder)
        self.encoder.eval()

    def freeze_llm(self):
        freeze_module(self.llm)

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name_or_path,
            padding_side="right",
        )
        # We only support QWen now
        tokenizer.bos_token = tokenizer.eos_token
        return tokenizer
