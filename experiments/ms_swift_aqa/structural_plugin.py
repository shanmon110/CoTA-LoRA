"""Omni decoder metadata and native Swift checkpoint handling for the AQA comparison."""
from functools import wraps
from pathlib import Path
from types import MethodType

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from swift.model import MODEL_ARCH_MAPPING
from swift.tuner_plugin import Tuner, tuners_map
from swift.tuners import Swift
from swift.tuners.mapping import SWIFT_MAPPING, SwiftTuners
from swift.tuners.reft import Reft, ReftConfig


class OmniSwiftTuner(Tuner):
    """Delegate attachment and serialization to Swift's native bottleneck/block tuners."""

    @staticmethod
    def _prepare_wrapper(model):
        # LLaMA-Pro updates the outer config. The Omni thinker and its KV cache read the nested one.
        model.config.get_text_config().num_hidden_layers = len(model.thinker.model.layers)
        return model

    @classmethod
    def prepare_model(cls, args, model):
        from swift.pipelines.train.tuner import prepare_adapter
        return cls._prepare_wrapper(prepare_adapter(args, model))

    @classmethod
    def from_pretrained(cls, model, model_id, is_trainable=False, **kwargs):
        # Swift's native wrapper calls this flag inference_mode. PEFT calls it is_trainable.
        return cls._prepare_wrapper(Swift.from_pretrained(model, model_id, inference_mode=not is_trainable, **kwargs))

    @staticmethod
    def save_pretrained(model, save_directory, **kwargs):
        model.save_pretrained(save_directory, **kwargs)


class OmniReft(Reft):
    """Use Swift's PyReFT interventions and its per-intervention native checkpoint layout."""

    @staticmethod
    def prepare_model(model, config, adapter_name):
        # Swift's ReFT builder reads the dimension from the outer config.
        model.config.hidden_size = model.config.get_text_config().hidden_size
        config.model_type = 'qwen2_5_omni'
        output = Reft.prepare_model(model, config, adapter_name)
        reft = output.model
        native_save = output.save_callback

        def save(swift_model, folder, name):
            native_save(swift_model, folder, name)
            # PyReFT's compact LoReFT files store the effective orthogonal matrix. Exact
            # optimizer continuation also needs PyTorch's coordinate tensor and its basis.
            rotation = {}
            for key, intervention in reft.interventions.items():
                for k, v in intervention.rotate_layer.state_dict().items():
                    rotation[f'{key}.{k}'] = v.detach().cpu().contiguous()
            save_file(rotation, str(Path(folder) / 'rotation_state.safetensors'))

        def load(swift_model, folder, name):
            # PyVene 0.1.8 load_intervention hardcodes cuda:0. Restore on the attached device.
            with safe_open(str(Path(folder) / 'rotation_state.safetensors'), framework='pt', device='cpu') as rotation:
                for key, intervention in reft.interventions.items():
                    state = torch.load(Path(folder) / f'intkey_{key}.bin', map_location='cpu', weights_only=True)
                    intervention.learned_source.load_state_dict({k: state[k] for k in ('weight', 'bias')})
                    intervention.rotate_layer.load_state_dict({
                        k: rotation.get_tensor(f'{key}.{k}') for k in intervention.rotate_layer.state_dict()
                    })

        output.save_callback = save
        output.load_callback = load
        return output


class OmniReftTuner(OmniSwiftTuner):
    @staticmethod
    def _prepare_wrapper(model):
        reft = model.base_model

        @wraps(reft.model.forward.__func__)
        def forward(self, input_ids=None, **kwargs):
            # Swift training supplies audio-conditioned inputs_embeds plus RoPE position_ids.
            # Passing PyVene's base explicitly preserves both and bypasses the text-only Swift input hook.
            base = {'input_ids': input_ids} if input_ids is not None else {'inputs_embeds': kwargs.pop('inputs_embeds')}
            labels = kwargs.pop('labels', None)
            return reft(base={**base, **kwargs}, labels=labels)

        @wraps(reft.model.generate.__func__)
        def generate(self, input_ids=None, **kwargs):
            # All prompt positions receive the same interventions as SFT. The AQA output is a single new token.
            return reft.generate_origin(base={'input_ids': input_ids}, intervene_on_prompt=True, **kwargs)[1]

        model.forward = MethodType(forward, model)
        model.generate = MethodType(generate, model)
        return model


arch = MODEL_ARCH_MAPPING['qwen2_5_omni']
for field, value in {
    'module_list': 'thinker.model.layers',
    'attention': 'thinker.model.layers.{}.self_attn',
    'o_proj': 'thinker.model.layers.{}.self_attn.o_proj',
    'mlp': 'thinker.model.layers.{}.mlp',
    'down_proj': 'thinker.model.layers.{}.mlp.down_proj',
}.items():
    if getattr(arch, field) not in (None, value):
        raise ValueError(f'Unexpected Omni architecture entry {field}={getattr(arch, field)}')
    setattr(arch, field, value)

SWIFT_MAPPING[SwiftTuners.REFT] = (ReftConfig, OmniReft)
tuners_map.update(adapter=OmniSwiftTuner, llamapro=OmniSwiftTuner, reft=OmniReftTuner)
