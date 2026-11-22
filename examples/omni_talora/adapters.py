"""Omni host binding and compatibility exports for existing checkpoints."""
from west_adapters.base import targets
from west_adapters.install import baseline_trainables, install_from_config as install, validate_options
from west_adapters.qv_lora import qv_budget
from west_adapters.registry import checkpoint_method

MODES = ("ta", "lora", "toplora", "bora", "dora", "eva")


def is_legacy(config):
    return checkpoint_method(config).attention


def install_from_config(model, config, *, directions=None):
    from .attention import install_adapter
    return install(model, config, directions=directions, attention_installer=install_adapter)
