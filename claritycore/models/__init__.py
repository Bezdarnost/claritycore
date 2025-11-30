# Copyright (c) Aman Urumbekov and other contributors.
"""
ClarityCore Models

This module provides model architectures and training wrappers for low-level vision tasks.

Available Models:
    - RRDBNet: ESRGAN-style generator for super-resolution (~16.7M params full, ~1M lite)

Usage:
    from claritycore.models import AutoModel, AutoConfig

    # Load a model by name
    config = AutoConfig.from_name("rrdbnet", scale=4)
    model = AutoModel.from_config(config)

    # Use presets
    from claritycore.models.rrdbnet import RRDBNetConfig
    config, training_preset = RRDBNetConfig.get_preset("rrdbnetx4")

    # Or import directly
    from claritycore.models.rrdbnet import RRDBNet, RRDBNetConfig, RRDBNetModel
"""

from claritycore.models.auto import AutoConfig, AutoModel, register_config, register_model
from claritycore.models.base import BaseConfig, BaseModel

# Import model modules to register them with AutoConfig/AutoModel
# These imports trigger the @register_config and @register_model decorators
from claritycore.models.rrdbnet import RRDBNetConfig, RRDBNetModel

__all__ = [
    # Base classes
    "BaseConfig",
    "BaseModel",
    # Auto classes
    "AutoConfig",
    "AutoModel",
    "register_config",
    "register_model",
    # RRDBNet (explicit exports)
    "RRDBNetConfig",
    "RRDBNetModel",
]
