# Copyright (c) Aman Urumbekov and other contributors.
"""
ClarityCore Models

This module provides model architectures and training wrappers for low-level vision tasks.

Usage:
    from claritycore.models import AutoModel, AutoConfig

    # Load a model
    config = AutoConfig.from_name("rrdbnet")
    model = AutoModel.from_config(config)

    # Or directly
    from claritycore.models.rrdbnet import RRDBNet, RRDBNetConfig
"""

from claritycore.models.base import BaseConfig, BaseModel
from claritycore.models.auto import AutoConfig, AutoModel

# Import models to register them
from claritycore.models import rrdbnet  # noqa: F401

__all__ = [
    "BaseConfig",
    "BaseModel",
    "AutoConfig",
    "AutoModel",
]

