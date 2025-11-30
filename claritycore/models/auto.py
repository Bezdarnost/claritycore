# Copyright (c) Aman Urumbekov and other contributors.
"""Auto classes for automatic model loading."""

from typing import Any

from claritycore.models.base import BaseConfig, BaseModel

# Registry mapping model_type -> (ConfigClass, ModelClass)
_CONFIG_REGISTRY: dict[str, type[BaseConfig]] = {}
_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(model_type: str):
    """
    Decorator to register a model class.

    Usage:
        @register_model("rrdbnet")
        class RRDBNetModel(BaseModel):
            ...
    """

    def decorator(cls: type[BaseModel]):
        _MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def register_config(model_type: str):
    """
    Decorator to register a config class.

    Usage:
        @register_config("rrdbnet")
        @dataclass
        class RRDBNetConfig(BaseConfig):
            ...
    """

    def decorator(cls: type[BaseConfig]):
        _CONFIG_REGISTRY[model_type] = cls
        return cls

    return decorator


class AutoConfig:
    """
    Auto class for loading model configurations.

    Usage:
        config = AutoConfig.from_name("rrdbnet")
        config = AutoConfig.from_dict({"model_type": "rrdbnet", "scale": 4})
    """

    @classmethod
    def from_name(cls, model_type: str, **kwargs) -> BaseConfig:
        """
        Create a config for a model type.

        Args:
            model_type: The type of model (e.g., "rrdbnet", "hat").
            **kwargs: Override default config values.

        Returns:
            Config instance.
        """
        if model_type not in _CONFIG_REGISTRY:
            available = list(_CONFIG_REGISTRY.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        config_cls = _CONFIG_REGISTRY[model_type]
        return config_cls(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> BaseConfig:
        """
        Create a config from a dictionary.

        The dict must contain a 'model_type' key.
        """
        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Config dict must contain 'model_type' key")

        if model_type not in _CONFIG_REGISTRY:
            available = list(_CONFIG_REGISTRY.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        config_cls = _CONFIG_REGISTRY[model_type]
        return config_cls.from_dict(config_dict)

    @classmethod
    def from_pretrained(cls, path: str) -> BaseConfig:
        """Load config from a saved model directory."""
        import json
        import os

        config_path = os.path.join(path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(_CONFIG_REGISTRY.keys())


class AutoModel:
    """
    Auto class for loading models.

    Usage:
        model = AutoModel.from_config(config)
        model = AutoModel.from_pretrained("path/to/model")
    """

    @classmethod
    def from_config(cls, config: BaseConfig) -> BaseModel:
        """
        Create a model from a config.

        Args:
            config: Model configuration.

        Returns:
            Model instance.
        """
        model_type = config.model_type

        if model_type not in _MODEL_REGISTRY:
            available = list(_MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        model_cls = _MODEL_REGISTRY[model_type]
        return model_cls(config)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> BaseModel:
        """
        Load a model from a saved directory.

        Args:
            path: Path to saved model directory.
            **kwargs: Override config parameters.

        Returns:
            Loaded model.
        """
        config = AutoConfig.from_pretrained(path)
        model_type = config.model_type

        if model_type not in _MODEL_REGISTRY:
            available = list(_MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        model_cls = _MODEL_REGISTRY[model_type]
        return model_cls.from_pretrained(path, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types."""
        return list(_MODEL_REGISTRY.keys())


__all__ = [
    "AutoConfig",
    "AutoModel",
    "register_config",
    "register_model",
]

