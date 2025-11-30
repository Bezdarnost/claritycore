# Copyright (c) Aman Urumbekov and other contributors.
"""Base classes for ClarityCore models."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

# ═══════════════════════════════════════════════════════════════════════════════
# BASE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BaseConfig:
    """
    Base configuration class for all models.

    Subclass this to create model-specific configurations.
    All config classes use dataclass for clean, typed configuration.

    Example:
        @dataclass
        class RRDBNetConfig(BaseConfig):
            num_feat: int = 64
            num_block: int = 23
            scale: int = 4
    """

    # Model identification
    model_type: str = "base"

    # Common parameters (shared by all models)
    in_channels: int = 3
    out_channels: int = 3
    scale: int = 4

    # Fields that are common to all models (not architecture-specific)
    _common_fields: tuple[str, ...] = ("model_type", "in_channels", "out_channels", "scale")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def get_arch_config(self) -> dict[str, Any]:
        """
        Get architecture-specific parameters (excludes common fields).

        Returns dict of parameters unique to this model architecture.
        """
        result = {}
        for field_name in self.__dataclass_fields__:
            if field_name not in self._common_fields and not field_name.startswith("_"):
                result[field_name] = getattr(self, field_name)
        return result

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """Load config from JSON file."""
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))


# ═══════════════════════════════════════════════════════════════════════════════
# BASE MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all ClarityCore models.

    A Model combines:
    - Architecture (nn.Module) - the actual neural network
    - Config - hyperparameters and settings
    - Training logic - loss computation, optimization

    Unlike raw architectures, Models are "training-aware" and know how to:
    - Compute losses
    - Handle optimization steps
    - Manage EMA weights
    - Save/load checkpoints

    Subclasses must implement:
    - `_build_network()`: Construct the architecture
    - `forward()`: Forward pass

    Args:
        config: Model configuration.
    """

    config_class: type[BaseConfig] = BaseConfig
    supports_gradient_checkpointing: bool = False

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config

        # Core network (built by subclass)
        self.net: nn.Module | None = None

        # EMA network
        self.net_ema: nn.Module | None = None
        self.ema_decay: float = 0.999
        self.use_ema: bool = False

        # Loss functions
        self._losses: nn.ModuleDict = nn.ModuleDict()
        self._loss_weights: dict[str, float] = {}

        # Build the network
        self._build_network()

    @abstractmethod
    def _build_network(self) -> None:
        """Build the network architecture. Must set self.net."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Loss Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_loss(self, name: str, loss_fn: nn.Module, weight: float = 1.0) -> None:
        """
        Add a loss function to the model.

        Args:
            name: Name for the loss (used in logging).
            loss_fn: Loss module.
            weight: Weight for this loss in the total.
        """
        self._losses[name] = loss_fn
        self._loss_weights[name] = weight

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute weighted sum of all registered losses.

        Args:
            pred: Model prediction.
            target: Ground truth.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        device = pred.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, float] = {}

        for name, loss_fn in self._losses.items():
            loss_val = loss_fn(pred, target)
            weighted = self._loss_weights[name] * loss_val
            total_loss = total_loss + weighted
            loss_dict[name] = loss_val.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    # ─────────────────────────────────────────────────────────────────────────
    # EMA (Exponential Moving Average)
    # ─────────────────────────────────────────────────────────────────────────

    def init_ema(self, decay: float = 0.999) -> None:
        """Initialize EMA network."""
        if self.net is None:
            raise ValueError("Network must be built before initializing EMA")

        self.use_ema = True
        self.ema_decay = decay
        self.net_ema = deepcopy(self.net).eval()

        for p in self.net_ema.parameters():
            p.requires_grad_(False)

        logger.info(f"Initialized EMA with decay={decay}")

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA weights."""
        if not self.use_ema or self.net_ema is None:
            return

        for p_ema, p in zip(self.net_ema.parameters(), self.net.parameters()):
            p_ema.data.lerp_(p.data, 1 - self.ema_decay)

        for b_ema, b in zip(self.net_ema.buffers(), self.net.buffers()):
            b_ema.data.copy_(b.data)

    def forward_ema(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using EMA weights."""
        if self.net_ema is None:
            return self.forward(x)

        with torch.no_grad():
            return self.net_ema(x)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────────────────

    def save_pretrained(self, path: str, save_config: bool = True) -> None:
        """
        Save model weights and config.

        Args:
            path: Directory to save to.
            save_config: Whether to also save the config.
        """
        import os

        os.makedirs(path, exist_ok=True)

        # Save weights
        weights_path = os.path.join(path, "model.pt")
        state = {"net": self.net.state_dict() if self.net else {}}
        if self.net_ema is not None:
            state["net_ema"] = self.net_ema.state_dict()
        torch.save(state, weights_path)

        # Save config
        if save_config:
            config_path = os.path.join(path, "config.json")
            self.config.save(config_path)

        logger.info(f"Saved model to {path}")

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BaseModel":
        """
        Load model from saved weights and config.

        Args:
            path: Directory containing model.pt and config.json.
            **kwargs: Override config parameters.

        Returns:
            Loaded model.
        """
        import os

        # Load config
        config_path = os.path.join(path, "config.json")
        config = cls.config_class.load(config_path)

        # Override with kwargs
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = os.path.join(path, "model.pt")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)

        if "net" in state and model.net is not None:
            model.net.load_state_dict(state["net"])

        if "net_ema" in state and model.net_ema is not None:
            model.net_ema.load_state_dict(state["net_ema"])

        logger.info(f"Loaded model from {path}")
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if self.net is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.net.parameters())

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of model parameters."""
        return next(self.parameters()).dtype

    def __repr__(self) -> str:
        params = self.count_parameters()
        return f"{self.__class__.__name__}(params={params:,}, config={self.config.model_type})"
