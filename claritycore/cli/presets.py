# Copyright (c) Aman Urumbekov and other contributors.
"""Training presets and dataset configurations."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PRESETS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetPreset:
    """Known dataset configuration."""

    name: str
    expected_train_size: int
    expected_val_size: int | None = None
    train_subdir: str = "train"
    val_subdir: str = "val"
    target_subdir: str = "hr"
    input_pattern: str = "x{scale}"


DATASET_REGISTRY: dict[str, DatasetPreset] = {
    "div2k": DatasetPreset(
        name="DIV2K",
        expected_train_size=800,
        expected_val_size=100,
        train_subdir=".",
        val_subdir=".",
        target_subdir="hr",
        input_pattern="x{scale}",
    ),
    "df2k": DatasetPreset(
        name="DF2K",
        expected_train_size=3450,
        expected_val_size=100,
        target_subdir="hr",
        input_pattern="x{scale}",
    ),
    "ffhq": DatasetPreset(
        name="FFHQ",
        expected_train_size=70000,
        expected_val_size=None,
        target_subdir="images",
        input_pattern="x{scale}",
    ),
    "set5": DatasetPreset(name="Set5", expected_train_size=5, expected_val_size=5),
    "set14": DatasetPreset(name="Set14", expected_train_size=14, expected_val_size=14),
    "bsd100": DatasetPreset(name="BSD100", expected_train_size=100, expected_val_size=100),
    "urban100": DatasetPreset(name="Urban100", expected_train_size=100, expected_val_size=100),
    "manga109": DatasetPreset(name="Manga109", expected_train_size=109, expected_val_size=109),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PRESET
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingPreset:
    """
    Unified training configuration preset.

    Model-agnostic structure that holds both architecture config and training params.
    The `model_config` dict stores the actual model configuration object.
    """

    # Model identification
    model: str
    scale: int

    # Model config object (BaseConfig subclass instance)
    config: Any = None

    # Training hyperparameters
    total_steps: int = 400000
    batch_size: int = 16
    patch_size: int = 256
    lr: float = 2e-4
    loss: str = "l1"

    # Logging
    log_freq: int = 100
    val_freq: int = 5000
    save_freq: int = 5000

    # Data
    dataset: str = "div2k"
    dataset_paths: list[str] = field(default_factory=lambda: ["datasets/DIV2K", "data/DIV2K"])
    norm: str = "minus_one_one"

    # Features
    use_amp: bool = True
    use_ema: bool = False

    def get_arch_params(self) -> dict[str, Any]:
        """Get architecture-specific parameters from config."""
        if self.config is None:
            return {}
        if hasattr(self.config, "get_arch_config"):
            return self.config.get_arch_config()
        return {}

    def get_cli_args(self) -> list[str]:
        """
        Convert preset to CLI arguments for train.py.

        Returns list like: ["--num-feat", "64", "--num-block", "23"]
        """
        args = []
        for key, value in self.get_arch_params().items():
            cli_key = f"--{key.replace('_', '-')}"
            args.extend([cli_key, str(value)])
        return args


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

# Global registry of all training presets
_PRESET_REGISTRY: dict[str, TrainingPreset] = {}


def register_preset(name: str, preset: TrainingPreset) -> None:
    """Register a training preset globally."""
    _PRESET_REGISTRY[name.lower()] = preset


def _collect_model_presets() -> dict[str, TrainingPreset]:
    """
    Auto-discover and collect presets from all registered models.

    Each model module should define a PRESETS dict with (config, training_preset) tuples.
    This function converts them to unified TrainingPreset objects.
    """
    presets = dict(_PRESET_REGISTRY)

    # Auto-discover from models that define presets
    try:
        from claritycore.models.rrdbnet.config import RRDBNET_PRESETS

        for name, (config, training) in RRDBNET_PRESETS.items():
            if name.lower() not in presets:
                presets[name.lower()] = TrainingPreset(
                    model=config.model_type,
                    scale=config.scale,
                    config=config,
                    total_steps=training.total_steps,
                    batch_size=training.batch_size,
                    patch_size=training.patch_size,
                    lr=training.lr,
                    loss=training.loss,
                    log_freq=training.log_freq,
                    val_freq=training.val_freq,
                    save_freq=training.save_freq,
                    dataset=training.dataset,
                    dataset_paths=list(training.dataset_paths),
                    norm=training.norm,
                    use_amp=training.use_amp,
                    use_ema=training.use_ema,
                )
    except ImportError:
        pass

    # Future: Add more models here
    # The pattern is simple - import MODEL_PRESETS and convert to TrainingPreset
    #
    # try:
    #     from claritycore.models.hat.config import HAT_PRESETS
    #     for name, (config, training) in HAT_PRESETS.items():
    #         presets[name.lower()] = TrainingPreset(...)
    # except ImportError:
    #     pass

    return presets


def get_all_presets() -> dict[str, TrainingPreset]:
    """Get all available training presets."""
    return _collect_model_presets()


def list_presets() -> list[str]:
    """List all available preset names."""
    return sorted(get_all_presets().keys())


def get_preset(name: str) -> TrainingPreset | None:
    """Get a training preset by name."""
    return get_all_presets().get(name.lower())


# ═══════════════════════════════════════════════════════════════════════════════
# PRESET CREATION
# ═══════════════════════════════════════════════════════════════════════════════


def parse_preset_name(name: str) -> tuple[str, int] | None:
    """
    Parse a preset name like 'rrdbnetx4' into (model, scale).

    Returns None if not a valid preset pattern.
    """
    all_presets = get_all_presets()

    # Try exact match first
    if name.lower() in all_presets:
        preset = all_presets[name.lower()]
        return preset.model, preset.scale

    # Try to parse pattern like "modelx4" or "model-x4"
    patterns = [
        r"^([a-z0-9_-]+?)x(\d+)$",  # rrdbnetx4
        r"^([a-z0-9_-]+?)-x(\d+)$",  # rrdbnet-x4
        r"^([a-z0-9_-]+?)_x(\d+)$",  # rrdbnet_x4
    ]

    for pattern in patterns:
        match = re.match(pattern, name.lower())
        if match:
            return match.group(1), int(match.group(2))

    return None


def create_preset_from_name(name: str) -> TrainingPreset | None:
    """
    Create a preset from a name, either exact match or dynamically generated.

    Examples:
        "rrdbnetx4" -> exact preset if exists
        "rrdbnetx6" -> generated preset for scale 6 based on model defaults
    """
    # Try exact match
    if preset := get_preset(name):
        return preset

    # Try parsing and creating dynamically
    parsed = parse_preset_name(name)
    if parsed is None:
        return None

    model, scale = parsed
    all_presets = get_all_presets()

    # Find a base preset for this model (any scale)
    base_preset = None
    for preset in all_presets.values():
        if preset.model == model:
            base_preset = preset
            break

    if base_preset is None:
        # Try to create from model registry
        try:
            from claritycore.models import AutoConfig

            config = AutoConfig.from_name(model, scale=scale)
            return TrainingPreset(
                model=model,
                scale=scale,
                config=config,
                patch_size=64 * scale,
            )
        except ValueError:
            return None

    # Clone base preset with new scale
    if base_preset.config is not None:
        # Create new config with updated scale
        from dataclasses import replace

        new_config = replace(base_preset.config, scale=scale)
    else:
        new_config = None

    return TrainingPreset(
        model=model,
        scale=scale,
        config=new_config,
        total_steps=base_preset.total_steps,
        batch_size=base_preset.batch_size,
        patch_size=64 * scale,
        lr=base_preset.lr,
        loss=base_preset.loss,
        log_freq=base_preset.log_freq,
        val_freq=base_preset.val_freq,
        save_freq=base_preset.save_freq,
        dataset=base_preset.dataset,
        dataset_paths=list(base_preset.dataset_paths),
        norm=base_preset.norm,
        use_amp=base_preset.use_amp,
        use_ema=base_preset.use_ema,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def find_dataset(preset: TrainingPreset) -> Path | None:
    """Find dataset directory from preset's search paths."""
    for path_str in preset.dataset_paths:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            return path
    return None


def get_dataset_preset(name: str) -> DatasetPreset | None:
    """Get dataset preset by name (case-insensitive)."""
    return DATASET_REGISTRY.get(name.lower())


def detect_dataset_type(path: Path) -> DatasetPreset | None:
    """Try to detect dataset type from path name."""
    name = path.name.lower()

    if preset := DATASET_REGISTRY.get(name):
        return preset

    for key, preset in DATASET_REGISTRY.items():
        if key in name:
            return preset

    return None
