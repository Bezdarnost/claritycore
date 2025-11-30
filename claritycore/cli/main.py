# Copyright (c) Aman Urumbekov and other contributors.
"""ClarityCore CLI - Simple and powerful training interface."""

import importlib.util
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from claritycore.cli.presets import (
    create_preset_from_name,
    detect_dataset_type,
    find_dataset,
    get_dataset_preset,
    list_presets,
    parse_preset_name,
)
from claritycore.utils import (
    console,
    create_table,
    print_error,
    print_panel,
    print_rule,
    print_warning,
)
from claritycore.utils.common import print0, print_banner, setup_logger

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

HELP_TEXT = """\
[bold cyan]ClarityCore[/] - Next-generation toolkit for image & video restoration.

[bold]Usage:[/]
  [cyan]claritycore train <preset>[/]        Train with preset (e.g., rrdbnetx4)
  [cyan]claritycore train --help[/]          Show advanced training options
  [cyan]claritycore list[/]                  List available presets

[bold]Quick Start Examples:[/]
  [dim]$[/] claritycore train rrdbnetx4           [dim]# Train RRDBNet 4x SR with DIV2K[/]
  [dim]$[/] claritycore train rrdbnetx2           [dim]# Train RRDBNet 2x SR[/]
  [dim]$[/] claritycore train rrdbnet-litex4      [dim]# Train lightweight variant[/]

[bold]Override Preset Settings:[/]
  [dim]$[/] claritycore train rrdbnetx4 --steps 100000 --batch-size 8

[bold]Global Options:[/]
  -h, --help        Show this help
  -V, --version     Show version
"""


def count_images(directory: Path) -> int:
    """Count valid image files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.rglob("*") if f.suffix.lower() in VALID_EXTENSIONS)


def validate_dataset(
    data_path: Path,
    scale: int,
    dataset_name: str | None = None,
) -> tuple[bool, int, str | None]:
    """
    Validate dataset and return (is_valid, image_count, warning_message).
    """
    # Detect dataset type
    dataset_preset = None
    if dataset_name:
        dataset_preset = get_dataset_preset(dataset_name)
    if dataset_preset is None:
        dataset_preset = detect_dataset_type(data_path)

    # Find target directory
    target_dir = data_path / "hr"
    if not target_dir.exists():
        # Try other common names
        for name in ["HR", "gt", "GT", "target", "hq", "HQ"]:
            alt = data_path / name
            if alt.exists():
                target_dir = alt
                break

    if not target_dir.exists():
        return False, 0, None

    # Count images
    image_count = count_images(target_dir)

    if image_count == 0:
        return False, 0, None

    # Check against expected size
    warning = None
    if dataset_preset:
        expected = dataset_preset.expected_train_size
        if image_count != expected:
            diff = image_count - expected
            sign = "+" if diff > 0 else ""
            warning = f"Expected {expected:,} images for {dataset_preset.name}, found {image_count:,} ({sign}{diff:,})"

    return True, image_count, warning


def print_preset_info(preset, data_path: Path, image_count: int):
    """Print pretty information about the training configuration."""
    # Model info - build dynamically from config
    model_lines = [
        f"[bold]Architecture:[/] {preset.model.upper()}",
        f"[bold]Scale:[/] {preset.scale}x",
    ]
    # Add model-specific params
    for key, value in preset.get_arch_params().items():
        # Format key: num_feat -> Feat, num_block -> Block
        label = key.replace("_", " ").title().replace("Num ", "")
        model_lines.append(f"[bold]{label}:[/] {value}")
    model_info = "\n".join(model_lines)

    # Training info
    train_info = (
        f"[bold]Steps:[/] {preset.total_steps:,}\n"
        f"[bold]Batch Size:[/] {preset.batch_size}\n"
        f"[bold]Patch Size:[/] {preset.patch_size}×{preset.patch_size}\n"
        f"[bold]Learning Rate:[/] {preset.lr}\n"
        f"[bold]Loss:[/] {preset.loss.upper()}\n"
        f"[bold]AMP:[/] {'✓' if preset.use_amp else '✗'}\n"
        f"[bold]EMA:[/] {'✓' if preset.use_ema else '✗'}"
    )

    # Data info
    input_subdir = f"x{preset.scale}"
    data_info = (
        f"[bold]Dataset:[/] {data_path}\n"
        f"[bold]Target:[/] hr/\n"
        f"[bold]Input:[/] {input_subdir}/\n"
        f"[bold]Images:[/] {image_count:,}\n"
        f"[bold]Normalization:[/] {preset.norm}"
    )

    # Create table
    table = create_table(title="◈ Training Configuration", show_header=False)
    table.add_column("Section", style="cyan")
    table.add_column("Details")
    table.add_row("[bold cyan]Model[/]", model_info)
    table.add_row("[bold cyan]Training[/]", train_info)
    table.add_row("[bold cyan]Data[/]", data_info)

    console.print(table)


def print_available_presets():
    """Print available training presets."""
    from claritycore.cli.presets import get_all_presets

    all_presets = get_all_presets()

    table = create_table(title="Available Training Presets")
    table.add_column("Preset", style="cyan bold")
    table.add_column("Model")
    table.add_column("Scale")
    table.add_column("Config", overflow="fold")
    table.add_column("Steps")

    for name, preset in sorted(all_presets.items()):
        # Format model config as compact string
        arch_params = preset.get_arch_params()
        config_str = ", ".join(f"{k}={v}" for k, v in arch_params.items())
        table.add_row(
            name,
            preset.model,
            f"{preset.scale}x",
            config_str or "-",
            f"{preset.total_steps:,}",
        )

    console.print(table)
    console.print()
    console.print("[bold]Usage:[/]")
    console.print("  [cyan]claritycore train <preset>[/]  [dim]# Start training with preset[/]")
    console.print()
    console.print("[dim]Tip: You can also use custom scales like 'rrdbnetx6' or 'rrdbnetx16'[/]")


def run_preset_training(preset_name: str, extra_args: list[str]):
    """Run training with a preset configuration."""
    # Parse preset
    preset = create_preset_from_name(preset_name)
    if preset is None:
        print_error(f"Unknown preset: [bold]{preset_name}[/]")
        console.print()
        console.print("[dim]Available presets:[/]")
        for name in list_presets():
            console.print(f"  • {name}")
        console.print()
        console.print("[dim]Or use pattern like 'modelx4' (e.g., rrdbnetx4, hatx2)[/]")
        sys.exit(1)

    # Find dataset
    data_path = find_dataset(preset)
    if data_path is None:
        print_error("Dataset not found!")
        console.print()
        print_panel(
            f"[bold]Expected dataset for {preset.dataset.upper()}[/]\n\n"
            f"Searched paths:\n"
            + "\n".join(f"  • {p}" for p in preset.dataset_paths)
            + "\n\n[bold]Required structure:[/]\n"
            f"  {preset.dataset_paths[0]}/\n"
            f"    ├── hr/           [dim]# Target (high-res) images[/]\n"
            f"    └── x{preset.scale}/          [dim]# Input (low-res) images[/]\n\n"
            f"[dim]Download DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/[/]",
            title="◈ Dataset Setup",
            border_style="red",
        )
        sys.exit(1)

    # Validate dataset
    is_valid, image_count, warning = validate_dataset(
        data_path,
        preset.scale,
        preset.dataset,
    )

    if not is_valid:
        print_error(f"No images found in dataset: {data_path}")
        console.print()
        console.print(f"[dim]Looking for images in: {data_path}/hr/[/]")
        sys.exit(1)

    # Print configuration
    print_rule("Training")
    print_preset_info(preset, data_path, image_count)

    # Show warning if image count doesn't match
    if warning:
        console.print()
        print_warning(warning)

    console.print()

    # Build training command
    train_args = [
        "--model",
        preset.model,
        "--scale",
        str(preset.scale),
    ]

    # Add model-specific config args
    train_args.extend(preset.get_cli_args())

    # Add common training args
    train_args.extend(
        [
            "--data",
            str(data_path),
            "--patch-size",
            str(preset.patch_size),
            "--batch-size",
            str(preset.batch_size),
            "--steps",
            str(preset.total_steps),
            "--lr",
            str(preset.lr),
            "--loss",
            preset.loss,
            "--log-freq",
            str(preset.log_freq),
            "--val-freq",
            str(preset.val_freq),
            "--save-freq",
            str(preset.save_freq),
            "--norm",
            preset.norm,
            "--name",
            preset_name,
        ]
    )

    if preset.use_amp:
        train_args.append("--amp")
    if preset.use_ema:
        train_args.append("--ema")

    # Add extra args from command line (these override preset values)
    train_args.extend(extra_args)

    # Run training
    py = sys.executable
    train_module = importlib.util.find_spec("claritycore.cli.train")
    if train_module is None or train_module.origin is None:
        print_error("Cannot locate training module")
        sys.exit(2)

    cmd = [py, train_module.origin] + train_args

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


def _print_help_and_exit(code: int = 0) -> None:
    print_banner()
    console.print(HELP_TEXT)
    sys.exit(code)


def _print_version_and_exit() -> None:
    v = version("claritycore")
    print0(v)
    sys.exit(0)


def cli_main() -> None:
    """Main CLI entry point."""
    setup_logger(only_leader=True, level=os.getenv("CLARITY_LOG_LEVEL", "INFO"))

    argv = sys.argv[1:]

    # No args or help
    if not argv or argv[0] in ("-h", "--help"):
        _print_help_and_exit(0)

    if argv[0] in ("-V", "--version"):
        _print_version_and_exit()

    cmd = argv[0].lower()
    rest = argv[1:]

    # List presets command
    if cmd == "list":
        print_banner()
        print_available_presets()
        sys.exit(0)

    # Train command
    if cmd == "train":
        print_banner()

        if not rest or rest[0] in ("-h", "--help"):
            # Show advanced training help
            py = sys.executable
            train_module = importlib.util.find_spec("claritycore.cli.train")
            if train_module is None or train_module.origin is None:
                print_error("Cannot locate training module")
                sys.exit(2)

            cmd_list = [py, train_module.origin, "--help"]
            try:
                result = subprocess.run(cmd_list)
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                sys.exit(130)

        # Check if first arg is a preset or a flag
        first_arg = rest[0]

        # If it starts with "-", it's advanced mode with flags
        if first_arg.startswith("-"):
            # Advanced training mode - delegate to train.py
            py = sys.executable
            train_module = importlib.util.find_spec("claritycore.cli.train")
            if train_module is None or train_module.origin is None:
                print_error("Cannot locate training module")
                sys.exit(2)

            cmd_list = [py, train_module.origin] + rest
            try:
                result = subprocess.run(cmd_list)
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                sys.exit(130)

        # Otherwise, first arg should be a preset name
        preset_name = first_arg
        extra_args = rest[1:]

        # Check if it's a valid preset
        if parse_preset_name(preset_name) is not None or preset_name in list_presets():
            run_preset_training(preset_name, extra_args)
        else:
            print_error(f"Unknown preset: [bold]{preset_name}[/]")
            console.print()
            console.print("[dim]Use 'claritycore list' to see available presets[/]")
            console.print("[dim]Use 'claritycore train --help' for advanced options[/]")
            sys.exit(1)

    # Unknown command
    else:
        print_error(f"Unknown command: [bold]{cmd}[/]")
        console.print()
        console.print("[bold]Available commands:[/]")
        console.print("  [cyan]train[/]   Train a model")
        console.print("  [cyan]list[/]    List available presets")
        console.print()
        console.print("[dim]Use 'claritycore --help' for more information[/]")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
