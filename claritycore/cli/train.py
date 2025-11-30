#!/usr/bin/env python3
# Copyright (c) Aman Urumbekov and other contributors.
"""
ClarityCore Training CLI.

Usage:
    clarity train -c config.yaml
    clarity train --model rrdbnet --data datasets/DIV2K --scale 4
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from claritycore.data import DatasetConfig, NormConfig, Pixel2PixelDataset
from claritycore.models import AutoConfig, AutoModel
from claritycore.models.losses import CharbonnierLoss, L1Loss, PerceptualLoss
from claritycore.training import Trainer, TrainingConfig
from claritycore.training.callbacks import (
    CheckpointCallback,
    EMACallback,
    LoggingCallback,
    LRSchedulerCallback,
)
from claritycore.utils import (
    is_leader,
    print_error,
    print_info,
    print_panel,
    print_rule,
    print_success,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="clarity train",
        description="Train a ClarityCore model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (optional - can use CLI args instead)
    p.add_argument("-c", "--config", type=str, help="Path to YAML config file")

    # Model settings
    model_group = p.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="rrdbnet", help="Model type")
    model_group.add_argument("--scale", type=int, default=4, help="Upscaling factor")
    model_group.add_argument("--num-feat", type=int, default=64, help="Feature channels")
    model_group.add_argument("--num-block", type=int, default=23, help="Number of blocks")
    model_group.add_argument("--num-grow-ch", type=int, default=32, help="Growth channels (RRDBNet)")

    # Data settings
    data_group = p.add_argument_group("Data")
    data_group.add_argument("--data", type=str, required=True, help="Path to dataset root")
    data_group.add_argument("--target-dir", type=str, default="hr", help="Target (HQ) images subdirectory")
    data_group.add_argument(
        "--input-dir", type=str, default=None, help="Input (LQ) images subdirectory (default: x{scale})"
    )
    data_group.add_argument(
        "--input-suffix", type=str, default=None, help="Input filename suffix (auto-detected if not set)"
    )
    data_group.add_argument("--patch-size", type=int, default=256, help="Training patch size")
    data_group.add_argument("--batch-size", type=int, default=8, help="Batch size")
    data_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    data_group.add_argument(
        "--norm",
        type=str,
        default="minus_one_one",
        choices=["zero_one", "minus_one_one", "none"],
        help="Normalization mode",
    )

    # Training settings
    train_group = p.add_argument_group("Training")
    train_group.add_argument("--steps", type=int, default=100000, help="Total training steps")
    train_group.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_group.add_argument(
        "--loss", type=str, default="l1", choices=["l1", "charbonnier", "perceptual"], help="Loss function"
    )
    train_group.add_argument("--val-freq", type=int, default=5000, help="Validation frequency")
    train_group.add_argument("--log-freq", type=int, default=100, help="Logging frequency")
    train_group.add_argument("--save-freq", type=int, default=5000, help="Checkpoint save frequency")
    train_group.add_argument("--ema", action="store_true", help="Use EMA")
    train_group.add_argument("--amp", action="store_true", help="Use mixed precision")

    # Output settings
    out_group = p.add_argument_group("Output")
    out_group.add_argument("--output", type=str, default="experiments", help="Output directory")
    out_group.add_argument("--name", type=str, default=None, help="Experiment name")

    # Other
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    return p.parse_args()


def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def get_loss(name: str) -> torch.nn.Module:
    """Get loss function by name."""
    if name == "l1":
        return L1Loss()
    elif name == "charbonnier":
        return CharbonnierLoss()
    elif name == "perceptual":
        return PerceptualLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


def main():
    args = parse_args()

    # Load YAML config if provided (TODO: implement config merging)
    if args.config:
        _config = load_yaml_config(args.config)  # noqa: F841
        # Override with CLI args (CLI takes precedence)
        for key, value in vars(args).items():
            if value is not None and key != "config":
                pass

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_leader():
        print_rule("Configuration")
        print_panel(
            f"[bold]Model:[/bold] {args.model}\n"
            f"[bold]Scale:[/bold] {args.scale}x\n"
            f"[bold]Device:[/bold] {device}\n"
            f"[bold]Data:[/bold] {args.data}\n"
            f"[bold]Steps:[/bold] {args.steps:,}\n"
            f"[bold]Batch:[/bold] {args.batch_size}\n"
            f"[bold]LR:[/bold] {args.lr}\n"
            f"[bold]Loss:[/bold] {args.loss}\n"
            f"[bold]Norm:[/bold] {args.norm}\n"
            f"[bold]AMP:[/bold] {args.amp}",
            title="◈ Training Setup",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────────────────────
    data_root = Path(args.data)
    target_dir = data_root / args.target_dir

    # Determine input directory
    input_dir_name = args.input_dir or f"x{args.scale}"
    input_dir = data_root / input_dir_name

    if not target_dir.exists():
        print_error(f"Target directory not found: {target_dir}")
        sys.exit(1)

    # Create normalization config
    norm_config = NormConfig(mode=args.norm)

    # Create dataset config
    train_data_config = DatasetConfig(
        target_dir=str(target_dir),
        input_dir=str(input_dir) if input_dir.exists() else None,
        scale=args.scale,
        input_suffix=args.input_suffix,
        patch_size=args.patch_size,
        augment=True,
        normalize=norm_config,
    )

    val_data_config = DatasetConfig(
        target_dir=str(target_dir),
        input_dir=str(input_dir) if input_dir.exists() else None,
        scale=args.scale,
        input_suffix=args.input_suffix,
        patch_size=None,  # Full images for validation
        augment=False,
        normalize=norm_config,
    )

    # Create datasets
    if input_dir.exists():
        print_info(f"Paired data: target={target_dir}, input={input_dir}")
    else:
        print_info("Self-supervised mode: generating input from target")

    train_dataset = Pixel2PixelDataset(train_data_config, mode="train")
    val_dataset = Pixel2PixelDataset(val_data_config, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    if is_leader():
        print_success(f"Train: {len(train_dataset)} images")
        print_success(f"Val: {len(val_dataset)} images")

    # ─────────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────────
    print_rule("Model")

    model_config = AutoConfig.from_name(
        args.model,
        scale=args.scale,
        num_feat=args.num_feat,
        num_block=args.num_block,
        num_grow_ch=args.num_grow_ch,
    )

    model = AutoModel.from_config(model_config)
    model = model.to(device)

    # Add loss
    loss_fn = get_loss(args.loss)
    model.add_loss(args.loss, loss_fn, weight=1.0)

    if is_leader():
        print_success(f"Model: {model}")
        print_success(f"Parameters: {model.count_parameters():,}")

    # ─────────────────────────────────────────────────────────────────────────
    # Optimizer & Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=1e-7,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    print_rule("Training")

    exp_name = args.name or f"{args.model}_x{args.scale}"
    output_dir = Path(args.output) / exp_name

    training_config = TrainingConfig(
        total_steps=args.steps,
        val_freq=args.val_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        mixed_precision=args.amp and torch.cuda.is_available(),
        output_dir=args.output,
        experiment_name=exp_name,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
    )

    # Add callbacks
    trainer.add_callback(LoggingCallback(log_freq=args.log_freq))
    trainer.add_callback(
        CheckpointCallback(
            save_dir=output_dir / "checkpoints",
            save_freq=args.save_freq,
            save_best=True,
        )
    )

    if args.ema:
        trainer.add_callback(EMACallback(decay=0.999))

    trainer.add_callback(LRSchedulerCallback(step_on="step"))

    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)

    # Start training
    trainer.train()

    if is_leader():
        print_success(f"Training complete! Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
