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
from loguru import logger

from claritycore.models import AutoConfig, AutoModel
from claritycore.models.losses import L1Loss, CharbonnierLoss, PerceptualLoss
from claritycore.training import Trainer, TrainingConfig
from claritycore.training.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EMACallback,
    LRSchedulerCallback,
)
from claritycore.data import ImagePairDataset
from claritycore.utils import (
    set_seed,
    print_panel,
    print_success,
    print_error,
    print_info,
    print_rule,
    is_leader,
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

    # Data settings
    data_group = p.add_argument_group("Data")
    data_group.add_argument("--data", type=str, required=True, help="Path to dataset root")
    data_group.add_argument("--hq-dir", type=str, default="hr", help="HQ images subdirectory")
    data_group.add_argument("--lq-dir", type=str, default=None, help="LQ images subdirectory (default: x{scale})")
    data_group.add_argument("--lq-suffix", type=str, default=None, help="LQ filename suffix (e.g., 'x4' for 0001x4.png)")
    data_group.add_argument("--patch-size", type=int, default=256, help="Training patch size")
    data_group.add_argument("--batch-size", type=int, default=8, help="Batch size")
    data_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Training settings
    train_group = p.add_argument_group("Training")
    train_group.add_argument("--steps", type=int, default=100000, help="Total training steps")
    train_group.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_group.add_argument("--loss", type=str, default="l1", choices=["l1", "charbonnier", "perceptual"], help="Loss function")
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

    # Load YAML config if provided
    if args.config:
        config = load_yaml_config(args.config)
        # Override with CLI args (CLI takes precedence)
        for key, value in vars(args).items():
            if value is not None and key != "config":
                # Nested update would go here for complex configs
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
            f"[bold]AMP:[/bold] {args.amp}",
            title="◈ Training Setup",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────────────────────
    data_root = Path(args.data)
    hq_root = data_root / args.hq_dir
    lq_dir = args.lq_dir or f"x{args.scale}"
    lq_root = data_root / lq_dir

    if not hq_root.exists():
        print_error(f"HQ directory not found: {hq_root}")
        sys.exit(1)

    # Auto-detect lq_suffix if not provided
    lq_suffix = args.lq_suffix
    if lq_suffix is None and lq_root.exists():
        # Try to detect suffix by checking first HQ file
        first_hq = next(hq_root.rglob("*.[pP][nN][gG]"), None) or next(hq_root.rglob("*.[jJ][pP][gG]"), None)
        if first_hq:
            # Check if LQ has suffix pattern (e.g., x4)
            test_lq = lq_root / f"{first_hq.stem}x{args.scale}{first_hq.suffix}"
            if test_lq.exists():
                lq_suffix = f"x{args.scale}"
                print_info(f"Auto-detected LQ suffix: '{lq_suffix}'")

    # Check if we have paired LQ data or need self-supervised
    if lq_root.exists():
        print_info(f"Using paired data: HQ={hq_root}, LQ={lq_root}")
        train_dataset = ImagePairDataset(
            hq_root=str(hq_root),
            lq_root=str(lq_root),
            scale=args.scale,
            patch_size=args.patch_size,
            mode="train",
            lq_suffix=lq_suffix,
        )
    else:
        print_info(f"Using self-supervised mode (downscaling from HQ)")
        train_dataset = ImagePairDataset(
            hq_root=str(hq_root),
            lq_root=None,
            scale=args.scale,
            patch_size=args.patch_size,
            mode="train",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation loader (use full images, smaller batch)
    val_dataset = ImagePairDataset(
        hq_root=str(hq_root),
        lq_root=str(lq_root) if lq_root.exists() else None,
        scale=args.scale,
        patch_size=None,  # Full images
        mode="val",
        lq_suffix=lq_suffix,
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
    trainer.add_callback(CheckpointCallback(
        save_dir=output_dir / "checkpoints",
        save_freq=args.save_freq,
        save_best=True,
    ))

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
