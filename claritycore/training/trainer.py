# Copyright (c) Aman Urumbekov and other contributors.
"""Main trainer class for ClarityCore."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from claritycore.models.base import BaseModel
from claritycore.training.callbacks import Callback
from claritycore.utils import (
    create_training_progress,
    is_leader,
    print_metrics,
    print_panel,
    print_rule,
    print_success,
)


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Args:
        total_steps: Total number of training steps.
        val_freq: Validation frequency (in steps).
        log_freq: Logging frequency (in steps).
        save_freq: Checkpoint save frequency (in steps).
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping.
        mixed_precision: Whether to use mixed precision training.
        compile_model: Whether to use torch.compile (PyTorch 2.0+).
    """

    total_steps: int = 100000
    val_freq: int = 5000
    log_freq: int = 100
    save_freq: int = 5000

    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = None
    mixed_precision: bool = False
    compile_model: bool = False

    # Output
    output_dir: str = "experiments"
    experiment_name: str = "default"


class Trainer:
    """
    Main trainer class for ClarityCore models.

    Handles:
    - Training loop with progress tracking
    - Validation
    - Gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpointing (via callbacks)
    - Logging (via callbacks)

    Usage:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            config=config,
        )
        trainer.add_callback(LoggingCallback(log_freq=100))
        trainer.add_callback(CheckpointCallback(save_dir="checkpoints"))
        trainer.train()
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig | None = None,
        val_loader: DataLoader | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        val_fn: Callable[[BaseModel, DataLoader], dict[str, float]] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        self.val_fn = val_fn

        # Training state
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.start_time: float = 0.0

        # Callbacks
        self.callbacks: list[Callback] = []

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.config.mixed_precision else None

        # Compile model if requested
        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile")

    def add_callback(self, callback: Callback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def _run_callbacks(self, method: str, *args, **kwargs) -> None:
        """Run a callback method on all registered callbacks."""
        for callback in self.callbacks:
            getattr(callback, method)(self, *args, **kwargs)

    def train(self) -> None:
        """Run the training loop."""
        self._run_callbacks("on_train_start")
        self.start_time = time.time()
        self.model.train()

        if is_leader():
            print_rule("Training Started")
            print_panel(
                f"[bold]Total steps:[/bold] {self.config.total_steps:,}\n"
                f"[bold]Batch size:[/bold] {self.train_loader.batch_size}\n"
                f"[bold]Val freq:[/bold] {self.config.val_freq}\n"
                f"[bold]Mixed precision:[/bold] {self.config.mixed_precision}",
                title="◈ Training Config",
            )

        data_iter = iter(self.train_loader)

        with create_training_progress() as progress:
            task = progress.add_task(
                "Training",
                total=self.config.total_steps,
                visible=is_leader(),
            )
            progress.update(task, completed=self.global_step)

            while self.global_step < self.config.total_steps:
                # Get batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.current_epoch += 1
                    self._run_callbacks("on_epoch_start")
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                self._run_callbacks("on_step_start")

                # Training step
                loss_dict = self._training_step(batch)

                self.global_step += 1
                progress.update(task, advance=1)

                self._run_callbacks("on_step_end", loss_dict)

                # Validation
                if (
                    self.val_loader is not None
                    and self.global_step % self.config.val_freq == 0
                ):
                    self._validate()

                self._run_callbacks("on_epoch_end")

        self._run_callbacks("on_train_end")

        if is_leader():
            elapsed = time.time() - self.start_time
            print_rule("Training Complete")
            print_success(f"Finished {self.config.total_steps:,} steps in {elapsed/3600:.2f} hours")

    def _training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step."""
        device = self.model.device

        # Move batch to device
        lq = batch["lq"].to(device)
        gt = batch["gt"].to(device)

        # Forward pass with optional mixed precision
        with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
            output = self.model(lq)
            loss, loss_dict = self.model.compute_loss(output, gt)

            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if self.global_step % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return loss_dict

    def _validate(self) -> None:
        """Run validation."""
        self._run_callbacks("on_validation_start")
        self.model.eval()

        if self.val_fn is not None:
            metrics = self.val_fn(self.model, self.val_loader)
        else:
            metrics = self._default_validation()

        self.model.train()

        if is_leader():
            print_rule(f"Validation @ Step {self.global_step:,}")
            print_metrics(metrics)

        self._run_callbacks("on_validation_end", metrics)

    @torch.no_grad()
    def _default_validation(self) -> dict[str, float]:
        """Default validation using PSNR/SSIM."""
        from claritycore.metrics import psnr, ssim

        psnr_sum, ssim_sum, count = 0.0, 0.0, 0
        device = self.model.device

        for batch in self.val_loader:
            lq = batch["lq"].to(device)
            gt = batch["gt"].to(device)

            # Inference
            if hasattr(self.model, "inference"):
                output = self.model.inference(lq)
            else:
                output = self.model(lq)

            output = output.clamp(0, 1)
            gt = gt.clamp(0, 1)

            # Compute metrics
            psnr_sum += psnr(output, gt).sum().item()
            ssim_sum += ssim(output, gt).sum().item()
            count += output.shape[0]

        return {
            "psnr": psnr_sum / count if count > 0 else 0,
            "ssim": ssim_sum / count if count > 0 else 0,
        }

    def resume(self, checkpoint_path: str) -> None:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)

        if is_leader():
            print_success(f"Resumed from step {self.global_step:,}")


__all__ = ["Trainer", "TrainingConfig"]

