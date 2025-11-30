# Copyright (c) Aman Urumbekov and other contributors.
"""Training callbacks for ClarityCore."""

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from claritycore.utils import is_leader, print_success

if TYPE_CHECKING:
    from claritycore.training.trainer import Trainer


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks allow custom behavior at various points in the training loop.
    Override the methods you need.
    """

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: "Trainer") -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, trainer: "Trainer") -> None:
        """Called before each training step."""
        pass

    def on_step_end(self, trainer: "Trainer", loss_dict: dict[str, float]) -> None:
        """Called after each training step."""
        pass

    def on_validation_start(self, trainer: "Trainer") -> None:
        """Called before validation."""
        pass

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        """Called after validation."""
        pass


class LoggingCallback(Callback):
    """
    Callback for logging training progress.

    Args:
        log_freq: Log every N steps.
    """

    def __init__(self, log_freq: int = 100) -> None:
        self.log_freq = log_freq
        self._loss_accum: dict[str, float] = {}
        self._accum_count: int = 0

    def on_step_end(self, trainer: "Trainer", loss_dict: dict[str, float]) -> None:
        # Accumulate losses
        for k, v in loss_dict.items():
            self._loss_accum[k] = self._loss_accum.get(k, 0) + v
        self._accum_count += 1

        # Log periodically
        if trainer.global_step % self.log_freq == 0 and is_leader():
            avg_losses = {k: v / self._accum_count for k, v in self._loss_accum.items()}
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())

            lr = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else 0

            logger.info(
                f"[cyan]Step {trainer.global_step:,}[/] | Epoch {trainer.current_epoch} | lr: {lr:.2e} | {loss_str}",
            )

            # Reset accumulator
            self._loss_accum = {}
            self._accum_count = 0


class CheckpointCallback(Callback):
    """
    Callback for saving checkpoints.

    Args:
        save_dir: Directory to save checkpoints.
        save_freq: Save every N steps.
        save_best: Whether to save best model based on validation.
        metric_name: Metric to track for best model.
        higher_is_better: Whether higher metric is better.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_freq: int = 5000,
        save_best: bool = True,
        metric_name: str = "psnr",
        higher_is_better: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best_metric: float | None = None

    def on_train_start(self, trainer: "Trainer") -> None:
        if is_leader():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: "Trainer", loss_dict: dict[str, float]) -> None:
        if trainer.global_step % self.save_freq == 0 and is_leader():
            self._save_checkpoint(trainer, "latest")
            self._save_checkpoint(trainer, f"step_{trainer.global_step}")

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        if not self.save_best or not is_leader():
            return

        current = metrics.get(self.metric_name)
        if current is None:
            return

        is_best = (
            self.best_metric is None
            or (self.higher_is_better and current > self.best_metric)
            or (not self.higher_is_better and current < self.best_metric)
        )

        if is_best:
            self.best_metric = current
            self._save_checkpoint(trainer, "best")
            print_success(f"New best {self.metric_name}: {current:.4f}")

    def _save_checkpoint(self, trainer: "Trainer", name: str) -> None:
        path = self.save_dir / f"{name}.pt"

        checkpoint = {
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "best_metric": self.best_metric,
        }

        if trainer.optimizer:
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

        if trainer.scheduler:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")


class EMACallback(Callback):
    """
    Callback for updating EMA weights.

    Args:
        decay: EMA decay rate.
        update_freq: Update EMA every N steps.
    """

    def __init__(self, decay: float = 0.999, update_freq: int = 1) -> None:
        self.decay = decay
        self.update_freq = update_freq

    def on_train_start(self, trainer: "Trainer") -> None:
        if not trainer.model.use_ema:
            trainer.model.init_ema(self.decay)

    def on_step_end(self, trainer: "Trainer", loss_dict: dict[str, float]) -> None:
        if trainer.global_step % self.update_freq == 0:
            trainer.model.update_ema()


class LRSchedulerCallback(Callback):
    """
    Callback for learning rate scheduling.

    Wraps any PyTorch scheduler.

    Args:
        scheduler: Learning rate scheduler.
        step_on: When to step ('step', 'epoch', 'validation').
    """

    def __init__(
        self,
        step_on: str = "step",
    ) -> None:
        self.step_on = step_on

    def on_step_end(self, trainer: "Trainer", loss_dict: dict[str, float]) -> None:
        if self.step_on == "step" and trainer.scheduler:
            trainer.scheduler.step()

    def on_epoch_end(self, trainer: "Trainer") -> None:
        if self.step_on == "epoch" and trainer.scheduler:
            trainer.scheduler.step()

    def on_validation_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        if self.step_on == "validation" and trainer.scheduler:
            # For ReduceLROnPlateau, pass metric
            if hasattr(trainer.scheduler, "step"):
                try:
                    trainer.scheduler.step(metrics.get("psnr", 0))
                except TypeError:
                    trainer.scheduler.step()


__all__ = [
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "EMACallback",
    "LRSchedulerCallback",
]
