# Copyright (c) Aman Urumbekov and other contributors.
"""Common utilities for claritycore."""

import os
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from loguru import logger

from claritycore.utils.console import console, get_rich_handler
from claritycore.utils.console import print_banner as _console_print_banner

P = ParamSpec("P")
T = TypeVar("T")


# ---------- rank helpers ----------


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _dist_ready() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def rank_info() -> dict[str, int]:
    """
    Returns a dict with: global_rank, world_size, local_rank, node_rank.
    Works for DDP/FSDP/DeepSpeed (via torch.distributed) or env fallbacks.
    """
    if _dist_ready():
        import torch.distributed as dist

        g = dist.get_rank()
        w = dist.get_world_size()
    else:
        # torchrun/accelerate/deepspeed envs
        g = _env_int("RANK")
        w = _env_int("WORLD_SIZE")
        # SLURM fallbacks
        if g is None:
            g = _env_int("SLURM_PROCID")
        if w is None:
            w = _env_int("SLURM_NTASKS")
        # single-process fallback
        if g is None:
            g = 0
        if w is None:
            w = 1

    local_rank = _env_int("LOCAL_RANK")
    if local_rank is None:
        local_rank = _env_int("SLURM_LOCALID")
    if local_rank is None:
        local_rank = _env_int("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank is None:
        local_rank = 0

    node_rank = _env_int("NODE_RANK")
    if node_rank is None:
        node_rank = _env_int("SLURM_NODEID")
    if node_rank is None:
        node_rank = 0

    return {"global_rank": g, "world_size": w, "local_rank": local_rank, "node_rank": node_rank}


def is_leader() -> bool:
    """Check if current process is the global rank 0 (leader)."""
    return rank_info()["global_rank"] == 0


def leader_only(fn: Callable[P, T]) -> Callable[P, T | None]:
    """Decorator that ensures a function only executes on rank 0."""

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        if is_leader():
            return fn(*args, **kwargs)
        return None

    return wrapper


# ---------- loguru + Rich setup (idempotent) ----------


def setup_logger(
    *,
    only_leader: bool = True,
    level: str = "INFO",
    use_rich: bool = True,
    fmt: str | None = None,
) -> None:
    """
    Configure loguru with Rich integration. Non-leader ranks get a null sink when only_leader=True.
    Safe to call early and multiple times.

    Args:
        only_leader: Only emit logs from global rank 0.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        use_rich: Use Rich for beautiful formatted output (default True).
        fmt: Custom loguru format string (only used when use_rich=False).
    """
    logger.remove()

    info = rank_info()
    _logger = logger.bind(**info)

    if (not only_leader) or is_leader():
        if use_rich:
            # Use Rich handler for beautiful output
            # Format is minimal since Rich handles the formatting
            _logger.add(
                get_rich_handler(level=level.upper()),
                format="{message}",
                level=level.upper(),
                enqueue=True,
            )
        else:
            # Fallback to plain loguru format
            default_fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "G{extra[global_rank]}/N{extra[node_rank]}/L{extra[local_rank]} | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            )
            _logger.add(sys.stderr, level=level.upper(), format=fmt or default_fmt, enqueue=True)
    else:
        # null sink to prevent accidental output on workers
        _logger.add(lambda _: None)

    # expose bound logger globally
    globals()["logger"] = _logger


# ---------- rank-safe convenience ----------

# Wrap console.print for rank-0 only output
print0 = leader_only(console.print)
print0.__doc__ = "Print to console from rank 0 only. Accepts same args as rich.console.print()."


@leader_only
def log0(msg: str, level: str = "INFO", **kwargs: Any) -> None:
    """Log a message from rank 0 only."""
    logger.log(level.upper(), msg, **kwargs)


# Wrap banner for rank-0 only
print_banner = leader_only(_console_print_banner)
print_banner.__doc__ = "Print the ClarityCore banner (rank 0 only)."
