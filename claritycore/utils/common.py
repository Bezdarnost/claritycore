# Copyright (c) Aman Urumbekov and other contributors.
"""Common utilities for claritycore."""

import os
import sys
from typing import Dict

from loguru import logger


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


def rank_info() -> Dict[str, int]:
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

    l = _env_int("LOCAL_RANK")
    if l is None:
        l = _env_int("SLURM_LOCALID")
    if l is None:
        l = _env_int("OMPI_COMM_WORLD_LOCAL_RANK")
    if l is None:
        l = 0

    n = _env_int("NODE_RANK")
    if n is None:
        n = _env_int("SLURM_NODEID")
    if n is None:
        n = 0

    return {"global_rank": g, "world_size": w, "local_rank": l, "node_rank": n}


def is_leader() -> bool:
    return rank_info()["global_rank"] == 0


# ---------- loguru setup (idempotent) ----------


def setup_logger(
    *,
    only_leader: bool = True,
    level: str = "INFO",
    fmt: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "G{extra[global_rank]}/N{extra[node_rank]}/L{extra[local_rank]} | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
) -> None:
    """
    Configure loguru once. Non-leader ranks get a null sink when only_leader=True.
    Safe to call early and multiple times.
    """
    logger.remove()

    info = rank_info()
    _logger = logger.bind(**info)

    if (not only_leader) or is_leader():
        _logger.add(sys.stderr, level=level.upper(), format=fmt, enqueue=True)
    else:
        # null sink to prevent accidental output on workers
        _logger.add(lambda _: None)

    # expose bound logger globally
    globals()["logger"] = _logger


# ---------- rank-safe convenience ----------


def print0(*args, **kwargs) -> None:
    """
    print from global rank 0 only
    recommended to use loguru everywhere
    """
    if is_leader():
        print(*args, **kwargs)


def log0(msg: str, level: str = "INFO") -> None:
    """
    loguru log from global rank 0 only
    """
    if is_leader():
        logger.log(level.upper(), msg)


def print_banner():
    """
    DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    """
    banner = f"""
 ██████╗██╗      █████╗ ██████╗ ██╗████████╗██╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██║     ██╔══██╗██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ██║     ███████║██████╔╝██║   ██║    ╚████╔╝ ██║     ██║   ██║██████╔╝█████╗  
██║     ██║     ██╔══██║██╔══██╗██║   ██║     ╚██╔╝  ██║     ██║   ██║██╔══██╗██╔══╝  
╚██████╗███████╗██║  ██║██║  ██║██║   ██║      ██║   ╚██████╗╚██████╔╝██║  ██║███████╗
 ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
    """

    log0(banner)
