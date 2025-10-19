# Copyright (c) Aman Urumbekov and other contributors.
import os
import sys
import subprocess
import importlib.util
from importlib.metadata import version
from typing import Dict, List, Optional

from claritycore.utils.common import print_banner, print0, setup_logger

# Map `clarity <subcmd>` -> Python module
ROUTE_MAPPING: Dict[str, str] = {
    "train": "claritycore.cli.train",
    # "eval":  "claritycore.cli.eval",
    # "export":"claritycore.cli.export",
}

HELP_TEXT = """\
ClarityCore - Next-generation Open Source toolkit for low-level vision. Engineered for state-of-the-art performance in image and video pixel2pixel tasks, including Super-Resolution, Denoising, Deblurring, and more. 

Usage:
  clarity <command> [args]

Commands:
  train     Train a model.

Global:
  -h, --help        Show this help
  -V, --version     Show version
"""

# ---------- helper functions ----------


def _use_torchrun(subcmd: str) -> bool:
    """Enable torchrun when NPROC_PER_NODE or NNODES is set and command is eligible."""
    if subcmd not in {"train"}:
        return False
    return (os.getenv("NPROC_PER_NODE") is not None) or (os.getenv("NNODES") is not None)


def _torchrun_args() -> List[str]:
    args: List[str] = []
    for key in ("NPROC_PER_NODE", "MASTER_PORT", "NNODES", "NODE_RANK", "MASTER_ADDR"):
        val = os.getenv(key)
        if val:
            args += [f"--{key.lower()}", val]
    return args


def _resolve_module_file(mod: str) -> str:
    """Return the file path of a module, or exit with a helpful error."""
    spec = importlib.util.find_spec(mod)
    if spec is None or spec.origin is None:
        sys.stderr.write(f"[clarity] Cannot locate module: {mod}\n")
        sys.exit(2)
    return spec.origin


def _print_help_and_exit(code: int = 0) -> None:
    print_banner()
    print0(HELP_TEXT)
    sys.exit(code)


def _print_version_and_exit() -> None:
    v = version("claritycore")
    print0(v)
    sys.exit(0)


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    # rank-aware logging early (only leader emits by default)
    setup_logger(only_leader=True, level=os.getenv("CLARITY_LOG_LEVEL", "INFO"))

    mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]

    # no args or help/version
    if not argv or argv[0] in ("-h", "--help"):
        _print_help_and_exit(0)
    if argv[0] in ("-V", "--version"):
        _print_version_and_exit()

    # map subcmd to module path
    subcmd, *rest = argv
    subcmd = subcmd.replace("_", "-")
    path = mapping.get(subcmd)
    if not path:
        sys.stderr.write(f"[clarity] Unknown command: {subcmd}\n\n")
        _print_help_and_exit(2)

    # Print banner once per invocation
    print_banner()

    # build command
    py = sys.executable
    target_file = _resolve_module_file(path)
    if _use_torchrun(subcmd):
        cmd = [py, "-m", "torch.distributed.run", *_torchrun_args(), target_file, *rest]
    else:
        cmd = [py, target_file, *rest]

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        # propagate Ctrl+C as a clean termination
        sys.exit(130)


if __name__ == "__main__":
    cli_main()
