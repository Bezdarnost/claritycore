# Copyright (c) Aman Urumbekov and other contributors.
import os
import argparse
from typing import Any, Dict, List

from claritycore.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("clarity: train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-c", "--config", required=False, help="Path to YAML config")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config items: key=value (e.g. train.lr=1e-4 data.train.root=/data)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed()


if __name__ == "__main__":
    main()
