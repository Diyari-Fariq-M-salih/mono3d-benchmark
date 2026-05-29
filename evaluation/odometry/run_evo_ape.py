#!/usr/bin/env python3
"""Run evo_ape with a benchmark-friendly CLI."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an evo_ape command.")
    parser.add_argument("--ref", required=True, type=Path, help="Reference trajectory")
    parser.add_argument("--est", required=True, type=Path, help="Estimated trajectory")
    parser.add_argument(
        "--alignment",
        choices=("none", "se3", "sim3"),
        default="sim3",
        help="Alignment mode to use when invoking evo_ape",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = ["evo_ape", "tum", str(args.ref), str(args.est)]
    if args.alignment == "se3":
        cmd.append("--align")
    elif args.alignment == "sim3":
        cmd.extend(["--align", "--correct_scale"])
    print(shlex.join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
