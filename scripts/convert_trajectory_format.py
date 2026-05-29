#!/usr/bin/env python3
"""Convert trajectory text files into the benchmark standard format."""

from __future__ import annotations

import argparse
from pathlib import Path


HEADER = "timestamp tx ty tz qx qy qz qw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a trajectory to TUM-style format.")
    parser.add_argument("input_path", type=Path, help="Source trajectory file")
    parser.add_argument("output_path", type=Path, help="Converted trajectory file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {HEADER}\n")
    print(f"created stub output at {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
