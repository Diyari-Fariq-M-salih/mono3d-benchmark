#!/usr/bin/env python3
"""Prepare EuRoC dataset metadata for benchmark wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EuRoC sequence metadata.")
    parser.add_argument("sequence_dir", type=Path, help="Path to a EuRoC sequence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "sequence_dir": str(args.sequence_dir.resolve()),
        "status": "stub",
        "next_step": "Add EuRoC folder validation and export normalized image/IMU paths.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
