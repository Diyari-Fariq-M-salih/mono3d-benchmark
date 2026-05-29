#!/usr/bin/env python3
"""Prepare ETH3D dataset metadata for benchmark wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ETH3D scene metadata.")
    parser.add_argument("scene_dir", type=Path, help="Path to an ETH3D scene")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "scene_dir": str(args.scene_dir.resolve()),
        "status": "stub",
        "next_step": "Add ETH3D folder validation and ground-truth geometry path discovery.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
