#!/usr/bin/env python3
"""Starter wrapper for ORB-SLAM3 benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark wrapper stub for ORB-SLAM3."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. euroc")
    parser.add_argument("--sequence", required=True, help="Sequence identifier")
    parser.add_argument(
        "--mode",
        choices=("mono", "mono-inertial"),
        default="mono",
        help="ORB-SLAM3 operating mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trajectories/orb_slam3"),
        help="Base output directory for converted trajectory files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_spec = {
        "method": "ORB-SLAM3",
        "dataset": args.dataset,
        "sequence": args.sequence,
        "mode": args.mode,
        "status": "stub",
        "next_step": "Replace this wrapper stub with the real ORB-SLAM3 invocation and trajectory conversion.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_spec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
