#!/usr/bin/env python3
"""Run SVO benchmark automation for EuRoC sequences."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wrapper stub for SVO.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. euroc")
    parser.add_argument("--sequence", required=True, help="Sequence identifier")
    parser.add_argument(
        "--mode",
        choices=("mono", "mono-inertial", "both"),
        default="mono",
        help="SVO operating mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trajectories/svo"),
        help="Base output directory for converted trajectory files",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate SVO benchmark datasets and configs, do not run the batch script.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    prepare_cmd = ["python3", str(repo_root / "scripts" / "prepare_svo_euroc.py"), "--write-configs", "--sequence", args.sequence]
    subprocess.run(prepare_cmd, check=True)

    if args.prepare_only:
        return 0

    batch_mode = "mono-imu" if args.mode == "mono-inertial" else args.mode
    run_cmd = [
        "bash",
        str(repo_root / "scripts" / "run_svo_euroc_batch.sh"),
        "--mode",
        batch_mode,
        "--sequence",
        args.sequence,
    ]
    subprocess.run(run_cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
