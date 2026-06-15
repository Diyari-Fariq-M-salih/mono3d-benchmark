#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks_isolated"
DEFAULT_SEQUENCE = "V1_01_easy"
DEFAULT_TAG = "isolated_v1_01"
DEFAULT_DA3_RUN = (
    REPO_ROOT
    / "outputs"
    / "reconstructions"
    / "depth_anything_3"
    / "euroc"
    / DEFAULT_SEQUENCE
    / "20260610T132107Z_fps5_da3_streaming"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "euroc_pose_swap"
VALIDATOR = REPO_ROOT / "scripts" / "validate_euroc_external_pose_reconstruction.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use the latest isolated SVO EuRoC run as external poses for DA3 reconstruction.")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--mode", choices=("mono", "mono-imu"), default="mono-imu")
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--da3-run-dir", type=Path, default=DEFAULT_DA3_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence-dir", type=Path, help="Optional EuRoC sequence dir override.")
    parser.add_argument("--association-mode", choices=("nearest", "interpolate"), default="interpolate")
    parser.add_argument("--allow-partial-overlap", action="store_true", default=True)
    parser.add_argument("--no-allow-partial-overlap", dest="allow_partial_overlap", action="store_false")
    parser.add_argument("--export-ply", action="store_true", default=True)
    parser.add_argument("--with-pointcloud", action="store_true", default=True)
    parser.add_argument("--cloud-sample", type=int, default=200000)
    return parser.parse_args()


def latest_trace_dir(root: Path, experiment_name: str, sequence: str) -> Path:
    exp_root = root / experiment_name
    candidates = [path for path in exp_root.iterdir() if path.is_dir()] if exp_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No isolated SVO runs found under {exp_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    trace_dir = latest / sequence
    if not trace_dir.exists():
        raise FileNotFoundError(f"Expected sequence trace not found: {trace_dir}")
    return trace_dir


def main() -> int:
    args = parse_args()
    exp_name = f"mono3d_euroc_{args.tag}_{'mono_imu' if args.mode == 'mono-imu' else 'mono'}"
    trace_dir = latest_trace_dir(args.trace_root.resolve(), exp_name, args.sequence)
    external_traj = trace_dir / "stamped_traj_estimate.txt"
    if not external_traj.exists():
        raise FileNotFoundError(f"Missing isolated SVO trajectory: {external_traj}")

    sequence_dir = (
        args.sequence_dir.resolve()
        if args.sequence_dir
        else (REPO_ROOT / "datasets" / "euroc" / args.sequence).resolve()
    )
    output_dir = (args.output_root.resolve() / f"{args.sequence}_{args.tag}_{'mono_imu' if args.mode == 'mono-imu' else 'mono'}")
    command = [
        "python3",
        str(VALIDATOR),
        "--da3-run-dir",
        str(args.da3_run_dir.resolve()),
        "--external-traj",
        str(external_traj),
        "--sequence-dir",
        str(sequence_dir),
        "--output-dir",
        str(output_dir),
        "--association-mode",
        args.association_mode,
        "--cloud-sample",
        str(args.cloud_sample),
    ]
    if args.allow_partial_overlap:
        command.append("--allow-partial-overlap")
    if args.export_ply:
        command.append("--export-ply")
    if args.with_pointcloud:
        command.append("--with-pointcloud")

    child_env = dict(os.environ)
    child_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_env)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
