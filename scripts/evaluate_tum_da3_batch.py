#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = (
    REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "hall_da3_large_matrix" / "runs"
)
DEFAULT_PREPARED_ROOT = REPO_ROOT / "datasets" / "tum_rgbd_prepared"
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_tum_da3.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate DA3 TUM runs against TUM trajectory and point-cloud ground truth."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing DA3 run folders.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=DEFAULT_PREPARED_ROOT,
        help="Directory containing prepared TUM frame folders.",
    )
    parser.add_argument(
        "--with-pointcloud",
        action="store_true",
        help="Also run point-cloud evaluation for each run.",
    )
    parser.add_argument(
        "--max-assoc-delta",
        type=float,
        default=0.02,
        help="Maximum timestamp association delta passed through to evaluate_tum_da3.py.",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=5.0,
        help="Maximum GT depth passed through to evaluate_tum_da3.py.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
        help="GT point-cloud pixel stride passed through to evaluate_tum_da3.py.",
    )
    parser.add_argument(
        "--cloud-sample",
        type=int,
        default=200000,
        help="Point-cloud sample size passed through to evaluate_tum_da3.py.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def infer_prepared_dir_name(run_name: str) -> str | None:
    if "__fps" not in run_name or "__chunk" not in run_name:
        return None
    sequence = run_name.split("__fps", 1)[0]
    fps_tail = run_name.split("__fps", 1)[1]
    fps = fps_tail.split("__", 1)[0]
    return f"{sequence}_fps{fps}"


def main() -> int:
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    prepared_root = resolve_path(args.prepared_root)
    eval_script = DEFAULT_EVAL_SCRIPT

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if not run_dirs:
        print(f"no run folders found in {runs_root}", file=sys.stderr)
        return 2

    failures = []
    evaluated = 0

    for run_dir in run_dirs:
        prepared_name = infer_prepared_dir_name(run_dir.name)
        if prepared_name is None:
            print(f"[skip] unrecognized run naming pattern: {run_dir.name}")
            continue

        prepared_dir = prepared_root / prepared_name
        if not prepared_dir.exists():
            print(f"[skip] missing prepared dir for {run_dir.name}: {prepared_dir}")
            continue

        command = [
            "python3",
            str(eval_script),
            "--run-dir",
            str(run_dir),
            "--prepared-dir",
            str(prepared_dir),
            "--max-assoc-delta",
            str(args.max_assoc_delta),
            "--depth-max-m",
            str(args.depth_max_m),
            "--pixel-stride",
            str(args.pixel_stride),
            "--cloud-sample",
            str(args.cloud_sample),
        ]
        if args.with_pointcloud:
            command.append("--with-pointcloud")

        print(f"[eval] {run_dir.name}")
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode != 0:
            failures.append(run_dir.name)
        else:
            evaluated += 1

    print(f"[done] evaluated {evaluated} runs")
    if failures:
        print(f"[fail] {len(failures)} runs failed")
        for name in failures:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
