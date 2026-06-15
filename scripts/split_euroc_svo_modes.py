#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "reports" / "evaluation" / "svo_full11_euroc_mono_vs_mono_imu_20260601"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split combined EuRoC SVO mono/mono+IMU summary into per-mode folders.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_single_bars(sequences: list[str], values: list[float], ylabel: str, title: str, out_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(sequences) * 1.15), 5.8))
    indices = list(range(len(sequences)))
    ax.bar(indices, values, color=color)
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_mode_folder(rows: list[dict[str, str]], output_dir: Path, mode_name: str, color: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "summary.csv")

    sequences = [row["sequence"] for row in rows]
    plot_single_bars(
        sequences,
        [to_float(row["tracking_ratio"]) or 0.0 for row in rows],
        "Tracking Ratio",
        f"EuRoC SVO {mode_name}: Tracking Ratio",
        output_dir / "tracking_ratio.png",
        color,
    )
    plot_single_bars(
        sequences,
        [to_float(row["trajectory_ratio"]) or 0.0 for row in rows],
        "Trajectory Coverage Ratio",
        f"EuRoC SVO {mode_name}: Trajectory Coverage",
        output_dir / "trajectory_ratio.png",
        color,
    )
    plot_single_bars(
        sequences,
        [to_float(row["se3_rmse_m"]) or 0.0 for row in rows],
        "SE3 RMSE (m)",
        f"EuRoC SVO {mode_name}: SE3 RMSE",
        output_dir / "se3_rmse.png",
        color,
    )
    plot_single_bars(
        sequences,
        [to_float(row["sim3_rmse_m"]) or 0.0 for row in rows],
        "Sim3 RMSE (m)",
        f"EuRoC SVO {mode_name}: Sim3 RMSE",
        output_dir / "sim3_rmse.png",
        color,
    )
    plot_single_bars(
        sequences,
        [to_float(row["path_ratio_est_over_gt"]) or 0.0 for row in rows],
        "Estimated / GT Path Ratio",
        f"EuRoC SVO {mode_name}: Path-Length Consistency",
        output_dir / "path_ratio.png",
        color,
    )


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    rows = read_rows(source_dir / "summary.csv")

    mono_rows = [row for row in rows if row.get("mode") == "mono"]
    mono_imu_rows = [row for row in rows if row.get("mode") == "mono-imu"]
    if not mono_rows or not mono_imu_rows:
        raise RuntimeError("Expected both mono and mono-imu rows in EuRoC summary.")

    mono_dir = output_root / "euroc_svo_mono_vs_gt_20260601"
    mono_imu_dir = output_root / "euroc_svo_mono_imu_vs_gt_20260601"

    build_mode_folder(mono_rows, mono_dir, "mono", "#b55d3d")
    build_mode_folder(mono_imu_rows, mono_imu_dir, "mono+IMU", "#2f7f6f")

    print(f"mono: {mono_dir}")
    print(f"mono+imu: {mono_imu_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
