#!/usr/bin/env python3
"""Generate EuRoC-style single-mode plots and CSV for TUM SVO runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks"
REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from TUM SVO sanity reports.")
    parser.add_argument(
        "--mono-dir",
        type=Path,
        help="Specific TUM mono experiment directory. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots and CSV output. Defaults to reports/evaluation/svo_tum_mono_<run>.",
    )
    return parser.parse_args()


def normalize_trace_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path_text = str(path)
    if path_text.startswith("/workspace/"):
        return REPO_ROOT / path_text[len("/workspace/"):]
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def latest_run(mode_name: str) -> Path:
    mode_root = BENCH_ROOT / mode_name
    runs = sorted(path for path in mode_root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {mode_root}")
    return runs[-1]


def load_reports(run_dir: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for seq_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        report_path = seq_dir / "sanity_report.json"
        if report_path.exists():
            reports[seq_dir.name] = json.loads(report_path.read_text(encoding="utf-8"))
            continue
        status_path = seq_dir / "status.txt"
        traj_path = seq_dir / "stamped_traj_estimate.txt"
        gt_path = seq_dir / "stamped_groundtruth.txt"
        if status_path.exists() and traj_path.exists() and gt_path.exists() and status_path.stat().st_size > 0 and traj_path.stat().st_size > 0:
            result = subprocess.run(
                ["python3", str(REPO_ROOT / "evaluation" / "odometry" / "svo_sanity_report.py"), str(seq_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            reports[seq_dir.name] = json.loads(result.stdout)
    if not reports:
        raise FileNotFoundError(f"No TUM sanity reports found under {run_dir}")
    return reports


def load_gpu_stats(trace_dir: str | Path) -> dict[str, float | int] | None:
    trace_dir = normalize_trace_dir(trace_dir)
    gpu_log = trace_dir / "log_gpu_usage.txt"
    if not gpu_log.exists():
        return None

    text = gpu_log.read_text(encoding="utf-8").strip()
    if not text:
        return None
    lowered = text.lower()
    if "skipped" in lowered or "no gpu samples" in lowered:
        return None

    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("gpu_index"):
            continue
        parts = stripped.split()
        if len(parts) != 7:
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue
    if not rows:
        return None

    gpu_utils = [row[1] for row in rows]
    mem_used = [row[3] for row in rows]
    temps = [row[5] for row in rows]
    powers = [row[6] for row in rows]
    return {
        "gpu_samples": len(rows),
        "gpu_mean_util_percent": sum(gpu_utils) / len(gpu_utils),
        "gpu_peak_util_percent": max(gpu_utils),
        "gpu_mean_mem_used_mb": sum(mem_used) / len(mem_used),
        "gpu_peak_mem_used_mb": max(mem_used),
        "gpu_mean_power_w": sum(powers) / len(powers),
        "gpu_peak_power_w": max(powers),
        "gpu_mean_temp_c": sum(temps) / len(temps),
        "gpu_peak_temp_c": max(temps),
    }


def load_pose_series(trace_dir: Path):
    import numpy as np

    trace_dir = normalize_trace_dir(trace_dir)
    gt = np.loadtxt(trace_dir / "stamped_groundtruth.txt")
    est = np.loadtxt(trace_dir / "stamped_traj_estimate.txt")
    return gt, est


def time_match_positions(gt, est):
    import numpy as np

    gt_t = gt[:, 0]
    gt_p = gt[:, 1:4]
    est_t = est[:, 0]
    est_p = est[:, 1:4]

    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = idx - 1
    right = idx
    use_right = np.abs(gt_t[right] - est_t) < np.abs(gt_t[left] - est_t)
    match_idx = np.where(use_right, right, left)
    matched_gt_t = gt_t[match_idx]
    matched_gt_p = gt_p[match_idx]

    time_err = np.abs(matched_gt_t - est_t)
    mask = time_err <= 0.01
    return matched_gt_p[mask], est_p[mask]


def sim3_align(est, gt):
    import numpy as np

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_centered = est - mu_est
    gt_centered = gt - mu_gt
    var_est = (est_centered**2).sum() / len(est)
    cov = est_centered.T @ gt_centered / len(est)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    rot = vt.T @ sign @ u.T
    scale = float(np.trace(np.diag(singular_values) @ sign) / var_est)
    trans = mu_gt - scale * (rot @ mu_est)
    return (scale * (rot @ est.T)).T + trans


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "tracking_ratio",
        "trajectory_ratio",
        "se3_rmse_m",
        "sim3_rmse_m",
        "sim3_scale",
        "path_ratio_est_over_gt",
        "tracking_frames",
        "frames_total",
        "failure_updates",
        "relocalization_frames",
        "paused_frames",
        "gpu_samples",
        "gpu_mean_util_percent",
        "gpu_peak_util_percent",
        "gpu_mean_mem_used_mb",
        "gpu_peak_mem_used_mb",
        "gpu_mean_power_w",
        "gpu_peak_power_w",
        "gpu_mean_temp_c",
        "gpu_peak_temp_c",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_or_blank(value):
    return "" if value is None else value


def filter_numeric(sequences, values):
    filtered_sequences = []
    filtered_values = []
    for sequence, value in zip(sequences, values):
        if value is None:
            continue
        filtered_sequences.append(sequence)
        filtered_values.append(float(value))
    return filtered_sequences, filtered_values


def plot_single_bars(plt, sequences, values, ylabel: str, title: str, out_path: Path) -> None:
    sequences, values = filter_numeric(sequences, values)
    if not sequences:
        return
    indices = list(range(len(sequences)))
    fig_width = max(10, len(sequences) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar(indices, values, color="#2f7f6f", width=0.65)
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_ratio(plt, sequences, values, out_path: Path) -> None:
    sequences, values = filter_numeric(sequences, values)
    if not sequences:
        return
    indices = list(range(len(sequences)))
    fig_width = max(10, len(sequences) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar(indices, values, color="#2f7f6f", width=0.65)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel("Estimated / GT Path Ratio")
    ax.set_title("Metric Scale Consistency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_optional_gpu_metric(plt, sequences, values, ylabel: str, title: str, out_path: Path) -> bool:
    filtered_sequences = [seq for seq, value in zip(sequences, values) if value is not None]
    filtered_values = [float(value) for value in values if value is not None]
    if not filtered_sequences:
        return False
    plot_single_bars(plt, filtered_sequences, filtered_values, ylabel, title, out_path)
    return True


def plot_trajectory_views(plt, sequence: str, gt_pos, est_pos, out_top: Path, out_side: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    ax.plot(est_pos[:, 0], est_pos[:, 1], label="mono", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Top View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_top, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], label="ground truth", color="#111111", linewidth=2.0)
    ax.plot(est_pos[:, 0], est_pos[:, 2], label="mono", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    mono_dir = args.mono_dir.resolve() if args.mono_dir else latest_run("mono3d_tum_mono")
    reports = load_reports(mono_dir)
    sequences = sorted(reports)

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = REPORT_ROOT / f"svo_tum_mono_{mono_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for sequence in sequences:
        report = reports[sequence]
        gpu_stats = load_gpu_stats(report["trace_dir"])
        rows.append(
            {
                "sequence": sequence,
                "tracking_ratio": value_or_blank(report["tracking_ratio"]),
                "trajectory_ratio": value_or_blank(report["trajectory_ratio"]),
                "se3_rmse_m": value_or_blank(report["se3"]["rmse_m"]),
                "sim3_rmse_m": value_or_blank(report["sim3"]["rmse_m"]),
                "sim3_scale": value_or_blank(report["scale_path"]["sim3_scale"]),
                "path_ratio_est_over_gt": value_or_blank(report["scale_path"]["path_ratio_est_over_gt"]),
                "tracking_frames": report["tracking_frames"],
                "frames_total": report["frames_total"],
                "failure_updates": report["failure_updates"],
                "relocalization_frames": report["relocalization_frames"],
                "paused_frames": report["paused_frames"],
                "gpu_samples": "" if gpu_stats is None else gpu_stats["gpu_samples"],
                "gpu_mean_util_percent": "" if gpu_stats is None else gpu_stats["gpu_mean_util_percent"],
                "gpu_peak_util_percent": "" if gpu_stats is None else gpu_stats["gpu_peak_util_percent"],
                "gpu_mean_mem_used_mb": "" if gpu_stats is None else gpu_stats["gpu_mean_mem_used_mb"],
                "gpu_peak_mem_used_mb": "" if gpu_stats is None else gpu_stats["gpu_peak_mem_used_mb"],
                "gpu_mean_power_w": "" if gpu_stats is None else gpu_stats["gpu_mean_power_w"],
                "gpu_peak_power_w": "" if gpu_stats is None else gpu_stats["gpu_peak_power_w"],
                "gpu_mean_temp_c": "" if gpu_stats is None else gpu_stats["gpu_mean_temp_c"],
                "gpu_peak_temp_c": "" if gpu_stats is None else gpu_stats["gpu_peak_temp_c"],
            }
        )
    write_csv(rows, output_dir / "summary.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps({"mono_dir": str(mono_dir), "output_dir": str(output_dir), "summary_csv": str(output_dir / "summary.csv"), "warning": "matplotlib is not installed; wrote CSV only"}, indent=2))
        return 0

    tracking = [reports[seq]["tracking_ratio"] for seq in sequences]
    traj = [reports[seq]["trajectory_ratio"] for seq in sequences]
    se3 = [reports[seq]["se3"]["rmse_m"] for seq in sequences]
    sim3 = [reports[seq]["sim3"]["rmse_m"] for seq in sequences]
    path_ratio = [reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences]
    gpu_stats = [load_gpu_stats(reports[seq]["trace_dir"]) for seq in sequences]

    plot_single_bars(plt, sequences, tracking, "Tracking Ratio", "SVO Tracking Ratio on TUM", output_dir / "tracking_ratio.png")
    plot_single_bars(plt, sequences, traj, "Trajectory Coverage Ratio", "SVO Trajectory Coverage on TUM", output_dir / "trajectory_ratio.png")
    plot_single_bars(plt, sequences, se3, "SE3 RMSE (m)", "Absolute Metric Error After SE3 Alignment", output_dir / "se3_rmse.png")
    plot_single_bars(plt, sequences, sim3, "Sim3 RMSE (m)", "Shape Error After Sim3 Alignment", output_dir / "sim3_rmse.png")
    plot_scale_ratio(plt, sequences, path_ratio, output_dir / "path_ratio.png")

    plot_optional_gpu_metric(
        plt,
        sequences,
        [None if stats is None else float(stats["gpu_mean_util_percent"]) for stats in gpu_stats],
        "Mean GPU Utilization (%)",
        "Mean GPU Utilization on TUM",
        output_dir / "gpu_mean_util.png",
    )
    plot_optional_gpu_metric(
        plt,
        sequences,
        [None if stats is None else float(stats["gpu_peak_mem_used_mb"]) for stats in gpu_stats],
        "Peak GPU Memory Used (MB)",
        "Peak GPU Memory Usage on TUM",
        output_dir / "gpu_peak_mem_used.png",
    )
    plot_optional_gpu_metric(
        plt,
        sequences,
        [None if stats is None else float(stats["gpu_mean_power_w"]) for stats in gpu_stats],
        "Mean GPU Power Draw (W)",
        "Mean GPU Power Draw on TUM",
        output_dir / "gpu_mean_power.png",
    )

    for sequence in sequences:
        trace_dir = normalize_trace_dir(reports[sequence]["trace_dir"])
        if reports[sequence].get("status") != "ok":
            continue
        gt, est = load_pose_series(trace_dir)
        gt_match, est_match = time_match_positions(gt, est)
        if len(gt_match) < 2 or len(est_match) < 2:
            continue
        est_aligned = sim3_align(est_match, gt_match)
        plot_trajectory_views(
            plt,
            sequence,
            gt_match,
            est_aligned,
            output_dir / f"{sequence}_trajectory_top.png",
            output_dir / f"{sequence}_trajectory_side.png",
        )

    print(
        json.dumps(
            {
                "mono_dir": str(mono_dir),
                "output_dir": str(output_dir),
                "summary_csv": str(output_dir / "summary.csv"),
                "sequences": sequences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
