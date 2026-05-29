#!/usr/bin/env python3
"""Plot SVO sanity-report comparisons for mono vs mono-imu runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots from SVO sanity reports.")
    parser.add_argument(
        "--mono-dir",
        type=Path,
        help="Specific mono experiment directory. Defaults to the latest mono run.",
    )
    parser.add_argument(
        "--mono-imu-dir",
        type=Path,
        help="Specific mono-imu experiment directory. Defaults to the latest mono-imu run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots and CSV output. Defaults to outputs/plots/svo_sanity/<mono>__vs__<mono-imu>.",
    )
    return parser.parse_args()


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
    if not reports:
        raise FileNotFoundError(f"No sanity_report.json files found under {run_dir}")
    return reports


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "mode",
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
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pose_series(trace_dir: Path) -> tuple[object, object]:
    import numpy as np

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
    aligned = (scale * (rot @ est.T)).T + trans
    return aligned


def plot_grouped_bars(
    plt,
    sequences: list[str],
    mono_values: list[float],
    mono_imu_values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    indices = list(range(len(sequences)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in indices], mono_values, width=width, label="mono", color="#b55d3d")
    ax.bar([i + width / 2 for i in indices], mono_imu_values, width=width, label="mono-imu", color="#2f7f6f")
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scale_ratio(
    plt,
    sequences: list[str],
    mono_values: list[float],
    mono_imu_values: list[float],
    out_path: Path,
) -> None:
    indices = list(range(len(sequences)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in indices], mono_values, width=width, label="mono", color="#b55d3d")
    ax.bar([i + width / 2 for i in indices], mono_imu_values, width=width, label="mono-imu", color="#2f7f6f")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences)
    ax.set_ylabel("Estimated / GT Path Ratio")
    ax.set_title("Metric Scale Consistency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_views(
    plt,
    sequence: str,
    gt_pos,
    mono_pos,
    mono_imu_pos,
    out_top: Path,
    out_side: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    ax.plot(mono_pos[:, 0], mono_pos[:, 1], label="mono", color="#b55d3d", linewidth=1.5, alpha=0.9)
    ax.plot(mono_imu_pos[:, 0], mono_imu_pos[:, 1], label="mono-imu", color="#2f7f6f", linewidth=1.5, alpha=0.9)
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
    ax.plot(mono_pos[:, 0], mono_pos[:, 2], label="mono", color="#b55d3d", linewidth=1.5, alpha=0.9)
    ax.plot(mono_imu_pos[:, 0], mono_imu_pos[:, 2], label="mono-imu", color="#2f7f6f", linewidth=1.5, alpha=0.9)
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

    mono_dir = args.mono_dir.resolve() if args.mono_dir else latest_run("mono3d_euroc_mono")
    mono_imu_dir = args.mono_imu_dir.resolve() if args.mono_imu_dir else latest_run("mono3d_euroc_mono_imu")

    mono_reports = load_reports(mono_dir)
    mono_imu_reports = load_reports(mono_imu_dir)
    sequences = sorted(set(mono_reports) & set(mono_imu_reports))
    if not sequences:
        raise RuntimeError("No overlapping sequences between mono and mono-imu runs.")

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            REPO_ROOT
            / "outputs"
            / "plots"
            / "svo_sanity"
            / f"{mono_dir.name}__vs__{mono_imu_dir.name}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for mode_name, reports in [("mono", mono_reports), ("mono-imu", mono_imu_reports)]:
        for sequence in sequences:
            report = reports[sequence]
            rows.append(
                {
                    "sequence": sequence,
                    "mode": mode_name,
                    "tracking_ratio": report["tracking_ratio"],
                    "trajectory_ratio": report["trajectory_ratio"],
                    "se3_rmse_m": report["se3"]["rmse_m"],
                    "sim3_rmse_m": report["sim3"]["rmse_m"],
                    "sim3_scale": report["scale_path"]["sim3_scale"],
                    "path_ratio_est_over_gt": report["scale_path"]["path_ratio_est_over_gt"],
                    "tracking_frames": report["tracking_frames"],
                    "frames_total": report["frames_total"],
                    "failure_updates": report["failure_updates"],
                    "relocalization_frames": report["relocalization_frames"],
                    "paused_frames": report["paused_frames"],
                }
            )

    write_csv(rows, output_dir / "summary.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            json.dumps(
                {
                    "mono_dir": str(mono_dir),
                    "mono_imu_dir": str(mono_imu_dir),
                    "output_dir": str(output_dir),
                    "summary_csv": str(output_dir / "summary.csv"),
                    "warning": "matplotlib is not installed; wrote CSV only",
                },
                indent=2,
            )
        )
        return 0

    mono_tracking = [mono_reports[seq]["tracking_ratio"] for seq in sequences]
    mono_imu_tracking = [mono_imu_reports[seq]["tracking_ratio"] for seq in sequences]
    mono_traj = [mono_reports[seq]["trajectory_ratio"] for seq in sequences]
    mono_imu_traj = [mono_imu_reports[seq]["trajectory_ratio"] for seq in sequences]
    mono_se3 = [mono_reports[seq]["se3"]["rmse_m"] for seq in sequences]
    mono_imu_se3 = [mono_imu_reports[seq]["se3"]["rmse_m"] for seq in sequences]
    mono_sim3 = [mono_reports[seq]["sim3"]["rmse_m"] for seq in sequences]
    mono_imu_sim3 = [mono_imu_reports[seq]["sim3"]["rmse_m"] for seq in sequences]
    mono_path_ratio = [mono_reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences]
    mono_imu_path_ratio = [mono_imu_reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences]

    plot_grouped_bars(
        plt,
        sequences,
        mono_tracking,
        mono_imu_tracking,
        ylabel="Tracking Ratio",
        title="SVO Tracking Ratio: Mono vs Mono-IMU",
        out_path=output_dir / "tracking_ratio.png",
    )
    plot_grouped_bars(
        plt,
        sequences,
        mono_traj,
        mono_imu_traj,
        ylabel="Trajectory Coverage Ratio",
        title="SVO Trajectory Coverage: Mono vs Mono-IMU",
        out_path=output_dir / "trajectory_ratio.png",
    )
    plot_grouped_bars(
        plt,
        sequences,
        mono_se3,
        mono_imu_se3,
        ylabel="SE3 RMSE (m)",
        title="Absolute Metric Error After SE3 Alignment",
        out_path=output_dir / "se3_rmse.png",
    )
    plot_grouped_bars(
        plt,
        sequences,
        mono_sim3,
        mono_imu_sim3,
        ylabel="Sim3 RMSE (m)",
        title="Shape Error After Sim3 Alignment",
        out_path=output_dir / "sim3_rmse.png",
    )
    plot_scale_ratio(
        plt,
        sequences,
        mono_path_ratio,
        mono_imu_path_ratio,
        out_path=output_dir / "path_ratio.png",
    )

    for sequence in sequences:
        mono_trace_dir = Path(mono_reports[sequence]["trace_dir"])
        mono_imu_trace_dir = Path(mono_imu_reports[sequence]["trace_dir"])

        gt_mono, est_mono = load_pose_series(mono_trace_dir)
        gt_match_mono, est_match_mono = time_match_positions(gt_mono, est_mono)
        mono_aligned = sim3_align(est_match_mono, gt_match_mono)

        gt_imu, est_imu = load_pose_series(mono_imu_trace_dir)
        gt_match_imu, est_match_imu = time_match_positions(gt_imu, est_imu)
        mono_imu_aligned = sim3_align(est_match_imu, gt_match_imu)

        gt_ref = gt_match_imu if len(gt_match_imu) >= len(gt_match_mono) else gt_match_mono
        plot_trajectory_views(
            plt,
            sequence=sequence,
            gt_pos=gt_ref,
            mono_pos=mono_aligned,
            mono_imu_pos=mono_imu_aligned,
            out_top=output_dir / f"{sequence}_trajectory_top.png",
            out_side=output_dir / f"{sequence}_trajectory_side.png",
        )

    print(
        json.dumps(
            {
                "mono_dir": str(mono_dir),
                "mono_imu_dir": str(mono_imu_dir),
                "output_dir": str(output_dir),
                "summary_csv": str(output_dir / "summary.csv"),
                "plots": [
                    str(output_dir / "tracking_ratio.png"),
                    str(output_dir / "trajectory_ratio.png"),
                    str(output_dir / "se3_rmse.png"),
                    str(output_dir / "sim3_rmse.png"),
                    str(output_dir / "path_ratio.png"),
                ]
                + [str(output_dir / f"{sequence}_trajectory_top.png") for sequence in sequences]
                + [str(output_dir / f"{sequence}_trajectory_side.png") for sequence in sequences],
                "sequences": sequences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
