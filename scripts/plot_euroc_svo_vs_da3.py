#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv as csv_std
import csv
import json
import os
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SVO_SUMMARY = REPO_ROOT / "reports" / "evaluation" / "svo_full11_euroc_mono_vs_mono_imu_20260601" / "summary.csv"
DEFAULT_DA3_RUNS_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "euroc"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"
DEFAULT_EUROC_ROOT = REPO_ROOT / "datasets" / "euroc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EuRoC SVO mono / mono+imu results against DA3 trajectory evaluation."
    )
    parser.add_argument("--svo-summary", type=Path, default=DEFAULT_SVO_SUMMARY)
    parser.add_argument("--da3-runs-root", type=Path, default=DEFAULT_DA3_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Defaults to reports/evaluation/euroc_svo_vs_da3_<latest>.") 
    return parser.parse_args()


def load_svo_summary(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_sequence: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_sequence[row["sequence"]][row["mode"]] = row
    return by_sequence


def load_best_da3(runs_root: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for summary_path in sorted(runs_root.glob("*/**/eval_euroc/trajectory_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        seq = payload["sequence"]
        sim3 = payload.get("sim3", {}).get("rmse_m")
        if sim3 is None:
            continue
        current = best.get(seq)
        if current is None or float(sim3) < float(current["summary"]["sim3"]["rmse_m"]):
            best[seq] = {
                "summary": payload,
                "run_dir": str(summary_path.parents[1]),
                "run_name": summary_path.parents[1].name,
            }
    if not best:
        raise FileNotFoundError(f"No EuRoC DA3 evaluation summaries found under {runs_root}")
    return best


def value_or_blank(value):
    return "" if value is None else value


def latest_token(runs_root: Path) -> str:
    tokens = sorted(
        path.name
        for path in runs_root.glob("*/*")
        if path.is_dir() and path.name.endswith("_da3_streaming")
    )
    return tokens[-1] if tokens else "latest"


def load_tum_trajectory(path: Path):
    import numpy as np

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_euroc_groundtruth(path: Path):
    import numpy as np

    entries = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv_std.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamp = int(row[0]) / 1_000_000_000.0
            position = np.array([float(v) for v in row[1:4]], dtype=np.float64)
            entries.append((timestamp, position))
    return entries


def time_match_positions(gt_entries, est_tum, max_dt: float = 0.01):
    import numpy as np

    gt_times = [item[0] for item in gt_entries]
    gt_positions = [item[1] for item in gt_entries]
    est_times = est_tum[:, 0]
    est_positions = est_tum[:, 1:4]

    matched_gt = []
    matched_est = []
    for idx, timestamp in enumerate(est_times):
        pos = bisect_left(gt_times, float(timestamp))
        candidates = []
        if pos < len(gt_times):
            candidates.append((abs(gt_times[pos] - timestamp), pos))
        if pos > 0:
            candidates.append((abs(gt_times[pos - 1] - timestamp), pos - 1))
        if not candidates:
            continue
        delta, gt_idx = min(candidates, key=lambda item: item[0])
        if delta <= max_dt:
            matched_gt.append(gt_positions[gt_idx])
            matched_est.append(est_positions[idx])

    if not matched_gt:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(matched_gt), np.vstack(matched_est)


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
    scale = float((np.diag(singular_values) @ sign).trace() / var_est)
    trans = mu_gt - scale * (rot @ mu_est)
    return (scale * (rot @ est.T)).T + trans


def plot_trajectory_views(sequence: str, gt_pos, da3_pos, out_top: Path, out_side: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    ax.plot(da3_pos[:, 0], da3_pos[:, 1], label="DA3", color="#3b6cb7", linewidth=1.5, alpha=0.9)
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
    ax.plot(da3_pos[:, 0], da3_pos[:, 2], label="DA3", color="#3b6cb7", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


def plot_trajectory_montage(sequence_payloads: list[tuple[str, object, object]], out_path: Path) -> None:
    if not sequence_payloads:
        return

    rows = len(sequence_payloads)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3.5 * rows, 10)))
    if rows == 1:
        axes = [axes]

    for row_idx, (sequence, gt_pos, da3_pos) in enumerate(sequence_payloads):
        top_ax, side_ax = axes[row_idx]

        top_ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
        top_ax.plot(da3_pos[:, 0], da3_pos[:, 1], label="DA3", color="#3b6cb7", linewidth=1.3, alpha=0.9)
        top_ax.set_title(f"{sequence} Top")
        top_ax.set_xlabel("x (m)")
        top_ax.set_ylabel("y (m)")
        top_ax.axis("equal")
        top_ax.grid(alpha=0.25)
        if row_idx == 0:
            top_ax.legend(fontsize=9)

        side_ax.plot(gt_pos[:, 0], gt_pos[:, 2], label="ground truth", color="#111111", linewidth=2.0)
        side_ax.plot(da3_pos[:, 0], da3_pos[:, 2], label="DA3", color="#3b6cb7", linewidth=1.3, alpha=0.9)
        side_ax.set_title(f"{sequence} Side")
        side_ax.set_xlabel("x (m)")
        side_ax.set_ylabel("z (m)")
        side_ax.axis("equal")
        side_ax.grid(alpha=0.25)

    fig.suptitle("EuRoC DA3 vs Ground Truth Trajectories", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grouped_bars(sequences: list[str], mono: list[float], mono_imu: list[float], da3: list[float], out_path: Path) -> None:
    indices = list(range(len(sequences)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(11, len(sequences) * 1.2), 5.8))
    ax.bar([i - width for i in indices], mono, width=width, label="SVO mono", color="#b55d3d")
    ax.bar(indices, mono_imu, width=width, label="SVO mono+imu", color="#2f7f6f")
    ax.bar([i + width for i in indices], da3, width=width, label="DA3", color="#3b6cb7")
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel("Sim3 RMSE (m)")
    ax.set_title("EuRoC Trajectory Shape Error After Sim3 Alignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    svo_summary = load_svo_summary(args.svo_summary.resolve())
    da3_best = load_best_da3(args.da3_runs_root.resolve())

    common = sorted(set(svo_summary) & set(da3_best))
    if not common:
        raise RuntimeError("No overlapping EuRoC sequences found between SVO summary and DA3 runs")

    output_dir = args.output_dir.resolve() if args.output_dir else (DEFAULT_REPORT_ROOT / f"euroc_svo_vs_da3_{latest_token(args.da3_runs_root.resolve())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mono_vals = []
    mono_imu_vals = []
    da3_vals = []
    for sequence in common:
        mono = svo_summary[sequence].get("mono")
        mono_imu = svo_summary[sequence].get("mono-imu")
        da3 = da3_best[sequence]["summary"]
        mono_sim3 = float(mono["sim3_rmse_m"]) if mono else None
        mono_imu_sim3 = float(mono_imu["sim3_rmse_m"]) if mono_imu else None
        da3_sim3 = float(da3["sim3"]["rmse_m"])
        rows.append(
            {
                "sequence": sequence,
                "svo_mono_sim3_rmse_m": value_or_blank(mono_sim3),
                "svo_mono_imu_sim3_rmse_m": value_or_blank(mono_imu_sim3),
                "da3_sim3_rmse_m": da3_sim3,
                "da3_match_count": da3["match_count"],
                "da3_path_ratio_est_over_gt": value_or_blank(da3.get("path_ratio_est_over_gt")),
                "da3_run_name": da3_best[sequence]["run_name"],
                "da3_run_dir": da3_best[sequence]["run_dir"],
            }
        )
        mono_vals.append(mono_sim3)
        mono_imu_vals.append(mono_imu_sim3)
        da3_vals.append(da3_sim3)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_grouped_bars(common, mono_vals, mono_imu_vals, da3_vals, output_dir / "sim3_rmse.png")

    montage_payloads = []
    for sequence in common:
        run_dir = Path(da3_best[sequence]["run_dir"])
        est_path = run_dir / "eval_euroc" / "estimate_raw_tum.txt"
        gt_csv = DEFAULT_EUROC_ROOT / sequence / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        if not est_path.exists() or not gt_csv.exists():
            continue
        gt_entries = load_euroc_groundtruth(gt_csv)
        est_tum = load_tum_trajectory(est_path)
        gt_pos, est_pos = time_match_positions(gt_entries, est_tum)
        if len(gt_pos) < 3 or len(est_pos) < 3:
            continue
        da3_aligned = sim3_align(est_pos, gt_pos)
        plot_trajectory_views(
            sequence,
            gt_pos,
            da3_aligned,
            output_dir / f"{sequence}_trajectory_top.png",
            output_dir / f"{sequence}_trajectory_side.png",
        )
        montage_payloads.append((sequence, gt_pos, da3_aligned))

    plot_trajectory_montage(montage_payloads, output_dir / "trajectory_montage.png")

    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"plot: {output_dir / 'sim3_rmse.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
