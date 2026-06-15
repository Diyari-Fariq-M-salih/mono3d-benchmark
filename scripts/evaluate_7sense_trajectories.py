#!/usr/bin/env python3
"""Evaluate 7sense SVO and DA3 trajectories against GT, then compare them."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from bisect import bisect_left
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_7SENSE_ROOT = REPO_ROOT / "datasets" / "7sense"
DEFAULT_PREPARED_ROOT = REPO_ROOT / "datasets" / "7sense_prepared"
DEFAULT_DA3_RUNS_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "7sense"
DEFAULT_SVO_RUNS_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks" / "mono3d_7sense_mono"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"
SVO_SANITY_SCRIPT = REPO_ROOT / "evaluation" / "odometry" / "svo_sanity_report.py"
ASSUMED_SOURCE_FPS = 30.0
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 7sense SVO and DA3 trajectories against GT, then compare them."
    )
    parser.add_argument("--seven-root", type=Path, default=DEFAULT_7SENSE_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--da3-runs-root", type=Path, default=DEFAULT_DA3_RUNS_ROOT)
    parser.add_argument("--svo-dir", type=Path, help="Specific SVO 7sense run directory.")
    parser.add_argument("--svo-runs-root", type=Path, default=DEFAULT_SVO_RUNS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--max-assoc-delta", type=float, default=0.05)
    parser.add_argument("--report-tag", help="Custom suffix for the three report folders.")
    return parser.parse_args()


def latest_svo_run(svo_runs_root: Path) -> Path:
    runs = sorted(path for path in svo_runs_root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No SVO 7sense runs found under {svo_runs_root}")
    return runs[-1]


def normalize_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    path_text = str(path)
    if path_text.startswith("/workspace/"):
        return REPO_ROOT / path_text[len("/workspace/") :]
    if "/datasets/" in path_text:
        suffix = path_text.split("/datasets/", 1)[1]
        candidate = REPO_ROOT / "datasets" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    if "/outputs/" in path_text:
        suffix = path_text.split("/outputs/", 1)[1]
        candidate = REPO_ROOT / "outputs" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    repo_name = REPO_ROOT.name
    marker = f"/{repo_name}/"
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1]
        return REPO_ROOT / suffix
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_pose_file(path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            timestamp = float(parts[0])
            t = np.array([float(v) for v in parts[1:4]], dtype=np.float64)
            q = np.array([float(v) for v in parts[4:8]], dtype=np.float64)
            entries.append((timestamp, t, q))
    return entries


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def read_prepared_timestamps(prepared_dir: Path) -> list[tuple[float, Path]]:
    frames = []
    for frame_path in sorted(prepared_dir.glob("frame_*.png")):
        source_path = frame_path.resolve()
        match = re.search(r"frame-(\d+)\.color\.png$", source_path.name)
        if not match:
            raise ValueError(f"Unexpected source frame name: {source_path.name}")
        timestamp = int(match.group(1)) / ASSUMED_SOURCE_FPS
        frames.append((timestamp, source_path))
    if not frames:
        raise FileNotFoundError(f"No prepared frames found in {prepared_dir}")
    return frames


def load_da3_camera_poses(path: Path) -> list[np.ndarray]:
    poses = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            values = [float(v) for v in line.strip().split()]
            if len(values) != 16:
                continue
            poses.append(np.array(values, dtype=np.float64).reshape(4, 4))
    if not poses:
        raise FileNotFoundError(f"No poses loaded from {path}")
    return poses


def associate_by_timestamp(
    est_timestamps: list[float],
    gt_entries: list[tuple[float, np.ndarray, np.ndarray]],
    max_delta: float,
) -> list[tuple[int, int]]:
    gt_times = [item[0] for item in gt_entries]
    matches = []
    for est_idx, ts in enumerate(est_timestamps):
        pos = bisect_left(gt_times, ts)
        candidates = []
        if pos < len(gt_times):
            candidates.append((abs(gt_times[pos] - ts), pos))
        if pos > 0:
            candidates.append((abs(gt_times[pos - 1] - ts), pos - 1))
        if not candidates:
            continue
        delta, gt_idx = min(candidates, key=lambda item: item[0])
        if delta <= max_delta:
            matches.append((est_idx, gt_idx))
    return matches


def umeyama_alignment(source: np.ndarray, target: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    cov = (tgt_centered.T @ src_centered) / source.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rotation = u @ s @ vt
    if with_scale:
        var = np.mean(np.sum(src_centered**2, axis=1))
        scale = np.trace(np.diag(d) @ s) / var
    else:
        scale = 1.0
    translation = tgt_mean - scale * rotation @ src_mean
    return float(scale), rotation, translation


def apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (rotation @ points.T)).T + translation


def summarize_errors(errors: np.ndarray) -> dict[str, float]:
    return {
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "max_m": float(np.max(errors)),
    }


def trajectory_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def save_tum_trajectory(path: Path, timestamps: list[float], poses: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, pose in zip(timestamps, poses):
            q = rotation_matrix_to_quaternion(pose[:3, :3])
            t = pose[:3, 3]
            handle.write(
                f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def plot_grouped_bars(plt, sequences, left_values, right_values, ylabel: str, title: str, out_path: Path, left_label: str, right_label: str) -> None:
    filtered = [
        (seq, left, right)
        for seq, left, right in zip(sequences, left_values, right_values)
        if left is not None and right is not None
    ]
    if not filtered:
        return
    seqs = [item[0] for item in filtered]
    left = [float(item[1]) for item in filtered]
    right = [float(item[2]) for item in filtered]
    indices = list(range(len(seqs)))
    width = 0.38
    fig_width = max(10, len(seqs) * 1.15)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar([i - width / 2 for i in indices], left, width=width, label=left_label, color="#b55d3d")
    ax.bar([i + width / 2 for i in indices], right, width=width, label=right_label, color="#2f7f6f")
    ax.set_xticks(indices)
    ax.set_xticklabels(seqs, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_single_bars(plt, sequences, values, ylabel: str, title: str, out_path: Path, color: str) -> None:
    filtered = [(seq, value) for seq, value in zip(sequences, values) if value is not None]
    if not filtered:
        return
    seqs = [item[0] for item in filtered]
    vals = [float(item[1]) for item in filtered]
    fig_width = max(10, len(seqs) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar(range(len(seqs)), vals, color=color)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(seqs, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_views(plt, sequence: str, gt_pos, svo_pos, da3_pos, out_top: Path, out_side: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    if svo_pos is not None:
        ax.plot(svo_pos[:, 0], svo_pos[:, 1], label="SVO", color="#b55d3d", linewidth=1.5, alpha=0.9)
    if da3_pos is not None:
        ax.plot(da3_pos[:, 0], da3_pos[:, 1], label="DA3", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Top View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_top, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], label="ground truth", color="#111111", linewidth=2.0)
    if svo_pos is not None:
        ax.plot(svo_pos[:, 0], svo_pos[:, 2], label="SVO", color="#b55d3d", linewidth=1.5, alpha=0.9)
    if da3_pos is not None:
        ax.plot(da3_pos[:, 0], da3_pos[:, 2], label="DA3", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


def infer_prepared_dir(run_dir: Path, prepared_root: Path) -> Path:
    name = run_dir.name
    match = re.search(r"_fps(\d+(?:p\d+)?)_da3_streaming$", name)
    if not match:
        raise ValueError(f"Cannot infer prepared dir from run name: {name}")
    fps_label = match.group(1)
    sequence_name = run_dir.parent.name
    prepared_dir = prepared_root / f"{sequence_name}_fps{fps_label}"
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Missing prepared dir: {prepared_dir}")
    return prepared_dir


def load_best_da3_reports(runs_root: Path, prepared_root: Path, seven_root: Path, max_assoc_delta: float) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for sequence_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        for run_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
            camera_path = run_dir / "camera_poses.txt"
            if not camera_path.exists():
                continue
            prepared_dir = infer_prepared_dir(run_dir, prepared_root)
            scene_name, seq_name = sequence.rsplit("_", 1)
            gt_sequence_dir = seven_root / scene_name / seq_name
            if not gt_sequence_dir.exists():
                raise FileNotFoundError(f"Missing GT sequence dir: {gt_sequence_dir}")

            prepared_frames = read_prepared_timestamps(prepared_dir)
            est_timestamps = [item[0] for item in prepared_frames]
            est_poses = load_da3_camera_poses(camera_path)
            if len(est_poses) != len(est_timestamps):
                raise RuntimeError(
                    f"Pose/frame count mismatch for {sequence}: {len(est_poses)} poses vs {len(est_timestamps)} frames"
                )

            gt_entries = load_pose_file(
                REPO_ROOT
                / "methods"
                / "svo"
                / "upstream"
                / "svo_benchmarking"
                / "data"
                / "mono3d"
                / "7sense"
                / "mono"
                / sequence
                / "data"
                / "stamped_groundtruth.txt"
            )
            matches = associate_by_timestamp(est_timestamps, gt_entries, max_assoc_delta)
            if len(matches) < 3:
                continue

            est_positions = np.array([est_poses[est_idx][:3, 3] for est_idx, _ in matches], dtype=np.float64)
            gt_positions = np.array([gt_entries[gt_idx][1] for _, gt_idx in matches], dtype=np.float64)

            se3_errors = np.linalg.norm(est_positions - gt_positions, axis=1)
            sim3_scale, sim3_rot, sim3_trans = umeyama_alignment(est_positions, gt_positions, with_scale=True)
            sim3_positions = apply_sim3(est_positions, sim3_scale, sim3_rot, sim3_trans)
            sim3_errors = np.linalg.norm(sim3_positions - gt_positions, axis=1)
            gt_path_length = trajectory_path_length(gt_positions)
            est_path_length = trajectory_path_length(est_positions)

            summary = {
                "sequence": sequence,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "prepared_dir": str(prepared_dir),
                "gt_sequence_dir": str(gt_sequence_dir),
                "match_count": len(matches),
                "trajectory_ratio": len(matches) / max(1, len(gt_entries)),
                "se3": summarize_errors(se3_errors),
                "sim3": {
                    **summarize_errors(sim3_errors),
                    "scale": sim3_scale,
                },
                "path_ratio_est_over_gt": est_path_length / gt_path_length if gt_path_length > 0 else None,
                "estimate_raw_tum": str(run_dir / "eval_7sense" / "estimate_raw_tum.txt"),
            }

            eval_dir = run_dir / "eval_7sense"
            eval_dir.mkdir(parents=True, exist_ok=True)
            save_tum_trajectory(eval_dir / "estimate_raw_tum.txt", est_timestamps, est_poses)
            write_json(eval_dir / "trajectory_summary.json", summary)

            current = best.get(sequence)
            if current is None or summary["sim3"]["rmse_m"] < current["trajectory_summary"]["sim3"]["rmse_m"]:
                best[sequence] = {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "prepared_dir": str(prepared_dir),
                    "gt_sequence_dir": str(gt_sequence_dir),
                    "trajectory_summary": summary,
                }
    if not best:
        raise FileNotFoundError(f"No evaluable DA3 runs found under {runs_root}")
    return best


def load_svo_reports(run_dir: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for seq_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        report_path = seq_dir / "sanity_report.json"
        if report_path.exists():
            reports[seq_dir.name] = json.loads(report_path.read_text(encoding="utf-8"))
            continue
        result = subprocess.run(
            ["python3", str(SVO_SANITY_SCRIPT), str(seq_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        reports[seq_dir.name] = json.loads(result.stdout)
    if not reports:
        raise FileNotFoundError(f"No SVO reports found under {run_dir}")
    return reports


def load_svo_pose_series(trace_dir: Path):
    trace_dir = normalize_repo_path(trace_dir)
    gt = np.loadtxt(trace_dir / "stamped_groundtruth.txt")
    est = np.loadtxt(trace_dir / "stamped_traj_estimate.txt")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    return gt, est


def load_da3_pose_series(run_dir: Path):
    est = np.loadtxt(run_dir / "eval_7sense" / "estimate_raw_tum.txt")
    if est.ndim == 1:
        est = est.reshape(1, -1)
    summary = json.loads((run_dir / "eval_7sense" / "trajectory_summary.json").read_text(encoding="utf-8"))
    gt = np.loadtxt(
        REPO_ROOT
        / "methods"
        / "svo"
        / "upstream"
        / "svo_benchmarking"
        / "data"
        / "mono3d"
        / "7sense"
        / "mono"
        / summary["sequence"]
        / "data"
        / "stamped_groundtruth.txt"
    )
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    return gt, est


def time_match_positions(gt, est, max_dt: float = 0.05):
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
    mask = time_err <= max_dt
    return matched_gt_p[mask], est_p[mask]


def sim3_align(est, gt):
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


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "svo_status",
        "svo_tracking_ratio",
        "svo_trajectory_ratio",
        "svo_se3_rmse_m",
        "svo_sim3_rmse_m",
        "svo_sim3_scale",
        "svo_path_ratio_est_over_gt",
        "da3_run_name",
        "da3_match_count",
        "da3_trajectory_ratio",
        "da3_se3_rmse_m",
        "da3_sim3_rmse_m",
        "da3_sim3_scale",
        "da3_path_ratio_est_over_gt",
        "winner_by_sim3",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seven_root = args.seven_root.resolve()
    prepared_root = args.prepared_root.resolve()
    da3_runs_root = args.da3_runs_root.resolve()
    svo_run_dir = args.svo_dir.resolve() if args.svo_dir else latest_svo_run(args.svo_runs_root.resolve())
    report_root = args.report_root.resolve()

    svo_reports = load_svo_reports(svo_run_dir)
    da3_reports = load_best_da3_reports(da3_runs_root, prepared_root, seven_root, args.max_assoc_delta)

    report_tag = args.report_tag or svo_run_dir.name
    svo_gt_dir = report_root / f"7sense_svo_vs_gt_{report_tag}"
    da3_gt_dir = report_root / f"7sense_da3_vs_gt_{report_tag}"
    compare_dir = report_root / f"7sense_svo_vs_da3_{report_tag}"
    for out_dir in (svo_gt_dir, da3_gt_dir, compare_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(set(svo_reports) & set(da3_reports))
    rows: list[dict[str, object]] = []
    svo_sim3_values: list[float | None] = []
    da3_sim3_values: list[float | None] = []
    svo_path_values: list[float | None] = []
    da3_path_values: list[float | None] = []

    import matplotlib.pyplot as plt

    for sequence in sequences:
        svo = svo_reports[sequence]
        da3 = da3_reports[sequence]["trajectory_summary"]
        row = {
            "sequence": sequence,
            "svo_status": svo.get("status"),
            "svo_tracking_ratio": svo.get("tracking_ratio"),
            "svo_trajectory_ratio": svo.get("trajectory_ratio"),
            "svo_se3_rmse_m": svo.get("se3", {}).get("rmse_m"),
            "svo_sim3_rmse_m": svo.get("sim3", {}).get("rmse_m"),
            "svo_sim3_scale": svo.get("sim3", {}).get("scale"),
            "svo_path_ratio_est_over_gt": svo.get("path_ratio_est_over_gt"),
            "da3_run_name": da3.get("run_name"),
            "da3_match_count": da3.get("match_count"),
            "da3_trajectory_ratio": da3.get("trajectory_ratio"),
            "da3_se3_rmse_m": da3.get("se3", {}).get("rmse_m"),
            "da3_sim3_rmse_m": da3.get("sim3", {}).get("rmse_m"),
            "da3_sim3_scale": da3.get("sim3", {}).get("scale"),
            "da3_path_ratio_est_over_gt": da3.get("path_ratio_est_over_gt"),
            "winner_by_sim3": None,
        }
        svo_sim3 = row["svo_sim3_rmse_m"]
        da3_sim3 = row["da3_sim3_rmse_m"]
        if svo_sim3 is not None and da3_sim3 is not None:
            row["winner_by_sim3"] = "SVO" if svo_sim3 < da3_sim3 else "DA3"
        rows.append(row)
        svo_sim3_values.append(svo_sim3)
        da3_sim3_values.append(da3_sim3)
        svo_path_values.append(row["svo_path_ratio_est_over_gt"])
        da3_path_values.append(row["da3_path_ratio_est_over_gt"])

        trace_dir = normalize_repo_path(svo["trace_dir"])
        gt_svo, est_svo = load_svo_pose_series(trace_dir)
        gt_da3, est_da3 = load_da3_pose_series(Path(da3["run_dir"]))
        gt_pos_svo, svo_pos = time_match_positions(gt_svo, est_svo, args.max_assoc_delta)
        gt_pos_da3, da3_pos = time_match_positions(gt_da3, est_da3, args.max_assoc_delta)
        if len(svo_pos) >= 3:
            svo_pos = sim3_align(svo_pos, gt_pos_svo)
        else:
            svo_pos = None
            gt_pos_svo = None
        if len(da3_pos) >= 3:
            da3_pos = sim3_align(da3_pos, gt_pos_da3)
        else:
            da3_pos = None
            gt_pos_da3 = None
        gt_plot = gt_pos_svo if gt_pos_svo is not None else gt_pos_da3
        if gt_plot is not None:
            plot_trajectory_views(
                plt,
                sequence,
                gt_plot,
                svo_pos,
                da3_pos,
                compare_dir / f"{sequence}_trajectory_top.png",
                compare_dir / f"{sequence}_trajectory_side.png",
            )

    write_csv(rows, compare_dir / "summary.csv")
    write_csv(rows, svo_gt_dir / "summary.csv")
    write_csv(rows, da3_gt_dir / "summary.csv")
    write_json(da3_gt_dir / "best_by_sequence.json", da3_reports)

    plot_grouped_bars(
        plt,
        sequences,
        svo_sim3_values,
        da3_sim3_values,
        "Sim3 RMSE (m)",
        "7sense Trajectory Shape Error: SVO vs Best DA3",
        compare_dir / "sim3_rmse.png",
        "SVO",
        "DA3",
    )
    plot_grouped_bars(
        plt,
        sequences,
        svo_path_values,
        da3_path_values,
        "Path ratio est/gt",
        "7sense Path Ratio: SVO vs Best DA3",
        compare_dir / "path_ratio.png",
        "SVO",
        "DA3",
    )
    plot_single_bars(
        plt,
        sequences,
        svo_sim3_values,
        "Sim3 RMSE (m)",
        "7sense SVO Mono vs GT",
        svo_gt_dir / "sim3_rmse.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt,
        sequences,
        [row["svo_tracking_ratio"] for row in rows],
        "Tracking ratio",
        "7sense SVO Tracking Ratio",
        svo_gt_dir / "tracking_ratio.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt,
        sequences,
        [row["svo_trajectory_ratio"] for row in rows],
        "Trajectory ratio",
        "7sense SVO Trajectory Ratio",
        svo_gt_dir / "trajectory_ratio.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt,
        sequences,
        da3_sim3_values,
        "Sim3 RMSE (m)",
        "7sense Best DA3 vs GT",
        da3_gt_dir / "sim3_rmse_best.png",
        "#2f7f6f",
    )
    plot_single_bars(
        plt,
        sequences,
        [row["da3_trajectory_ratio"] for row in rows],
        "Trajectory ratio",
        "7sense Best DA3 Trajectory Ratio",
        da3_gt_dir / "trajectory_ratio_best.png",
        "#2f7f6f",
    )
    plot_single_bars(
        plt,
        sequences,
        da3_path_values,
        "Path ratio est/gt",
        "7sense Best DA3 Path Ratio",
        da3_gt_dir / "path_ratio_best.png",
        "#2f7f6f",
    )

    print(f"[report] SVO vs GT: {svo_gt_dir}")
    print(f"[report] DA3 vs GT: {da3_gt_dir}")
    print(f"[report] SVO vs DA3: {compare_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
