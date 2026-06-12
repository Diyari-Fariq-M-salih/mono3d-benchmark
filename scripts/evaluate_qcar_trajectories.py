#!/usr/bin/env python3
"""Evaluate QCar trajectories for SVO and DA3, then compare them."""

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
DEFAULT_QCAR_ROOT = REPO_ROOT / "datasets" / "Qcar_evry"
DEFAULT_PREPARED_ROOT = REPO_ROOT / "datasets" / "qcar_evry_prepared"
DEFAULT_DA3_RUNS_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "qcar_evry"
DEFAULT_SVO_RUNS_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks" / "mono3d_qcar_evry_mono"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"
SVO_SANITY_SCRIPT = REPO_ROOT / "evaluation" / "odometry" / "svo_sanity_report.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QCar SVO and DA3 trajectories against GT, then compare them."
    )
    parser.add_argument("--qcar-root", type=Path, default=DEFAULT_QCAR_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--da3-runs-root", type=Path, default=DEFAULT_DA3_RUNS_ROOT)
    parser.add_argument("--svo-dir", type=Path, help="Specific SVO QCar run directory.")
    parser.add_argument("--svo-runs-root", type=Path, default=DEFAULT_SVO_RUNS_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--max-assoc-delta", type=float, default=0.05)
    parser.add_argument("--report-tag", help="Custom suffix for the three report folders.")
    return parser.parse_args()


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
    if "/reports/" in path_text:
        suffix = path_text.split("/reports/", 1)[1]
        candidate = REPO_ROOT / "reports" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def latest_svo_run(svo_runs_root: Path) -> Path:
    runs = sorted(path for path in svo_runs_root.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No SVO QCar runs found under {svo_runs_root}")
    return runs[-1]


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
        frames.append((float(source_path.stem), source_path))
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


def plot_trajectory_views(plt, sequence: str, gt_pos, left_pos, right_pos, out_top: Path, out_side: Path, left_label: str, right_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    if left_pos is not None:
        ax.plot(left_pos[:, 0], left_pos[:, 1], label=left_label, color="#b55d3d", linewidth=1.5, alpha=0.9)
    if right_pos is not None:
        ax.plot(right_pos[:, 0], right_pos[:, 1], label=right_label, color="#2f7f6f", linewidth=1.5, alpha=0.9)
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
    if left_pos is not None:
        ax.plot(left_pos[:, 0], left_pos[:, 2], label=left_label, color="#b55d3d", linewidth=1.5, alpha=0.9)
    if right_pos is not None:
        ax.plot(right_pos[:, 0], right_pos[:, 2], label=right_label, color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


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
            cwd=REPO_ROOT,
        )
        reports[seq_dir.name] = json.loads(result.stdout)
    if not reports:
        raise FileNotFoundError(f"No SVO reports found under {run_dir}")
    return reports


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pose_array(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def time_match_positions(gt: np.ndarray, est: np.ndarray, max_dt: float) -> tuple[np.ndarray, np.ndarray]:
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


def sim3_align(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    scale, rotation, translation = umeyama_alignment(est, gt, with_scale=True)
    return apply_sim3(est, scale, rotation, translation)


def evaluate_svo_vs_gt(svo_dir: Path, output_dir: Path, max_assoc_delta: float) -> dict[str, dict]:
    reports = load_svo_reports(svo_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = sorted(reports)

    rows = []
    for sequence in sequences:
        report = reports[sequence]
        rows.append(
            {
                "sequence": sequence,
                "status": report.get("status"),
                "tracking_ratio": report.get("tracking_ratio"),
                "trajectory_ratio": report.get("trajectory_ratio"),
                "matched_rows": report.get("timestamp_match", {}).get("matched_rows"),
                "se3_rmse_m": report.get("se3", {}).get("rmse_m"),
                "sim3_rmse_m": report.get("sim3", {}).get("rmse_m"),
                "sim3_scale": report.get("scale_path", {}).get("sim3_scale"),
                "path_ratio_est_over_gt": report.get("scale_path", {}).get("path_ratio_est_over_gt"),
                "trace_dir": report.get("trace_dir"),
            }
        )
    write_csv(
        output_dir / "summary.csv",
        [
            "sequence",
            "status",
            "tracking_ratio",
            "trajectory_ratio",
            "matched_rows",
            "se3_rmse_m",
            "sim3_rmse_m",
            "sim3_scale",
            "path_ratio_est_over_gt",
            "trace_dir",
        ],
        rows,
    )

    import matplotlib.pyplot as plt

    plot_single_bars(
        plt, sequences,
        [reports[seq]["sim3"]["rmse_m"] for seq in sequences],
        "Sim3 RMSE (m)",
        "QCar SVO vs GT: Trajectory Shape Error",
        output_dir / "sim3_rmse.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt, sequences,
        [reports[seq]["se3"]["rmse_m"] for seq in sequences],
        "SE3 RMSE (m)",
        "QCar SVO vs GT: Absolute Error",
        output_dir / "se3_rmse.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt, sequences,
        [reports[seq]["tracking_ratio"] for seq in sequences],
        "Tracking Ratio",
        "QCar SVO vs GT: Tracking Coverage",
        output_dir / "tracking_ratio.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt, sequences,
        [reports[seq]["trajectory_ratio"] for seq in sequences],
        "Trajectory Ratio",
        "QCar SVO vs GT: Trajectory Coverage",
        output_dir / "trajectory_ratio.png",
        "#b55d3d",
    )
    plot_single_bars(
        plt, sequences,
        [reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences],
        "Estimated / GT Path Ratio",
        "QCar SVO vs GT: Path-Length Consistency",
        output_dir / "path_ratio.png",
        "#b55d3d",
    )

    for sequence in sequences:
        report = reports[sequence]
        if report.get("status") != "ok":
            continue
        trace_dir = normalize_repo_path(report["trace_dir"])
        gt = load_pose_array(trace_dir / "stamped_groundtruth.txt")
        est = load_pose_array(trace_dir / "stamped_traj_estimate.txt")
        gt_match, est_match = time_match_positions(gt, est, max_assoc_delta)
        if len(gt_match) < 2 or len(est_match) < 2:
            continue
        aligned = sim3_align(est_match, gt_match)
        plot_trajectory_views(
            plt,
            sequence=sequence,
            gt_pos=gt_match,
            left_pos=aligned,
            right_pos=None,
            out_top=output_dir / f"{sequence}_trajectory_top.png",
            out_side=output_dir / f"{sequence}_trajectory_side.png",
            left_label="SVO",
            right_label="",
        )

    return reports


def infer_da3_run_spec(run_dir: Path, prepared_root: Path) -> dict[str, object]:
    match = re.match(r"(?P<stamp>\d{8}T\d{6}Z)_fps(?P<fps>[0-9p]+)_da3_streaming$", run_dir.name)
    if not match:
        raise ValueError(f"Unrecognized DA3 run dir name: {run_dir.name}")
    fps_token = match.group("fps").replace("p", ".")
    sequence = run_dir.parent.name
    prepared_dir = prepared_root / f"{sequence}_fps{match.group('fps')}"
    return {
        "sequence": sequence,
        "fps": fps_token,
        "fps_token": match.group("fps"),
        "prepared_dir": prepared_dir,
        "run_dir": run_dir,
    }


def evaluate_single_da3_run(run_dir: Path, prepared_dir: Path, qcar_sequence_dir: Path, max_assoc_delta: float) -> dict[str, object]:
    output_dir = run_dir / "eval_qcar"
    prepared_frames = read_prepared_timestamps(prepared_dir)
    gt_entries = load_pose_file(qcar_sequence_dir / "groundtruth.txt")
    camera_poses = load_da3_camera_poses(run_dir / "camera_poses.txt")
    est_timestamps = [ts for ts, _ in prepared_frames]
    if len(est_timestamps) != len(camera_poses):
        raise RuntimeError(
            f"Frame/pose count mismatch for {run_dir}: {len(est_timestamps)} frames vs {len(camera_poses)} poses"
        )

    save_tum_trajectory(output_dir / "estimate_raw_tum.txt", est_timestamps, camera_poses)
    matches = associate_by_timestamp(est_timestamps, gt_entries, max_assoc_delta)
    if len(matches) < 3:
        raise RuntimeError(f"Only {len(matches)} trajectory matches found for {run_dir}")

    est_pts = np.array([camera_poses[i][:3, 3] for i, _ in matches], dtype=np.float64)
    gt_pts = np.array([gt_entries[j][1] for _, j in matches], dtype=np.float64)

    se3_scale, se3_rot, se3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=False)
    sim3_scale, sim3_rot, sim3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=True)
    se3_aligned = apply_sim3(est_pts, se3_scale, se3_rot, se3_trans)
    sim3_aligned = apply_sim3(est_pts, sim3_scale, sim3_rot, sim3_trans)
    se3_errors = np.linalg.norm(se3_aligned - gt_pts, axis=1)
    sim3_errors = np.linalg.norm(sim3_aligned - gt_pts, axis=1)

    summary = {
        "prepared_dir": str(prepared_dir),
        "qcar_sequence_dir": str(qcar_sequence_dir),
        "match_count": len(matches),
        "se3": summarize_errors(se3_errors),
        "sim3": summarize_errors(sim3_errors),
        "sim3_scale": sim3_scale,
        "path_length_est_m": trajectory_path_length(est_pts),
        "path_length_gt_m": trajectory_path_length(gt_pts),
        "path_ratio_est_over_gt": (
            trajectory_path_length(est_pts) / trajectory_path_length(gt_pts)
            if trajectory_path_length(gt_pts) > 0
            else None
        ),
        "alignment": {
            "se3": {"scale": se3_scale, "rotation": se3_rot.tolist(), "translation": se3_trans.tolist()},
            "sim3": {"scale": sim3_scale, "rotation": sim3_rot.tolist(), "translation": sim3_trans.tolist()},
        },
    }
    write_json(output_dir / "trajectory_summary.json", summary)
    return summary


def evaluate_da3_vs_gt(da3_runs_root: Path, prepared_root: Path, qcar_root: Path, output_dir: Path, max_assoc_delta: float) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: dict[str, dict] = {}
    rows = []
    for run_dir in sorted(da3_runs_root.glob("*/*_fps*_da3_streaming")):
        if not (run_dir / "camera_poses.txt").exists():
            continue
        spec = infer_da3_run_spec(run_dir, prepared_root)
        sequence = str(spec["sequence"])
        prepared_dir = Path(spec["prepared_dir"])
        if not prepared_dir.exists():
            continue
        qcar_sequence_dir = qcar_root / sequence
        summary = evaluate_single_da3_run(run_dir, prepared_dir, qcar_sequence_dir, max_assoc_delta)
        key = f"{sequence}__fps{spec['fps_token']}"
        run_summaries[key] = {
            "sequence": sequence,
            "fps": spec["fps"],
            "fps_token": spec["fps_token"],
            "run_dir": str(run_dir),
            "trajectory_summary": summary,
        }
        rows.append(
            {
                "sequence": sequence,
                "fps": spec["fps"],
                "run_dir": str(run_dir),
                "match_count": summary["match_count"],
                "se3_rmse_m": summary["se3"]["rmse_m"],
                "sim3_rmse_m": summary["sim3"]["rmse_m"],
                "sim3_scale": summary["sim3_scale"],
                "path_ratio_est_over_gt": summary["path_ratio_est_over_gt"],
            }
        )

    if not rows:
        raise FileNotFoundError(f"No evaluable DA3 QCar runs found under {da3_runs_root}")

    write_csv(
        output_dir / "summary.csv",
        ["sequence", "fps", "run_dir", "match_count", "se3_rmse_m", "sim3_rmse_m", "sim3_scale", "path_ratio_est_over_gt"],
        rows,
    )

    best_by_sequence: dict[str, dict] = {}
    for payload in run_summaries.values():
        sequence = payload["sequence"]
        current = best_by_sequence.get(sequence)
        if current is None or payload["trajectory_summary"]["sim3"]["rmse_m"] < current["trajectory_summary"]["sim3"]["rmse_m"]:
            best_by_sequence[sequence] = payload

    import matplotlib.pyplot as plt

    sequences = sorted(best_by_sequence)
    plot_single_bars(
        plt,
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["sim3"]["rmse_m"] for seq in sequences],
        "Sim3 RMSE (m)",
        "QCar DA3 vs GT: Best Sim3 Error Per Sequence",
        output_dir / "sim3_rmse_best.png",
        "#2f7f6f",
    )
    plot_single_bars(
        plt,
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["se3"]["rmse_m"] for seq in sequences],
        "SE3 RMSE (m)",
        "QCar DA3 vs GT: Best Absolute Error Per Sequence",
        output_dir / "se3_rmse_best.png",
        "#2f7f6f",
    )
    plot_single_bars(
        plt,
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["path_ratio_est_over_gt"] for seq in sequences],
        "Estimated / GT Path Ratio",
        "QCar DA3 vs GT: Best Path-Length Consistency",
        output_dir / "path_ratio_best.png",
        "#2f7f6f",
    )

    for sequence in sequences:
        best = best_by_sequence[sequence]
        run_dir = Path(best["run_dir"])
        est = load_pose_array(run_dir / "eval_qcar" / "estimate_raw_tum.txt")
        gt = load_pose_array(normalize_repo_path(best["trajectory_summary"]["qcar_sequence_dir"]) / "groundtruth.txt")
        gt_match, est_match = time_match_positions(gt, est, max_assoc_delta)
        if len(gt_match) < 2 or len(est_match) < 2:
            continue
        aligned = sim3_align(est_match, gt_match)
        plot_trajectory_views(
            plt,
            sequence=sequence,
            gt_pos=gt_match,
            left_pos=aligned,
            right_pos=None,
            out_top=output_dir / f"{sequence}_trajectory_top.png",
            out_side=output_dir / f"{sequence}_trajectory_side.png",
            left_label="DA3",
            right_label="",
        )

    write_json(output_dir / "best_by_sequence.json", best_by_sequence)
    return best_by_sequence


def compare_svo_vs_da3(
    svo_reports: dict[str, dict],
    da3_best: dict[str, dict],
    output_dir: Path,
    max_assoc_delta: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = sorted(set(svo_reports) & set(da3_best))
    if not sequences:
        raise RuntimeError("No overlapping QCar sequences between SVO and DA3.")

    rows = []
    for sequence in sequences:
        svo = svo_reports[sequence]
        da3 = da3_best[sequence]
        svo_sim3 = svo["sim3"]["rmse_m"]
        da3_sim3 = da3["trajectory_summary"]["sim3"]["rmse_m"]
        winner = "SVO" if float(svo_sim3) < float(da3_sim3) else "DA3"
        rows.append(
            {
                "sequence": sequence,
                "svo_status": svo.get("status"),
                "svo_tracking_ratio": svo.get("tracking_ratio"),
                "svo_trajectory_ratio": svo.get("trajectory_ratio"),
                "svo_se3_rmse_m": svo["se3"]["rmse_m"],
                "svo_sim3_rmse_m": svo["sim3"]["rmse_m"],
                "svo_sim3_scale": svo["scale_path"]["sim3_scale"],
                "svo_path_ratio_est_over_gt": svo["scale_path"]["path_ratio_est_over_gt"],
                "da3_run_name": Path(da3["run_dir"]).name,
                "da3_fps": da3["fps"],
                "da3_match_count": da3["trajectory_summary"]["match_count"],
                "da3_se3_rmse_m": da3["trajectory_summary"]["se3"]["rmse_m"],
                "da3_sim3_rmse_m": da3["trajectory_summary"]["sim3"]["rmse_m"],
                "da3_sim3_scale": da3["trajectory_summary"]["sim3_scale"],
                "da3_path_ratio_est_over_gt": da3["trajectory_summary"]["path_ratio_est_over_gt"],
                "winner_by_sim3": winner,
            }
        )

    write_csv(
        output_dir / "summary.csv",
        [
            "sequence",
            "svo_status",
            "svo_tracking_ratio",
            "svo_trajectory_ratio",
            "svo_se3_rmse_m",
            "svo_sim3_rmse_m",
            "svo_sim3_scale",
            "svo_path_ratio_est_over_gt",
            "da3_run_name",
            "da3_fps",
            "da3_match_count",
            "da3_se3_rmse_m",
            "da3_sim3_rmse_m",
            "da3_sim3_scale",
            "da3_path_ratio_est_over_gt",
            "winner_by_sim3",
        ],
        rows,
    )

    import matplotlib.pyplot as plt

    plot_grouped_bars(
        plt, sequences,
        [svo_reports[seq]["sim3"]["rmse_m"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["sim3"]["rmse_m"] for seq in sequences],
        "Sim3 RMSE (m)",
        "QCar Trajectory Shape Error: SVO vs Best DA3",
        output_dir / "sim3_rmse.png",
        "SVO",
        "DA3",
    )
    plot_grouped_bars(
        plt, sequences,
        [svo_reports[seq]["se3"]["rmse_m"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["se3"]["rmse_m"] for seq in sequences],
        "SE3 RMSE (m)",
        "QCar Absolute Error: SVO vs Best DA3",
        output_dir / "se3_rmse.png",
        "SVO",
        "DA3",
    )
    plot_grouped_bars(
        plt, sequences,
        [svo_reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["path_ratio_est_over_gt"] for seq in sequences],
        "Estimated / GT Path Ratio",
        "QCar Path-Length Consistency: SVO vs Best DA3",
        output_dir / "path_ratio.png",
        "SVO",
        "DA3",
    )

    for sequence in sequences:
        gt_ref = None
        svo_aligned = None
        da3_aligned = None

        svo = svo_reports[sequence]
        if svo.get("status") == "ok":
            trace_dir = normalize_repo_path(svo["trace_dir"])
            gt_svo = load_pose_array(trace_dir / "stamped_groundtruth.txt")
            est_svo = load_pose_array(trace_dir / "stamped_traj_estimate.txt")
            gt_match_svo, est_match_svo = time_match_positions(gt_svo, est_svo, max_assoc_delta)
            if len(gt_match_svo) >= 2 and len(est_match_svo) >= 2:
                gt_ref = gt_match_svo
                svo_aligned = sim3_align(est_match_svo, gt_match_svo)

        da3 = da3_best[sequence]
        gt_da3 = load_pose_array(normalize_repo_path(da3["trajectory_summary"]["qcar_sequence_dir"]) / "groundtruth.txt")
        est_da3 = load_pose_array(Path(da3["run_dir"]) / "eval_qcar" / "estimate_raw_tum.txt")
        gt_match_da3, est_match_da3 = time_match_positions(gt_da3, est_da3, max_assoc_delta)
        if len(gt_match_da3) >= 2 and len(est_match_da3) >= 2:
            if gt_ref is None or len(gt_match_da3) > len(gt_ref):
                gt_ref = gt_match_da3
            da3_aligned = sim3_align(est_match_da3, gt_match_da3)

        if gt_ref is None:
            continue

        plot_trajectory_views(
            plt,
            sequence=sequence,
            gt_pos=gt_ref,
            left_pos=svo_aligned,
            right_pos=da3_aligned,
            out_top=output_dir / f"{sequence}_trajectory_top.png",
            out_side=output_dir / f"{sequence}_trajectory_side.png",
            left_label="SVO",
            right_label="DA3",
        )


def main() -> int:
    args = parse_args()
    qcar_root = args.qcar_root.resolve()
    prepared_root = args.prepared_root.resolve()
    da3_runs_root = args.da3_runs_root.resolve()
    svo_dir = args.svo_dir.resolve() if args.svo_dir else latest_svo_run(args.svo_runs_root.resolve())
    report_root = args.report_root.resolve()
    report_tag = args.report_tag or svo_dir.name

    svo_report_dir = report_root / f"qcar_svo_vs_gt_{report_tag}"
    da3_report_dir = report_root / f"qcar_da3_vs_gt_{report_tag}"
    compare_report_dir = report_root / f"qcar_svo_vs_da3_{report_tag}"

    svo_reports = evaluate_svo_vs_gt(svo_dir, svo_report_dir, args.max_assoc_delta)
    da3_best = evaluate_da3_vs_gt(da3_runs_root, prepared_root, qcar_root, da3_report_dir, args.max_assoc_delta)
    compare_svo_vs_da3(svo_reports, da3_best, compare_report_dir, args.max_assoc_delta)

    print(
        json.dumps(
            {
                "svo_dir": str(svo_dir),
                "svo_report_dir": str(svo_report_dir),
                "da3_report_dir": str(da3_report_dir),
                "compare_report_dir": str(compare_report_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
