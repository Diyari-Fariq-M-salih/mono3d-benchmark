#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EUROC_ROOT = REPO_ROOT / "datasets" / "euroc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DA3-Streaming EuRoC run against EuRoC trajectory ground truth."
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="DA3 run folder.")
    parser.add_argument("--sequence-dir", required=True, type=Path, help="EuRoC sequence directory.")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        help="Prepared EuRoC frame folder used for the DA3 run. Defaults to inferring from run metadata if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for evaluation outputs. Defaults to <run-dir>/eval_euroc.",
    )
    parser.add_argument(
        "--max-assoc-delta",
        type=float,
        default=0.01,
        help="Maximum timestamp association delta in seconds.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def read_euroc_image_timestamps(image_dir: Path) -> list[float]:
    frames = []
    for frame_path in sorted(image_dir.glob("*.png")):
        try:
            frames.append(int(frame_path.stem) / 1_000_000_000.0)
        except ValueError:
            continue
    if not frames:
        raise FileNotFoundError(f"No timestamped PNG frames found in {image_dir}")
    return frames


def infer_prepared_dir(run_dir: Path) -> Path | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    prepared_dir = payload.get("image_dir") or payload.get("frame_dir")
    if not prepared_dir:
        return None
    return Path(prepared_dir)


def load_euroc_groundtruth(path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamp = int(row[0]) / 1_000_000_000.0
            t = np.array([float(v) for v in row[1:4]], dtype=np.float64)
            q = np.array([float(v) for v in row[5:8]] + [float(row[4])], dtype=np.float64)
            entries.append((timestamp, t, q))
    if not entries:
        raise FileNotFoundError(f"No ground-truth poses loaded from {path}")
    return entries


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


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


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
        "max_m": float(np.max(errors)),
    }


def trajectory_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def save_tum_trajectory(path: Path, timestamps: list[float], poses: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for timestamp, pose in zip(timestamps, poses):
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)
            handle.write(
                f"{timestamp:.9f} {translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    sequence_dir = resolve_path(args.sequence_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else (run_dir / "eval_euroc")
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_dir = resolve_path(args.prepared_dir) if args.prepared_dir else infer_prepared_dir(run_dir)
    if prepared_dir is None:
        image_dir = sequence_dir / "mav0" / "cam0" / "data"
    else:
        image_dir = prepared_dir.resolve()
    gt_csv = sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    pose_path = run_dir / "camera_poses.txt"

    est_timestamps = read_euroc_image_timestamps(image_dir)
    camera_poses = load_da3_camera_poses(pose_path)
    if len(camera_poses) != len(est_timestamps):
        raise RuntimeError(
            f"Pose/frame count mismatch for {sequence_dir.name}: {len(camera_poses)} poses vs {len(est_timestamps)} frames"
        )

    gt_entries = load_euroc_groundtruth(gt_csv)
    matches = associate_by_timestamp(est_timestamps, gt_entries, args.max_assoc_delta)
    if len(matches) < 3:
        raise RuntimeError(f"Only {len(matches)} trajectory matches found; need at least 3")

    est_raw_tum_path = output_dir / "estimate_raw_tum.txt"
    save_tum_trajectory(est_raw_tum_path, est_timestamps, camera_poses)

    est_pts = np.array([camera_poses[i][:3, 3] for i, _ in matches], dtype=np.float64)
    gt_pts = np.array([gt_entries[j][1] for _, j in matches], dtype=np.float64)

    se3_scale, se3_rot, se3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=False)
    sim3_scale, sim3_rot, sim3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=True)
    se3_aligned = apply_sim3(est_pts, se3_scale, se3_rot, se3_trans)
    sim3_aligned = apply_sim3(est_pts, sim3_scale, sim3_rot, sim3_trans)
    se3_errors = np.linalg.norm(se3_aligned - gt_pts, axis=1)
    sim3_errors = np.linalg.norm(sim3_aligned - gt_pts, axis=1)

    gt_path_len = trajectory_path_length(gt_pts)
    est_path_len = trajectory_path_length(est_pts)
    trajectory_summary = {
        "sequence": sequence_dir.name,
        "match_count": len(matches),
        "frame_count": len(est_timestamps),
        "se3": summarize_errors(se3_errors),
        "sim3": summarize_errors(sim3_errors),
        "sim3_scale": sim3_scale,
        "path_ratio_est_over_gt": (est_path_len / gt_path_len) if gt_path_len > 0 else None,
        "alignment": {
            "se3": {"scale": se3_scale, "rotation": se3_rot.tolist(), "translation": se3_trans.tolist()},
            "sim3": {"scale": sim3_scale, "rotation": sim3_rot.tolist(), "translation": sim3_trans.tolist()},
        },
    }
    write_json(output_dir / "trajectory_summary.json", trajectory_summary)
    print(f"trajectory summary: {output_dir / 'trajectory_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
