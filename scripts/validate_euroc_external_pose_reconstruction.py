#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DA3_RUN = (
    REPO_ROOT
    / "outputs"
    / "reconstructions"
    / "depth_anything_3"
    / "euroc"
    / "MH_01_easy"
    / "20260610T132107Z_fps5_da3_streaming"
)
DEFAULT_EXTERNAL_TRAJ = (
    REPO_ROOT
    / "environment"
    / "ros_ws"
    / "svo"
    / "src"
    / "rpg_trajectory_evaluation"
    / "results"
    / "euroc_mono_stereo"
    / "laptop"
    / "vio_stereo"
    / "laptop_vio_stereo_MH_01"
    / "stamped_traj_estimate.txt"
)
DEFAULT_SEQUENCE_DIR = REPO_ROOT / "datasets" / "euroc" / "MH_01_easy"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "reconstructions"
    / "depth_anything_3"
    / "euroc_pose_swap"
    / "MH_01_easy_svo_vio_external"
)
DEFAULT_NPZ_PROCESS = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "da3_streaming" / "npz_output_process.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DA3 EuRoC reconstruction using an external pose trajectory."
    )
    parser.add_argument("--da3-run-dir", type=Path, default=DEFAULT_DA3_RUN)
    parser.add_argument("--external-traj", type=Path, default=DEFAULT_EXTERNAL_TRAJ)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-assoc-delta", type=float, default=0.05)
    parser.add_argument(
        "--association-mode",
        choices=("nearest", "interpolate"),
        default="nearest",
        help="How to map external poses onto DA3 timestamps.",
    )
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Keep only the overlapping DA3 frame window when the external trajectory does not cover the full run.",
    )
    parser.add_argument("--conf-threshold-coef", type=float, default=0.5)
    parser.add_argument("--sample-ratio", type=float, default=0.015)
    parser.add_argument("--with-pointcloud", action="store_true", help="Evaluate point cloud if GT cloud is available.")
    parser.add_argument("--cloud-sample", type=int, default=200000)
    parser.add_argument("--export-ply", action="store_true", help="Also export a fused PLY with swapped poses.")
    parser.add_argument(
        "--conda-env",
        default="da3stream",
        help="Conda environment used for npz_output_process.py. Use empty string to disable conda wrapping.",
    )
    parser.add_argument(
        "--python-cmd",
        default=sys.executable,
        help="Python executable used for npz_output_process.py when --export-ply is enabled.",
    )
    parser.add_argument("--npz-process-script", type=Path, default=DEFAULT_NPZ_PROCESS)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_tum_pose_file(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def tum_rows_to_c2w(rows: np.ndarray) -> list[np.ndarray]:
    poses = []
    for row in rows:
        tx, ty, tz = row[1:4]
        qx, qy, qz, qw = row[4:8]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        c2w[:3, 3] = [tx, ty, tz]
        poses.append(c2w)
    return poses


def associate_external_to_da3(da3_rows: np.ndarray, ext_rows: np.ndarray, max_delta: float) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    ext_times = ext_rows[:, 0]
    ext_poses = tum_rows_to_c2w(ext_rows)
    associations: list[dict[str, float]] = []
    matched_poses: list[np.ndarray] = []
    for da3_idx, da3_ts in enumerate(da3_rows[:, 0]):
        pos = bisect_left(ext_times, da3_ts)
        candidates = []
        if pos < len(ext_times):
            candidates.append((abs(ext_times[pos] - da3_ts), pos))
        if pos > 0:
            candidates.append((abs(ext_times[pos - 1] - da3_ts), pos - 1))
        if not candidates:
            raise RuntimeError(f"No external pose candidates found for DA3 frame {da3_idx}")
        delta, ext_idx = min(candidates, key=lambda item: item[0])
        if delta > max_delta:
            raise RuntimeError(
                f"Timestamp association failed for DA3 frame {da3_idx}: nearest external pose is {delta:.6f}s away"
            )
        matched_poses.append(ext_poses[ext_idx])
        associations.append(
            {
                "da3_frame_index": da3_idx,
                "da3_timestamp": float(da3_ts),
                "external_pose_index": int(ext_idx),
                "external_timestamp": float(ext_times[ext_idx]),
                "delta_s": float(delta),
            }
        )
    return matched_poses, associations


def interpolate_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, alpha: float) -> np.ndarray:
    interp = np.eye(4, dtype=np.float64)
    rot = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]], axis=0))
    slerp = Slerp([0.0, 1.0], rot)
    interp[:3, :3] = slerp([alpha]).as_matrix()[0]
    interp[:3, 3] = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    return interp


def interpolate_external_to_da3(
    da3_rows: np.ndarray,
    ext_rows: np.ndarray,
    allow_partial_overlap: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, float]]]:
    ext_times = ext_rows[:, 0]
    ext_poses = tum_rows_to_c2w(ext_rows)
    matched_indices: list[int] = []
    matched_poses: list[np.ndarray] = []
    associations: list[dict[str, float]] = []
    for da3_idx, da3_ts in enumerate(da3_rows[:, 0]):
        if da3_ts < ext_times[0] or da3_ts > ext_times[-1]:
            if allow_partial_overlap:
                continue
            raise RuntimeError(
                f"DA3 frame {da3_idx} timestamp {da3_ts:.9f} lies outside external trajectory range "
                f"[{ext_times[0]:.9f}, {ext_times[-1]:.9f}]"
            )
        pos = bisect_left(ext_times, da3_ts)
        if pos == 0:
            lower_idx = upper_idx = 0
            alpha = 0.0
            interp_pose = ext_poses[0]
        elif pos == len(ext_times):
            lower_idx = upper_idx = len(ext_times) - 1
            alpha = 0.0
            interp_pose = ext_poses[-1]
        elif math.isclose(ext_times[pos], da3_ts, abs_tol=1e-9):
            lower_idx = upper_idx = pos
            alpha = 0.0
            interp_pose = ext_poses[pos]
        else:
            lower_idx = pos - 1
            upper_idx = pos
            span = ext_times[upper_idx] - ext_times[lower_idx]
            alpha = 0.0 if span <= 0 else float((da3_ts - ext_times[lower_idx]) / span)
            interp_pose = interpolate_pose(ext_poses[lower_idx], ext_poses[upper_idx], alpha)
        matched_indices.append(da3_idx)
        matched_poses.append(interp_pose)
        associations.append(
            {
                "da3_frame_index": int(da3_idx),
                "da3_timestamp": float(da3_ts),
                "external_lower_index": int(lower_idx),
                "external_lower_timestamp": float(ext_times[lower_idx]),
                "external_upper_index": int(upper_idx),
                "external_upper_timestamp": float(ext_times[upper_idx]),
                "interpolation_alpha": float(alpha),
            }
        )
    if not matched_indices:
        raise RuntimeError("No overlapping DA3 timestamps found for interpolation.")
    return da3_rows[np.asarray(matched_indices, dtype=int)], matched_poses, associations


def save_camera_pose_txt(path: Path, poses: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pose in poses:
            handle.write(" ".join(str(x) for x in pose.flatten()) + "\n")


def save_tum_trajectory(path: Path, timestamps: np.ndarray, poses: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for ts, pose in zip(timestamps, poses):
            rot = pose[:3, :3]
            tx, ty, tz = pose[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(rot)
            handle.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def create_npz_subset(source_npz_dir: Path, target_npz_dir: Path, frame_indices: list[int]) -> None:
    target_npz_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, source_idx in enumerate(frame_indices):
        source = source_npz_dir / f"frame_{source_idx}.npz"
        if not source.exists():
            raise FileNotFoundError(f"Missing source npz frame: {source}")
        target = target_npz_dir / f"frame_{local_idx}.npz"
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(source, target)


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


def load_euroc_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamp = int(row[0]) / 1_000_000_000.0
            position = np.array([float(v) for v in row[1:4]], dtype=np.float64)
            entries.append((timestamp, position))
    return entries


def associate_to_groundtruth(est_timestamps: np.ndarray, gt_entries: list[tuple[float, np.ndarray]], max_delta: float) -> list[tuple[int, int]]:
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


def read_ascii_ply_xyzrgb(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in PLY header: {path}")

        rows = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            payload = {name: parts[idx] for idx, name in enumerate(properties[: len(parts)])}
            x = float(payload["x"])
            y = float(payload["y"])
            z = float(payload["z"])
            r = float(payload.get("diffuse_red", payload.get("red", 255)))
            g = float(payload.get("diffuse_green", payload.get("green", 255)))
            b = float(payload.get("diffuse_blue", payload.get("blue", 255)))
            rows.append([x, y, z, r, g, b])
    return np.asarray(rows, dtype=np.float64)


def read_binary_ply_xyzrgb(path: Path) -> np.ndarray:
    import struct

    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in header: {path}")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
        if vertex_count is None:
            raise ValueError(f"No vertex count in PLY header: {path}")
        record = struct.Struct("<fffBBB")
        data = np.empty((vertex_count, 6), dtype=np.float64)
        for idx in range(vertex_count):
            chunk = handle.read(record.size)
            if len(chunk) != record.size:
                raise ValueError(f"Unexpected EOF in PLY body: {path}")
            x, y, z, r, g, b = record.unpack(chunk)
            data[idx] = [x, y, z, r, g, b]
    return data


def write_binary_ply_xyzrgb(path: Path, cloud: np.ndarray) -> None:
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {len(cloud)}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "",
            ]
        )
        handle.write(header.encode("ascii"))
        record = struct.Struct("<fffBBB")
        for row in cloud:
            handle.write(
                record.pack(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    int(np.clip(row[3], 0, 255)),
                    int(np.clip(row[4], 0, 255)),
                    int(np.clip(row[5], 0, 255)),
                )
            )


def sample_cloud(cloud: np.ndarray, limit: int) -> np.ndarray:
    if len(cloud) <= limit:
        return cloud
    rng = np.random.default_rng(42)
    indices = rng.choice(len(cloud), size=limit, replace=False)
    return cloud[indices]


def evaluate_clouds(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict[str, float]:
    est_tree = cKDTree(est_xyz)
    gt_tree = cKDTree(gt_xyz)
    gt_to_est = est_tree.query(gt_xyz, k=1)[0]
    est_to_gt = gt_tree.query(est_xyz, k=1)[0]
    return {
        "gt_to_est_rmse_m": float(np.sqrt(np.mean(gt_to_est**2))),
        "gt_to_est_mean_m": float(np.mean(gt_to_est)),
        "est_to_gt_rmse_m": float(np.sqrt(np.mean(est_to_gt**2))),
        "est_to_gt_mean_m": float(np.mean(est_to_gt)),
        "symmetric_chamfer_m": float(np.mean(gt_to_est) + np.mean(est_to_gt)),
    }


def summarize_cloud_against_gt(
    gt_cloud: np.ndarray,
    est_cloud: np.ndarray,
    sim3_scale: float,
    sim3_rot: np.ndarray,
    sim3_trans: np.ndarray,
    output_dir: Path,
    prefix: str,
    cloud_sample: int,
) -> dict[str, float]:
    est_xyz = apply_sim3(est_cloud[:, :3], sim3_scale, sim3_rot, sim3_trans)
    est_cloud_aligned = np.concatenate([est_xyz, est_cloud[:, 3:6]], axis=1)
    write_binary_ply_xyzrgb(output_dir / f"{prefix}_cloud_sim3_aligned.ply", est_cloud_aligned)

    gt_cloud_sampled = sample_cloud(gt_cloud, cloud_sample)
    est_cloud_sampled = sample_cloud(est_cloud_aligned, cloud_sample)
    cloud_summary = evaluate_clouds(est_cloud_sampled[:, :3], gt_cloud_sampled[:, :3])
    cloud_summary.update(
        {
            "gt_point_count": int(len(gt_cloud)),
            "estimate_point_count": int(len(est_cloud)),
            "sampled_point_count": int(min(cloud_sample, len(gt_cloud), len(est_cloud))),
        }
    )
    return cloud_summary


def export_point_cloud(args: argparse.Namespace, pose_file: Path, output_dir: Path, npz_folder: Path | None = None) -> None:
    npz_folder = npz_folder or (args.da3_run_dir / "results_output")
    output_ply = output_dir / "combined_pcd_external.ply"
    command = []
    if args.conda_env:
        command.extend(["conda", "run", "-n", args.conda_env, "python"])
    else:
        command.append(args.python_cmd)
    command.extend(
        [
            str(resolve_path(args.npz_process_script)),
            "--npz_folder",
            str(npz_folder),
            "--pose_file",
            str(pose_file),
            "--output_file",
            str(output_ply),
            "--conf_threshold_coef",
            str(args.conf_threshold_coef),
            "--sample_ratio",
            str(args.sample_ratio),
        ]
    )
    child_env = os.environ.copy()
    upstream_src = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "src"
    child_env["PYTHONPATH"] = (
        f"{upstream_src}:{child_env['PYTHONPATH']}"
        if child_env.get("PYTHONPATH")
        else str(upstream_src)
    )
    child_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_env)


def main() -> int:
    args = parse_args()
    args.da3_run_dir = resolve_path(args.da3_run_dir)
    args.external_traj = resolve_path(args.external_traj)
    args.sequence_dir = resolve_path(args.sequence_dir)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    da3_estimate = load_tum_pose_file(args.da3_run_dir / "eval_euroc" / "estimate_raw_tum.txt")
    ext_estimate = load_tum_pose_file(args.external_traj)

    if args.association_mode == "interpolate":
        da3_eval_rows, external_poses, associations = interpolate_external_to_da3(
            da3_estimate,
            ext_estimate,
            allow_partial_overlap=args.allow_partial_overlap,
        )
    else:
        external_poses, associations = associate_external_to_da3(da3_estimate, ext_estimate, args.max_assoc_delta)
        da3_eval_rows = da3_estimate
    pose_txt = args.output_dir / "camera_poses_external.txt"
    save_camera_pose_txt(pose_txt, external_poses)
    save_tum_trajectory(args.output_dir / "estimate_external_tum.txt", da3_eval_rows[:, 0], external_poses)

    gt_entries = load_euroc_groundtruth(args.sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv")
    matches = associate_to_groundtruth(da3_eval_rows[:, 0], gt_entries, args.max_assoc_delta)
    if len(matches) < 3:
        raise RuntimeError(f"Only {len(matches)} GT matches found; need at least 3")

    est_pts = np.array([external_poses[i][:3, 3] for i, _ in matches], dtype=np.float64)
    gt_pts = np.array([gt_entries[j][1] for _, j in matches], dtype=np.float64)
    se3_scale, se3_rot, se3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=False)
    sim3_scale, sim3_rot, sim3_trans = umeyama_alignment(est_pts, gt_pts, with_scale=True)
    se3_errors = np.linalg.norm(apply_sim3(est_pts, se3_scale, se3_rot, se3_trans) - gt_pts, axis=1)
    sim3_errors = np.linalg.norm(apply_sim3(est_pts, sim3_scale, sim3_rot, sim3_trans) - gt_pts, axis=1)

    gt_path_len = trajectory_path_length(gt_pts)
    est_path_len = trajectory_path_length(est_pts)
    summary = {
        "sequence": args.sequence_dir.name,
        "source_da3_run_dir": str(args.da3_run_dir),
        "external_pose_trajectory": str(args.external_traj),
        "association_mode": args.association_mode,
        "association_count": len(associations),
        "da3_frame_count_used": int(len(da3_eval_rows)),
        "da3_frame_count_total": int(len(da3_estimate)),
        "gt_match_count": len(matches),
        "se3": summarize_errors(se3_errors),
        "sim3": summarize_errors(sim3_errors),
        "sim3_scale": sim3_scale,
        "path_ratio_est_over_gt": (est_path_len / gt_path_len) if gt_path_len > 0 else None,
        "alignment": {
            "se3": {"scale": se3_scale, "rotation": se3_rot.tolist(), "translation": se3_trans.tolist()},
            "sim3": {"scale": sim3_scale, "rotation": sim3_rot.tolist(), "translation": sim3_trans.tolist()},
        },
    }
    (args.output_dir / "trajectory_summary_external.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "association_report.json").write_text(json.dumps(associations, indent=2), encoding="utf-8")

    if args.export_ply:
        npz_folder = args.da3_run_dir / "results_output"
        if len(da3_eval_rows) != len(da3_estimate):
            subset_indices = [entry["da3_frame_index"] for entry in associations]
            subset_dir = args.output_dir / "results_output_overlap"
            create_npz_subset(npz_folder, subset_dir, subset_indices)
            npz_folder = subset_dir
        export_point_cloud(args, pose_txt, args.output_dir, npz_folder=npz_folder)

    if args.with_pointcloud:
        gt_cloud_path = args.sequence_dir / "mav0" / "pointcloud0" / "data.ply"
        if not gt_cloud_path.exists():
            raise FileNotFoundError(f"Ground-truth cloud not found: {gt_cloud_path}")
        gt_cloud = read_ascii_ply_xyzrgb(gt_cloud_path)
        write_binary_ply_xyzrgb(args.output_dir / "gt_cloud.ply", gt_cloud)

        original_cloud = read_binary_ply_xyzrgb(args.da3_run_dir / "pcd" / "combined_pcd.ply")
        original_summary = json.loads((args.da3_run_dir / "eval_euroc" / "trajectory_summary.json").read_text(encoding="utf-8"))
        orig_sim3 = original_summary["alignment"]["sim3"]
        original_cloud_summary = summarize_cloud_against_gt(
            gt_cloud=gt_cloud,
            est_cloud=original_cloud,
            sim3_scale=float(orig_sim3["scale"]),
            sim3_rot=np.asarray(orig_sim3["rotation"], dtype=np.float64),
            sim3_trans=np.asarray(orig_sim3["translation"], dtype=np.float64),
            output_dir=args.output_dir,
            prefix="original_da3",
            cloud_sample=args.cloud_sample,
        )
        (args.output_dir / "pointcloud_summary_original_da3.json").write_text(
            json.dumps(original_cloud_summary, indent=2), encoding="utf-8"
        )

        external_cloud_path = args.output_dir / "combined_pcd_external.ply"
        if external_cloud_path.exists():
            external_cloud = read_binary_ply_xyzrgb(external_cloud_path)
            external_cloud_summary = summarize_cloud_against_gt(
                gt_cloud=gt_cloud,
                est_cloud=external_cloud,
                sim3_scale=sim3_scale,
                sim3_rot=sim3_rot,
                sim3_trans=sim3_trans,
                output_dir=args.output_dir,
                prefix="external_pose_da3",
                cloud_sample=args.cloud_sample,
            )
            (args.output_dir / "pointcloud_summary_external_da3.json").write_text(
                json.dumps(external_cloud_summary, indent=2), encoding="utf-8"
            )

    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
