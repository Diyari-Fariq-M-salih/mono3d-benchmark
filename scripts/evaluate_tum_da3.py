#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from bisect import bisect_left
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TUM_ROOT = REPO_ROOT / "datasets" / "tum_rgbd"

TUM_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DA3-Streaming TUM run against TUM RGB-D trajectory and point-cloud ground truth."
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="DA3 run folder.")
    parser.add_argument(
        "--prepared-dir",
        required=True,
        type=Path,
        help="Prepared TUM frame folder, e.g. datasets/tum_rgbd_prepared/freiburg1_desk_fps2",
    )
    parser.add_argument(
        "--tum-root",
        default=DEFAULT_TUM_ROOT,
        type=Path,
        help="Root directory containing extracted TUM sequences.",
    )
    parser.add_argument(
        "--tum-sequence-dir",
        type=Path,
        help="Override extracted TUM sequence directory. Defaults to datasets/tum_rgbd/rgbd_dataset_<name>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for evaluation outputs. Defaults to <run-dir>/eval_tum.",
    )
    parser.add_argument(
        "--max-assoc-delta",
        type=float,
        default=0.02,
        help="Maximum timestamp association delta in seconds.",
    )
    parser.add_argument(
        "--with-pointcloud",
        action="store_true",
        help="Also build a ground-truth point cloud from TUM depth and compare to the DA3 point cloud.",
    )
    parser.add_argument(
        "--depth-max-m",
        type=float,
        default=5.0,
        help="Maximum GT depth to include when building the TUM point cloud.",
    )
    parser.add_argument(
        "--pixel-stride",
        type=int,
        default=4,
        help="Sample every Nth pixel when building the GT point cloud.",
    )
    parser.add_argument(
        "--cloud-sample",
        type=int,
        default=200000,
        help="Maximum points sampled per cloud for point-cloud metrics.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def infer_tum_sequence_dir(prepared_dir: Path, tum_root: Path) -> Path:
    base = prepared_dir.name
    if "_fps" not in base:
        raise ValueError(f"Prepared dir name must end with _fpsX: {prepared_dir}")
    stem = base.rsplit("_fps", 1)[0]
    return tum_root / f"rgbd_dataset_{stem}"


def read_prepared_timestamps(prepared_dir: Path) -> list[tuple[float, Path]]:
    frames = []
    for frame_path in sorted(prepared_dir.glob("frame_*.png")):
        source_path = frame_path.resolve()
        timestamp = float(source_path.stem)
        frames.append((timestamp, source_path))
    if not frames:
        raise FileNotFoundError(f"No prepared frames found in {prepared_dir}")
    return frames


def load_tum_pose_file(path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
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


def load_tum_index_file(path: Path) -> list[tuple[float, Path]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), path.parent / parts[1]))
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


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


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
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, pose in zip(timestamps, poses):
            q = rotation_matrix_to_quaternion(pose[:3, :3])
            t = pose[:3, 3]
            handle.write(
                f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
            )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def infer_intrinsics(sequence_name: str) -> dict[str, float]:
    for prefix, intrinsics in TUM_INTRINSICS.items():
        if sequence_name.startswith(prefix):
            return intrinsics
    raise ValueError(f"No TUM intrinsics mapping for sequence {sequence_name}")


def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def build_gt_cloud(
    tum_sequence_dir: Path,
    prepared_frames: list[tuple[float, Path]],
    gt_entries: list[tuple[float, np.ndarray, np.ndarray]],
    max_delta: float,
    pixel_stride: int,
    depth_max_m: float,
) -> np.ndarray:
    sequence_name = tum_sequence_dir.name.replace("rgbd_dataset_", "")
    intr = infer_intrinsics(sequence_name)
    depth_entries = load_tum_index_file(tum_sequence_dir / "depth.txt")
    depth_times = [ts for ts, _ in depth_entries]
    gt_times = [ts for ts, *_ in gt_entries]
    points_world = []

    for rgb_ts, rgb_path in prepared_frames:
        depth_pos = bisect_left(depth_times, rgb_ts)
        depth_candidates = []
        if depth_pos < len(depth_times):
            depth_candidates.append((abs(depth_times[depth_pos] - rgb_ts), depth_pos))
        if depth_pos > 0:
            depth_candidates.append((abs(depth_times[depth_pos - 1] - rgb_ts), depth_pos - 1))
        if not depth_candidates:
            continue
        depth_delta, depth_idx = min(depth_candidates, key=lambda item: item[0])
        if depth_delta > max_delta:
            continue

        gt_pos = bisect_left(gt_times, rgb_ts)
        gt_candidates = []
        if gt_pos < len(gt_times):
            gt_candidates.append((abs(gt_times[gt_pos] - rgb_ts), gt_pos))
        if gt_pos > 0:
            gt_candidates.append((abs(gt_times[gt_pos - 1] - rgb_ts), gt_pos - 1))
        if not gt_candidates:
            continue
        gt_delta, gt_idx = min(gt_candidates, key=lambda item: item[0])
        if gt_delta > max_delta:
            continue

        depth_img = load_png(depth_entries[depth_idx][1]).astype(np.uint16)
        rgb_img = load_png(rgb_path).astype(np.uint8)
        tx, ty, tz = gt_entries[gt_idx][1]
        rot = quaternion_to_matrix(gt_entries[gt_idx][2])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = rot
        c2w[:3, 3] = [tx, ty, tz]

        v = np.arange(0, depth_img.shape[0], pixel_stride)
        u = np.arange(0, depth_img.shape[1], pixel_stride)
        uu, vv = np.meshgrid(u, v)
        depth = depth_img[vv, uu].astype(np.float64) / 5000.0
        mask = (depth > 0.0) & (depth <= depth_max_m)
        if not np.any(mask):
            continue
        z = depth[mask]
        u_valid = uu[mask].astype(np.float64)
        v_valid = vv[mask].astype(np.float64)
        x = (u_valid - intr["cx"]) * z / intr["fx"]
        y = (v_valid - intr["cy"]) * z / intr["fy"]
        pts_cam = np.stack([x, y, z], axis=1)
        pts_world = (rot @ pts_cam.T).T + np.array([tx, ty, tz], dtype=np.float64)
        colors = rgb_img[vv[mask], uu[mask]]
        points_world.append(np.concatenate([pts_world, colors.astype(np.float64)], axis=1))

    if not points_world:
        raise RuntimeError("No GT cloud points were built from the selected TUM frames.")
    return np.concatenate(points_world, axis=0)


def read_binary_ply_xyzrgb(path: Path) -> np.ndarray:
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


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    prepared_dir = resolve_path(args.prepared_dir)
    tum_root = resolve_path(args.tum_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else run_dir / "eval_tum"
    output_dir.mkdir(parents=True, exist_ok=True)

    tum_sequence_dir = (
        resolve_path(args.tum_sequence_dir)
        if args.tum_sequence_dir
        else infer_tum_sequence_dir(prepared_dir, tum_root)
    )

    prepared_frames = read_prepared_timestamps(prepared_dir)
    gt_entries = load_tum_pose_file(tum_sequence_dir / "groundtruth.txt")
    camera_poses = load_da3_camera_poses(run_dir / "camera_poses.txt")

    est_timestamps = [ts for ts, _ in prepared_frames]
    if len(est_timestamps) != len(camera_poses):
        raise RuntimeError(
            f"Frame/pose count mismatch: {len(est_timestamps)} prepared frames vs {len(camera_poses)} poses"
        )

    est_raw_tum_path = output_dir / "estimate_raw_tum.txt"
    save_tum_trajectory(est_raw_tum_path, est_timestamps, camera_poses)

    matches = associate_by_timestamp(est_timestamps, gt_entries, args.max_assoc_delta)
    if len(matches) < 3:
        raise RuntimeError(f"Only {len(matches)} trajectory matches found; need at least 3")

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
        "prepared_dir": str(prepared_dir),
        "tum_sequence_dir": str(tum_sequence_dir),
        "match_count": len(matches),
        "se3": summarize_errors(se3_errors),
        "sim3": summarize_errors(sim3_errors),
        "sim3_scale": sim3_scale,
        "path_length_est_m": est_path_len,
        "path_length_gt_m": gt_path_len,
        "path_ratio_est_over_gt": est_path_len / gt_path_len if gt_path_len > 0 else None,
        "alignment": {
            "se3": {"scale": se3_scale, "rotation": se3_rot.tolist(), "translation": se3_trans.tolist()},
            "sim3": {"scale": sim3_scale, "rotation": sim3_rot.tolist(), "translation": sim3_trans.tolist()},
        },
    }
    write_json(output_dir / "trajectory_summary.json", trajectory_summary)

    if args.with_pointcloud:
        gt_cloud = build_gt_cloud(
            tum_sequence_dir=tum_sequence_dir,
            prepared_frames=prepared_frames,
            gt_entries=gt_entries,
            max_delta=args.max_assoc_delta,
            pixel_stride=args.pixel_stride,
            depth_max_m=args.depth_max_m,
        )
        write_binary_ply_xyzrgb(output_dir / "gt_cloud.ply", gt_cloud)

        est_cloud = read_binary_ply_xyzrgb(run_dir / "pcd" / "combined_pcd.ply")
        est_xyz = apply_sim3(est_cloud[:, :3], sim3_scale, sim3_rot, sim3_trans)
        est_cloud_aligned = np.concatenate([est_xyz, est_cloud[:, 3:6]], axis=1)
        write_binary_ply_xyzrgb(output_dir / "estimate_cloud_sim3_aligned.ply", est_cloud_aligned)

        gt_cloud_sampled = sample_cloud(gt_cloud, args.cloud_sample)
        est_cloud_sampled = sample_cloud(est_cloud_aligned, args.cloud_sample)
        cloud_summary = evaluate_clouds(est_cloud_sampled[:, :3], gt_cloud_sampled[:, :3])
        cloud_summary.update(
            {
                "gt_point_count": int(len(gt_cloud)),
                "estimate_point_count": int(len(est_cloud)),
                "sampled_point_count": int(min(args.cloud_sample, len(gt_cloud), len(est_cloud))),
            }
        )
        write_json(output_dir / "pointcloud_summary.json", cloud_summary)

    print(f"trajectory summary: {output_dir / 'trajectory_summary.json'}")
    if args.with_pointcloud:
        print(f"pointcloud summary: {output_dir / 'pointcloud_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
