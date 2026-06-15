#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "datasets" / "euroc_prepared" / "V1_01_easy_fps5"
DEFAULT_SEQUENCE_DIR = REPO_ROOT / "datasets" / "euroc" / "V1_01_easy"
DEFAULT_TRACE_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks_isolated"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "euroc_api_pose_conditioned"
DEFAULT_MODEL_DIR = Path(
    "/home/qcar/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run plain DA3 API in pose-conditioned mode on EuRoC.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--external-traj", type=Path, help="Optional SVO stamped_traj_estimate.txt")
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--tag", default="isolated_v1_01")
    parser.add_argument("--mode", choices=("mono", "mono-imu"), default="mono-imu")
    parser.add_argument("--sequence", default="V1_01_easy")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pose-source",
        choices=("svo", "da3"),
        default="svo",
        help="Use external SVO poses or let plain DA3 estimate poses itself.",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--ref-view-strategy", default="middle")
    parser.add_argument("--export-format", default="mini_npz-glb-depth_vis")
    parser.add_argument("--align-to-input-ext-scale", action="store_true", default=True)
    parser.add_argument("--no-align-to-input-ext-scale", dest="align_to_input_ext_scale", action="store_false")
    parser.add_argument("--overlap-only", action="store_true", default=True)
    parser.add_argument("--no-overlap-only", dest="overlap_only", action="store_false")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset within the overlap-filtered frame list.")
    parser.add_argument("--max-frames", type=int, default=96, help="Maximum frames to pass to plain DA3 for this test.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride within the overlap-filtered frame list.")
    return parser.parse_args()


def latest_trace_dir(root: Path, experiment_name: str, sequence: str) -> Path:
    exp_root = root / experiment_name
    candidates = [path for path in exp_root.iterdir() if path.is_dir()] if exp_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No isolated SVO runs found under {exp_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    trace_dir = latest / sequence
    if not trace_dir.exists():
        raise FileNotFoundError(f"Expected sequence trace not found: {trace_dir}")
    return trace_dir


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


def interpolate_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, alpha: float) -> np.ndarray:
    interp = np.eye(4, dtype=np.float64)
    rot = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]], axis=0))
    slerp = Slerp([0.0, 1.0], rot)
    interp[:3, :3] = slerp([alpha]).as_matrix()[0]
    interp[:3, 3] = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    return interp


def image_timestamp_from_path(path: Path) -> float:
    return int(path.stem) / 1_000_000_000.0


def build_pose_conditioned_subset(
    image_paths: list[Path],
    ext_rows: np.ndarray,
    overlap_only: bool,
) -> tuple[list[Path], np.ndarray]:
    ext_times = ext_rows[:, 0]
    ext_poses = tum_rows_to_c2w(ext_rows)
    selected_images: list[Path] = []
    extrinsics_w2c: list[np.ndarray] = []
    for image_path in image_paths:
        ts = image_timestamp_from_path(image_path)
        if ts < ext_times[0] or ts > ext_times[-1]:
            if overlap_only:
                continue
            raise RuntimeError(
                f"Image timestamp {ts:.9f} for {image_path.name} lies outside external trajectory range "
                f"[{ext_times[0]:.9f}, {ext_times[-1]:.9f}]"
            )
        pos = bisect_left(ext_times, ts)
        if pos == 0:
            interp_c2w = ext_poses[0]
        elif pos == len(ext_times):
            interp_c2w = ext_poses[-1]
        elif math.isclose(ext_times[pos], ts, abs_tol=1e-9):
            interp_c2w = ext_poses[pos]
        else:
            lower_idx = pos - 1
            upper_idx = pos
            span = ext_times[upper_idx] - ext_times[lower_idx]
            alpha = 0.0 if span <= 0 else float((ts - ext_times[lower_idx]) / span)
            interp_c2w = interpolate_pose(ext_poses[lower_idx], ext_poses[upper_idx], alpha)
        selected_images.append(image_path)
        extrinsics_w2c.append(np.linalg.inv(interp_c2w))
    if not selected_images:
        raise RuntimeError("No overlapping image timestamps found for pose-conditioned DA3 run.")
    return selected_images, np.stack(extrinsics_w2c, axis=0)


def load_euroc_intrinsics(sequence_dir: Path, count: int) -> np.ndarray:
    sensor_yaml = sequence_dir / "mav0" / "cam0" / "sensor.yaml"
    lines = sensor_yaml.read_text(encoding="utf-8").splitlines()
    intrinsics_line = next(line for line in lines if line.strip().startswith("intrinsics:"))
    values = intrinsics_line.split("[", 1)[1].split("]", 1)[0]
    fx, fy, cx, cy = [float(x.strip()) for x in values.split(",")]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.stack([k.copy() for _ in range(count)], axis=0)


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    sequence_dir = args.sequence_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    external_traj: Path | None = None
    subset_w2c: np.ndarray | None = None

    if args.pose_source == "svo":
        if args.external_traj:
            external_traj = args.external_traj.resolve()
        else:
            exp_name = f"mono3d_euroc_{args.tag}_{'mono_imu' if args.mode == 'mono-imu' else 'mono'}"
            trace_dir = latest_trace_dir(args.trace_root.resolve(), exp_name, args.sequence)
            external_traj = trace_dir / "stamped_traj_estimate.txt"
        if not external_traj.exists():
            raise FileNotFoundError(f"Missing external trajectory: {external_traj}")
        ext_rows = load_tum_pose_file(external_traj)
        overlap_images, overlap_w2c = build_pose_conditioned_subset(image_paths, ext_rows, args.overlap_only)
        if args.stride < 1:
            raise ValueError("--stride must be >= 1")
        subset_images = overlap_images[args.start_index :: args.stride]
        subset_w2c = overlap_w2c[args.start_index :: args.stride]
    else:
        if args.stride < 1:
            raise ValueError("--stride must be >= 1")
        subset_images = image_paths[args.start_index :: args.stride]

    if args.max_frames > 0:
        subset_images = subset_images[: args.max_frames]
        if subset_w2c is not None:
            subset_w2c = subset_w2c[: args.max_frames]
    if not subset_images:
        raise RuntimeError("Selected DA3 subset is empty.")

    intrinsics = load_euroc_intrinsics(sequence_dir, len(subset_images))

    run_name = (
        f"{args.sequence}_plain_da3_{args.tag}_{args.mode}_{args.pose_source}"
        f"_start{args.start_index}_stride{args.stride}_n{len(subset_images)}"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_images.txt").write_text(
        "\n".join(str(path) for path in subset_images) + "\n",
        encoding="utf-8",
    )
    if subset_w2c is not None:
        np.save(output_dir / "input_extrinsics_w2c.npy", subset_w2c.astype(np.float32))
    np.save(output_dir / "input_intrinsics.npy", intrinsics.astype(np.float32))
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "image_dir": str(image_dir),
                "sequence_dir": str(sequence_dir),
                "pose_source": args.pose_source,
                "external_traj": str(external_traj) if external_traj is not None else None,
                "model_dir": str(model_dir),
                "selected_frame_count": len(subset_images),
                "start_index": args.start_index,
                "max_frames": args.max_frames,
                "stride": args.stride,
                "overlap_only": args.overlap_only,
                "export_format": args.export_format,
                "process_res": args.process_res,
                "ref_view_strategy": args.ref_view_strategy,
                "align_to_input_ext_scale": args.align_to_input_ext_scale,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    upstream_src = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "src"
    if str(upstream_src) not in sys.path:
        sys.path.insert(0, str(upstream_src))
    from depth_anything_3.services.inference_service import run_inference

    run_inference(
        image_paths=[str(path) for path in subset_images],
        export_dir=str(output_dir),
        model_dir=str(model_dir),
        device="cuda",
        export_format=args.export_format,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        extrinsics=None if subset_w2c is None else subset_w2c.astype(np.float32),
        intrinsics=None if subset_w2c is None else intrinsics.astype(np.float32),
        align_to_input_ext_scale=args.align_to_input_ext_scale,
        use_ray_pose=False,
        ref_view_strategy=args.ref_view_strategy,
        conf_thresh_percentile=40.0,
        num_max_points=1_000_000,
        show_cameras=True,
        feat_vis_fps=15,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
