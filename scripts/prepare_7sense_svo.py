#!/usr/bin/env python3
"""Prepare extracted 7sense sequences for SVO mono benchmarking."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "7sense"
SVO_BENCH_ROOT = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_benchmarking"
SVO_DATA_ROOT = SVO_BENCH_ROOT / "data" / "mono3d" / "7sense" / "mono"
SVO_EXP_ROOT = SVO_BENCH_ROOT / "experiments"
BASE_EXP_SOURCE = SVO_EXP_ROOT / "exp_euroc_nolc.yaml"
MONO_EXP_NAME = "mono3d_7sense_mono"
ASSUMED_SOURCE_FPS = 30.0
DEFAULT_INTRINSICS = {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 7sense sequences for SVO mono benchmarking.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Scene name, e.g. chess")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Specific sequence identifier, e.g. chess/seq-01 or redkitchen_seq-14",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to prepare when --sequence is not provided.",
    )
    parser.add_argument("--write-configs", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def ensure_relative_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target_path.resolve(), start=link_path.parent.resolve())
    link_path.symlink_to(relative_target)


def split_file_for(scene_dir: Path, split: str) -> Path:
    if split == "train":
        return scene_dir / "TrainSplit.txt"
    if split == "test":
        return scene_dir / "TestSplit.txt"
    raise ValueError(f"unsupported split: {split}")


def split_entries(scene_dir: Path, split: str) -> list[str]:
    if split == "all":
        return sorted(path.name for path in scene_dir.iterdir() if path.is_dir() and path.name.startswith("seq-"))
    path = split_file_for(scene_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    selected: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            match = re.search(r"(\d+)$", stripped)
            if not match:
                continue
            selected.append(f"seq-{int(match.group(1)):02d}")
    return selected


def parse_sequence_filter(value: str) -> tuple[str, str]:
    if "/" in value:
        scene, seq_name = value.split("/", 1)
        return scene, seq_name
    match = re.match(r"(.+)_seq-(\d+)$", value)
    if match:
        return match.group(1), f"seq-{int(match.group(2)):02d}"
    raise ValueError(f"Sequence filter must look like scene/seq-01 or scene_seq-01, got: {value}")


def selected_sequences(root: Path, scenes: list[str], sequence_filters: list[str], split: str) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing 7sense root: {root}")
    requested_scenes = set(scenes)
    selected: list[tuple[str, Path]] = []
    if sequence_filters:
        for item in sequence_filters:
            scene_name, seq_name = parse_sequence_filter(item)
            scene_dir = root / scene_name
            seq_dir = scene_dir / seq_name
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing extracted sequence: {seq_dir}")
            selected.append((scene_name, seq_dir))
        return sorted(selected)

    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested_scenes and scene_dir.name not in requested_scenes:
            continue
        for seq_name in split_entries(scene_dir, split):
            seq_dir = scene_dir / seq_name
            if not seq_dir.exists():
                raise FileNotFoundError(f"Missing extracted sequence directory: {seq_dir}")
            selected.append((scene_dir.name, seq_dir))
    if not selected:
        raise FileNotFoundError("No 7sense sequences matched the requested filters.")
    return selected


def image_size_from_png(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Unsupported or invalid PNG header in {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def frame_index_from_name(path: Path) -> int:
    match = re.search(r"frame-(\d+)\.color\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected frame name: {path.name}")
    return int(match.group(1))


def ordered_color_frames(seq_dir: Path) -> list[Path]:
    frames = sorted(seq_dir.glob("frame-*.color.png"))
    if not frames:
        raise FileNotFoundError(f"No color frames found in {seq_dir}")
    return frames


def rotation_matrix_to_quaternion(rotation: list[list[float]]) -> list[float]:
    m = rotation
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return [qx / norm, qy / norm, qz / norm, qw / norm]


def load_pose(path: Path) -> tuple[list[float], list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(item) for item in stripped.split()])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise RuntimeError(f"Unexpected pose matrix shape in {path}")
    rotation = [row[:3] for row in rows[:3]]
    translation = [rows[0][3], rows[1][3], rows[2][3]]
    quaternion = rotation_matrix_to_quaternion(rotation)
    return translation, quaternion


def write_images_txt(frames: list[Path], out_path: Path) -> tuple[int, list[tuple[int, float]]]:
    timestamps: list[tuple[int, float]] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for dataset_idx, frame_path in enumerate(frames):
            timestamp = frame_index_from_name(frame_path) / ASSUMED_SOURCE_FPS
            handle.write(f"{dataset_idx} {timestamp:.6f} img/{frame_path.name}\n")
            timestamps.append((dataset_idx, timestamp))
    return len(frames), timestamps


def write_groundtruth(frames: list[Path], out_groundtruth: Path, out_stamped: Path) -> None:
    with out_groundtruth.open("w", encoding="utf-8") as gt_handle, out_stamped.open(
        "w", encoding="utf-8"
    ) as stamped_handle:
        for dataset_idx, frame_path in enumerate(frames):
            timestamp = frame_index_from_name(frame_path) / ASSUMED_SOURCE_FPS
            pose_path = frame_path.with_suffix("").with_suffix(".pose.txt")
            translation, quaternion = load_pose(pose_path)
            gt_handle.write(
                f"{dataset_idx} {timestamp:.6f} "
                f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )
            stamped_handle.write(
                f"{timestamp:.6f} "
                f"{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
                f"{quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}\n"
            )


def write_groundtruth_matches(frames: list[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for dataset_idx, _ in enumerate(frames):
            handle.write(f"{dataset_idx} {dataset_idx}\n")


def make_calib(sequence_name: str, width: int, height: int) -> dict[str, Any]:
    return {
        "label": f"7sense {sequence_name}",
        "id": sequence_name,
        "cameras": [
            {
                "camera": {
                    "label": "cam0",
                    "id": f"{sequence_name}_cam0",
                    "line-delay-nanoseconds": 0,
                    "image_height": height,
                    "image_width": width,
                    "type": "pinhole",
                    "intrinsics": {
                        "cols": 1,
                        "rows": 4,
                        "data": [
                            DEFAULT_INTRINSICS["fx"],
                            DEFAULT_INTRINSICS["fy"],
                            DEFAULT_INTRINSICS["cx"],
                            DEFAULT_INTRINSICS["cy"],
                        ],
                    },
                    "distortion": {
                        "type": "radial-tangential",
                        "parameters": {"cols": 1, "rows": 4, "data": [0.0, 0.0, 0.0, 0.0]},
                    },
                },
                "T_B_C": {
                    "cols": 4,
                    "rows": 4,
                    "data": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
            }
        ],
    }


def prepare_sequence(scene_name: str, seq_dir: Path) -> dict[str, Any]:
    dataset_id = f"{scene_name}_{seq_dir.name}"
    frames = ordered_color_frames(seq_dir)
    width, height = image_size_from_png(frames[0])
    out_dir = SVO_DATA_ROOT / dataset_id
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_relative_symlink(data_dir / "img", seq_dir)

    image_count, timestamps = write_images_txt(frames, data_dir / "images.txt")
    write_groundtruth(frames, data_dir / "groundtruth.txt", data_dir / "stamped_groundtruth.txt")
    write_groundtruth_matches(frames, data_dir / "groundtruth_matches.txt")
    dump_yaml(out_dir / "calib.yaml", make_calib(dataset_id, width, height))
    dump_yaml(
        out_dir / "dataset.yaml",
        {"dataset_name": dataset_id, "dataset_first_frame": 0, "dataset_last_frame": image_count},
    )

    return {
        "scene_name": scene_name,
        "sequence_name": seq_dir.name,
        "dataset_id": dataset_id,
        "dataset_name": f"mono3d/7sense/mono/{dataset_id}",
        "dataset_directory": str(out_dir),
        "dataset_first_frame": 0,
        "dataset_last_frame": image_count,
        "counts": {"images": image_count, "groundtruth": len(timestamps)},
        "assumed_source_fps": ASSUMED_SOURCE_FPS,
    }


def make_experiment(base: dict[str, Any], prepared: list[dict[str, Any]]) -> dict[str, Any]:
    settings = dict(base["settings"])
    settings["dataset_is_stereo"] = False
    settings["pipeline_is_stereo"] = False
    settings["trace_statistics"] = False
    settings["calib_name"] = "calib.yaml"
    settings["runlc"] = False
    settings["use_imu"] = False
    settings["trace_only_keyframes"] = False
    settings["dataset_first_frame"] = 0
    settings["use_ceres_backend"] = False
    settings["poseoptim_prior_lambda"] = 0.0
    settings["img_align_prior_lambda_rot"] = 0.0
    settings["img_align_prior_lambda_trans"] = 0.0

    trace_root = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks" / MONO_EXP_NAME
    return {
        "experiment_label": MONO_EXP_NAME,
        "ros_node": "svo_ros",
        "ros_node_name": "svo_benchmark",
        "flags": {"v": 0, "logtostderr": 0},
        "trace_base_dir": str(trace_root),
        "settings": settings,
        "datasets": [
            {
                "name": item["dataset_name"],
                "settings": {
                    "dataset_first_frame": item["dataset_first_frame"],
                    "dataset_last_frame": item["dataset_last_frame"],
                },
            }
            for item in prepared
        ],
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    selected = selected_sequences(root, args.scene, args.sequence, args.split)
    prepared_payload = [prepare_sequence(scene_name, seq_dir) for scene_name, seq_dir in selected]
    payload: dict[str, Any] = {
        "seven_sense_root": str(root),
        "split": args.split,
        "prepared_datasets": prepared_payload,
    }

    if args.write_configs:
        base_exp = load_yaml(BASE_EXP_SOURCE)
        exp = make_experiment(base_exp, prepared_payload)
        mono_path = SVO_EXP_ROOT / f"{MONO_EXP_NAME}.yaml"
        dump_yaml(mono_path, exp)
        payload["generated_experiments"] = [str(mono_path)]

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
