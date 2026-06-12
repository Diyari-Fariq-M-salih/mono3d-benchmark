#!/usr/bin/env python3
"""Prepare QCar Evry RGB sequences for SVO monocular benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QCAR_ROOT = REPO_ROOT / "datasets" / "Qcar_evry"
SVO_BENCH_ROOT = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_benchmarking"
SVO_DATA_ROOT = SVO_BENCH_ROOT / "data" / "mono3d" / "qcar_evry" / "mono"
SVO_EXP_ROOT = SVO_BENCH_ROOT / "experiments"
BASE_EXP_SOURCE = SVO_EXP_ROOT / "exp_euroc_nolc.yaml"
MONO_EXP_NAME = "mono3d_qcar_evry_mono"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare QCar Evry sequences for SVO mono benchmarking.")
    parser.add_argument("--qcar-root", type=Path, default=DEFAULT_QCAR_ROOT)
    parser.add_argument("--sequence", action="append", default=[], help="Scene name, e.g. warehouse1")
    parser.add_argument("--write-configs", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def selected_sequences(qcar_root: Path, names: list[str]) -> list[Path]:
    candidates = sorted(
        path for path in qcar_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "rgb.txt").exists() and (path / "groundtruth.txt").exists()
    )
    if not names:
        if not candidates:
            raise FileNotFoundError(f"No QCar Evry sequences found in {qcar_root}")
        return candidates
    resolved = []
    for name in names:
        path = qcar_root / name
        if not (path / "rgb.txt").exists() or not (path / "groundtruth.txt").exists():
            raise FileNotFoundError(f"Missing QCar Evry sequence: {path}")
        resolved.append(path)
    return resolved


def load_rgb_index(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            rows.append((float(parts[0]), parts[1]))
    if not rows:
        raise FileNotFoundError(f"No RGB frames found in {path}")
    return rows


def load_groundtruth(path: Path) -> list[tuple[float, list[float]]]:
    rows: list[tuple[float, list[float]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            rows.append((float(parts[0]), [float(item) for item in parts[1:8]]))
    if not rows:
        raise FileNotFoundError(f"No ground-truth poses found in {path}")
    return rows


def ensure_relative_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target_path.resolve(), start=link_path.parent.resolve())
    link_path.symlink_to(relative_target)


def write_images_txt(entries: list[tuple[float, str]], out_path: Path) -> int:
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, (timestamp, rel_path) in enumerate(entries):
            handle.write(f"{idx} {timestamp:.6f} img/{Path(rel_path).name}\n")
    return len(entries)


def write_groundtruth_txt(
    gt_rows: list[tuple[float, list[float]]], out_groundtruth: Path, out_stamped: Path
) -> list[tuple[int, float]]:
    gt_index: list[tuple[int, float]] = []
    with out_groundtruth.open("w", encoding="utf-8") as gt_handle, out_stamped.open("w", encoding="utf-8") as stamped_handle:
        for idx, (timestamp, values) in enumerate(gt_rows):
            tx, ty, tz, qx, qy, qz, qw = values
            gt_handle.write(
                f"{idx} {timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            stamped_handle.write(
                f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
            gt_index.append((idx, timestamp))
    return gt_index


def write_groundtruth_matches(image_entries: list[tuple[float, str]], gt_index: list[tuple[int, float]], out_path: Path) -> None:
    gt_ptr = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for img_idx, (img_time, _) in enumerate(image_entries):
            while gt_ptr + 1 < len(gt_index) and abs(gt_index[gt_ptr + 1][1] - img_time) <= abs(gt_index[gt_ptr][1] - img_time):
                gt_ptr += 1
            handle.write(f"{img_idx} {gt_index[gt_ptr][0]}\n")


def load_camera_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_calib(camera_info: dict[str, Any], sequence_name: str) -> dict[str, Any]:
    k = camera_info["K"]
    d = list(camera_info.get("D", [0.0, 0.0, 0.0, 0.0, 0.0]))
    return {
        "label": f"QCar Evry {sequence_name}",
        "id": sequence_name,
        "cameras": [
            {
                "camera": {
                    "label": "cam0",
                    "id": f"{sequence_name}_cam0",
                    "line-delay-nanoseconds": 0,
                    "image_height": int(camera_info["height"]),
                    "image_width": int(camera_info["width"]),
                    "type": "pinhole",
                    "intrinsics": {"cols": 1, "rows": 4, "data": [k[0], k[4], k[2], k[5]]},
                    "distortion": {
                        "type": "radial-tangential",
                        "parameters": {"cols": 1, "rows": 4, "data": d[:4]},
                    },
                },
                "T_B_C": {
                    "cols": 4,
                    "rows": 4,
                    "data": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }


def prepare_sequence(sequence_dir: Path, camera_info: dict[str, Any]) -> dict[str, Any]:
    sequence_name = sequence_dir.name
    rgb_entries = load_rgb_index(sequence_dir / "rgb.txt")
    gt_rows = load_groundtruth(sequence_dir / "groundtruth.txt")

    out_dir = SVO_DATA_ROOT / sequence_name
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_relative_symlink(data_dir / "img", sequence_dir / "rgb")

    image_count = write_images_txt(rgb_entries, data_dir / "images.txt")
    gt_index = write_groundtruth_txt(gt_rows, data_dir / "groundtruth.txt", data_dir / "stamped_groundtruth.txt")
    write_groundtruth_matches(rgb_entries, gt_index, data_dir / "groundtruth_matches.txt")
    dump_yaml(out_dir / "calib.yaml", make_calib(camera_info, sequence_name))
    dump_yaml(
        out_dir / "dataset.yaml",
        {"dataset_name": sequence_name, "dataset_first_frame": 0, "dataset_last_frame": image_count},
    )

    return {
        "sequence_name": sequence_name,
        "dataset_name": f"mono3d/qcar_evry/mono/{sequence_name}",
        "dataset_directory": str(out_dir),
        "dataset_first_frame": 0,
        "dataset_last_frame": image_count,
        "counts": {"images": image_count, "groundtruth": len(gt_index)},
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
    qcar_root = args.qcar_root.resolve()
    camera_info = load_camera_info(qcar_root / "camera_info.txt")
    selected = selected_sequences(qcar_root, args.sequence)
    prepared_payload = [prepare_sequence(path, camera_info) for path in selected]
    payload: dict[str, Any] = {"qcar_root": str(qcar_root), "prepared_datasets": prepared_payload}

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
