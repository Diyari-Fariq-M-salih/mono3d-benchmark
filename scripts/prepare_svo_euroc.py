#!/usr/bin/env python3
"""Prepare extracted EuRoC sequences for SVO benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EUROC_ROOT = REPO_ROOT / "datasets" / "euroc"
SVO_BENCH_ROOT = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_benchmarking"
SVO_DATA_ROOT = SVO_BENCH_ROOT / "data" / "mono3d" / "euroc" / "mono"
SVO_EXP_ROOT = SVO_BENCH_ROOT / "experiments"
CALIB_SOURCE = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_ros" / "param" / "calib" / "euroc_mono.yaml"
BASE_EXP_SOURCE = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_benchmarking" / "experiments" / "exp_euroc_nolc.yaml"

MONO_EXP_NAME = "mono3d_euroc_mono"
MONO_IMU_EXP_NAME = "mono3d_euroc_mono_imu"

FIRST_FRAME_OVERRIDES = {
    "MH_01": 0,
    "MH_02": 0,
    "MH_03": 300,
    "MH_04": 340,
    "MH_05": 350,
    "V1_01": 0,
    "V1_02": 0,
    "V1_03": 0,
    "V2_01": 0,
    "V2_02": 100,
    "V2_03": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare extracted EuRoC data for SVO benchmarking.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EUROC_ROOT)
    parser.add_argument("--sequence", action="append", default=[], help="Specific sequence(s) to prepare, e.g. MH_01_easy")
    parser.add_argument("--write-configs", action="store_true", help="Also generate benchmark experiment YAMLs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def discover_sequences(dataset_root: Path) -> list[Path]:
    return sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "mav0" / "cam0" / "data.csv").exists()
    )


def selected_sequences(dataset_root: Path, names: list[str]) -> list[Path]:
    if not names:
        return discover_sequences(dataset_root)
    resolved = []
    for name in names:
        seq_dir = dataset_root / name
        if not (seq_dir / "mav0" / "cam0" / "data.csv").exists():
            raise FileNotFoundError(f"Unsupported or missing EuRoC sequence: {seq_dir}")
        resolved.append(seq_dir)
    return sorted(resolved)


def ns_to_sec(ns_text: str) -> float:
    return int(ns_text) / 1_000_000_000.0


def sequence_key(sequence_name: str) -> str:
    match = re.match(r"^(MH_\d\d|V1_\d\d|V2_\d\d)", sequence_name)
    return match.group(1) if match else sequence_name


def first_frame_for(sequence_name: str) -> int:
    return FIRST_FRAME_OVERRIDES.get(sequence_key(sequence_name), 0)


def read_csv_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append(row)
    return rows


def ensure_relative_symlink(link_path: Path, target_path: Path) -> None:
    resolved_target = target_path.resolve()
    if link_path.is_symlink() or link_path.exists():
        if (
            link_path.is_symlink()
            and Path(os.readlink(link_path)).is_absolute()
            and link_path.resolve() == resolved_target
        ):
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    # Use an absolute symlink because SVO accesses the prepared dataset through a
    # workspace symlink path, and relative links become fragile in that setup.
    link_path.symlink_to(resolved_target)


def write_images_txt(rows: list[list[str]], out_path: Path) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, (stamp_ns, filename) in enumerate(rows):
            handle.write(f"{idx} {ns_to_sec(stamp_ns):.9f} img/{filename}\n")
            count += 1
    return count


def write_imu_txt(rows: list[list[str]], out_path: Path) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            stamp_ns, wx, wy, wz, ax, ay, az = row
            handle.write(
                f"{idx} {ns_to_sec(stamp_ns):.9f} {wx} {wy} {wz} {ax} {ay} {az}\n"
            )
            count += 1
    return count


def write_groundtruth_txt(rows: list[list[str]], out_groundtruth: Path, out_stamped: Path) -> list[tuple[int, float]]:
    gt_index: list[tuple[int, float]] = []
    with out_groundtruth.open("w", encoding="utf-8") as gt_handle, out_stamped.open("w", encoding="utf-8") as stamped_handle:
        for idx, row in enumerate(rows):
            stamp_ns = row[0]
            px, py, pz = row[1:4]
            qw, qx, qy, qz = row[4:8]
            time_sec = ns_to_sec(stamp_ns)
            gt_handle.write(f"{idx} {time_sec:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")
            stamped_handle.write(f"{time_sec:.9f} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")
            gt_index.append((idx, time_sec))
    return gt_index


def write_groundtruth_matches(image_rows: list[list[str]], gt_index: list[tuple[int, float]], out_path: Path) -> None:
    gt_ptr = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for img_idx, (stamp_ns, _) in enumerate(image_rows):
            img_time = ns_to_sec(stamp_ns)
            while gt_ptr + 1 < len(gt_index) and abs(gt_index[gt_ptr + 1][1] - img_time) <= abs(gt_index[gt_ptr][1] - img_time):
                gt_ptr += 1
            handle.write(f"{img_idx} {gt_index[gt_ptr][0]}\n")


def prepare_sequence(sequence_dir: Path) -> dict[str, Any]:
    sequence_name = sequence_dir.name
    mav0 = sequence_dir / "mav0"
    cam_csv = mav0 / "cam0" / "data.csv"
    imu_csv = mav0 / "imu0" / "data.csv"
    gt_csv = mav0 / "state_groundtruth_estimate0" / "data.csv"

    image_rows = read_csv_rows(cam_csv)
    imu_rows = read_csv_rows(imu_csv)
    gt_rows = read_csv_rows(gt_csv)

    out_dir = SVO_DATA_ROOT / sequence_name
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir = mav0 / "cam0" / "data"
    ensure_relative_symlink(data_dir / "img", image_dir)
    shutil.copyfile(CALIB_SOURCE, out_dir / "calib.yaml")

    image_count = write_images_txt(image_rows, data_dir / "images.txt")
    imu_count = write_imu_txt(imu_rows, data_dir / "imu.txt")
    gt_index = write_groundtruth_txt(gt_rows, data_dir / "groundtruth.txt", data_dir / "stamped_groundtruth.txt")
    write_groundtruth_matches(image_rows, gt_index, data_dir / "groundtruth_matches.txt")

    dataset_meta = {
        "dataset_name": sequence_name,
        "dataset_first_frame": first_frame_for(sequence_name),
        "dataset_last_frame": image_count,
    }
    dump_yaml(out_dir / "dataset.yaml", dataset_meta)

    if image_rows:
        sample_image = data_dir / "img" / image_rows[0][1]
        if not sample_image.is_file():
            raise FileNotFoundError(
                f"Prepared SVO image link is invalid for {sequence_name}: {sample_image}"
            )

    return {
        "sequence_name": sequence_name,
        "dataset_name": f"mono3d/euroc/mono/{sequence_name}",
        "dataset_directory": str(out_dir),
        "dataset_first_frame": dataset_meta["dataset_first_frame"],
        "dataset_last_frame": dataset_meta["dataset_last_frame"],
        "counts": {
            "images": image_count,
            "imu": imu_count,
            "groundtruth": len(gt_index),
        },
    }


def make_experiment(base: dict[str, Any], prepared: list[dict[str, Any]], use_imu: bool) -> dict[str, Any]:
    settings = dict(base["settings"])
    settings["dataset_is_stereo"] = False
    settings["pipeline_is_stereo"] = False
    settings["trace_statistics"] = use_imu
    settings["calib_name"] = "calib.yaml"
    settings["runlc"] = False
    settings["use_imu"] = use_imu
    settings["trace_only_keyframes"] = False
    settings["dataset_first_frame"] = 0

    if use_imu:
        settings["use_ceres_backend"] = True
        settings["poseoptim_prior_lambda"] = 0.0
        settings["img_align_prior_lambda_rot"] = 0.5
        settings["img_align_prior_lambda_trans"] = 0.0
    else:
        settings["use_ceres_backend"] = False
        settings["poseoptim_prior_lambda"] = 0.0
        settings["img_align_prior_lambda_rot"] = 0.0
        settings["img_align_prior_lambda_trans"] = 0.0

    experiment_name = MONO_IMU_EXP_NAME if use_imu else MONO_EXP_NAME
    trace_root = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks" / experiment_name

    return {
        "experiment_label": experiment_name,
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
    sequences = selected_sequences(args.dataset_root, args.sequence)
    prepared = [prepare_sequence(seq_dir) for seq_dir in sequences]

    payload: dict[str, Any] = {"prepared_sequences": prepared}
    if args.write_configs:
        base_exp = load_yaml(BASE_EXP_SOURCE)
        mono_exp = make_experiment(base_exp, prepared, use_imu=False)
        mono_imu_exp = make_experiment(base_exp, prepared, use_imu=True)
        mono_path = SVO_EXP_ROOT / f"{MONO_EXP_NAME}.yaml"
        mono_imu_path = SVO_EXP_ROOT / f"{MONO_IMU_EXP_NAME}.yaml"
        dump_yaml(mono_path, mono_exp)
        dump_yaml(mono_imu_path, mono_imu_exp)
        payload["generated_experiments"] = [str(mono_path), str(mono_imu_path)]

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
