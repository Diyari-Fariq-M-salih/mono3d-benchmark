#!/usr/bin/env python3
"""Prepare full-resolution TUM RGB-D sequences for SVO monocular benchmarking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUM_ROOT = REPO_ROOT / "datasets" / "tum_rgbd"
SVO_BENCH_ROOT = REPO_ROOT / "methods" / "svo" / "upstream" / "svo_benchmarking"
SVO_DATA_ROOT = SVO_BENCH_ROOT / "data" / "mono3d" / "tum" / "mono"
SVO_EXP_ROOT = SVO_BENCH_ROOT / "experiments"
BASE_EXP_SOURCE = SVO_EXP_ROOT / "exp_euroc_nolc.yaml"

MONO_EXP_NAME = "mono3d_tum_mono"

SVO_TUM_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {
        "experiment_name": MONO_EXP_NAME,
        "description": "Current monocular baseline with official RGB intrinsics.",
        "settings": {},
    },
    "loop": {
        "experiment_name": "mono3d_tum_mono_loop",
        "description": "Enable loop closure without backend refinement.",
        "settings": {
            "runlc": True,
            "trace_statistics": True,
        },
    },
    "conservative": {
        "experiment_name": "mono3d_tum_mono_conservative",
        "description": "Tighter monocular initialization and keyframe gating for noisier sequences.",
        "settings": {
            "quality_min_fts": 55,
            "init_min_features": 80,
            "init_min_disparity": 40,
            "kfselect_min_dist_metric": 0.15,
            "kfselect_min_angle": 25,
            "kfselect_min_disparity": 45,
            "poseoptim_thresh": 1.5,
            "outlier_rejection_px_threshold": 1.5,
            "max_unconverged_seeds_ratio": 0.15,
            "trace_statistics": True,
        },
    },
    "conservative_loop": {
        "experiment_name": "mono3d_tum_mono_conservative_loop",
        "description": "Conservative monocular tracking with loop closure.",
        "settings": {
            "quality_min_fts": 55,
            "init_min_features": 80,
            "init_min_disparity": 40,
            "kfselect_min_dist_metric": 0.15,
            "kfselect_min_angle": 25,
            "kfselect_min_disparity": 45,
            "poseoptim_thresh": 1.5,
            "outlier_rejection_px_threshold": 1.5,
            "max_unconverged_seeds_ratio": 0.15,
            "runlc": True,
            "trace_statistics": True,
        },
    },
}

TUM_CALIBRATION_PROFILES = {
    "ros-default": {
        "freiburg1": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "dist": [0.0, 0.0, 0.0, 0.0]},
        "freiburg2": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "dist": [0.0, 0.0, 0.0, 0.0]},
        "freiburg3": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "dist": [0.0, 0.0, 0.0, 0.0]},
    },
    "official-rgb": {
        "freiburg1": {
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
            "dist": [0.2624, -0.9531, -0.0054, 0.0026],
        },
        "freiburg2": {
            "fx": 520.9,
            "fy": 521.0,
            "cx": 325.1,
            "cy": 249.7,
            "dist": [0.2312, -0.7849, -0.0033, -0.0001],
        },
        "freiburg3": {
            "fx": 535.4,
            "fy": 539.2,
            "cx": 320.1,
            "cy": 247.6,
            "dist": [0.0, 0.0, 0.0, 0.0],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full TUM RGB-D sequences for SVO mono benchmarking."
    )
    parser.add_argument("--tum-root", type=Path, default=DEFAULT_TUM_ROOT)
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Sequence stem, e.g. freiburg1_desk",
    )
    parser.add_argument(
        "--write-configs",
        action="store_true",
        help="Also generate the SVO experiment YAML for the selected datasets.",
    )
    parser.add_argument(
        "--calib-mode",
        choices=sorted(TUM_CALIBRATION_PROFILES),
        default="official-rgb",
        help="Which TUM calibration profile to write into calib.yaml.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(SVO_TUM_VARIANTS),
        default=[],
        help="Named SVO mono variant to generate. May be passed multiple times.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate all recommended non-IMU TUM SVO variants.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def discover_sequences(tum_root: Path, filters: list[str]) -> list[Path]:
    wanted = set(filters)
    selected: list[Path] = []
    for path in sorted(tum_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("rgbd_dataset_"):
            continue
        stem = path.name.replace("rgbd_dataset_", "", 1)
        if wanted and stem not in wanted:
            continue
        if not (path / "rgb.txt").exists():
            continue
        selected.append(path)
    if not selected:
        raise FileNotFoundError("No TUM sequences matched the requested filters.")
    return selected


def infer_intrinsics(sequence_name: str, calib_mode: str) -> dict[str, float | list[float]]:
    profile = TUM_CALIBRATION_PROFILES[calib_mode]
    for prefix, intrinsics in profile.items():
        if sequence_name.startswith(prefix):
            return intrinsics
    raise KeyError(f"No intrinsics configured for TUM sequence: {sequence_name}")


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


def load_tum_groundtruth(path: Path) -> list[tuple[float, list[float]]]:
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
    with out_groundtruth.open("w", encoding="utf-8") as gt_handle, out_stamped.open(
        "w", encoding="utf-8"
    ) as stamped_handle:
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


def write_groundtruth_matches(
    image_entries: list[tuple[float, str]], gt_index: list[tuple[int, float]], out_path: Path
) -> None:
    gt_ptr = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for img_idx, (img_time, _) in enumerate(image_entries):
            while gt_ptr + 1 < len(gt_index) and abs(gt_index[gt_ptr + 1][1] - img_time) <= abs(
                gt_index[gt_ptr][1] - img_time
            ):
                gt_ptr += 1
            handle.write(f"{img_idx} {gt_index[gt_ptr][0]}\n")


def image_size_from_png(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"Unsupported or invalid PNG header in {image_path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def make_calib(sequence_name: str, width: int, height: int, calib_mode: str) -> dict[str, Any]:
    intrinsics = infer_intrinsics(sequence_name, calib_mode)
    return {
        "label": f"TUM {sequence_name} ({calib_mode})",
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
                            intrinsics["fx"],
                            intrinsics["fy"],
                            intrinsics["cx"],
                            intrinsics["cy"],
                        ],
                    },
                    "distortion": {
                        "type": "radial-tangential",
                        "parameters": {"cols": 1, "rows": 4, "data": intrinsics["dist"]},
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


def prepare_sequence(sequence_dir: Path, calib_mode: str) -> dict[str, Any]:
    sequence_name = sequence_dir.name.replace("rgbd_dataset_", "", 1)
    rgb_entries = load_rgb_index(sequence_dir / "rgb.txt")
    gt_rows = load_tum_groundtruth(sequence_dir / "groundtruth.txt")

    first_image = sequence_dir / rgb_entries[0][1]
    width, height = image_size_from_png(first_image)

    out_dir = SVO_DATA_ROOT / sequence_name
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_relative_symlink(data_dir / "img", sequence_dir / "rgb")

    image_count = write_images_txt(rgb_entries, data_dir / "images.txt")
    gt_index = write_groundtruth_txt(
        gt_rows, data_dir / "groundtruth.txt", data_dir / "stamped_groundtruth.txt"
    )
    write_groundtruth_matches(rgb_entries, gt_index, data_dir / "groundtruth_matches.txt")
    dump_yaml(out_dir / "calib.yaml", make_calib(sequence_name, width, height, calib_mode))

    dataset_meta = {
        "dataset_name": sequence_name,
        "dataset_first_frame": 0,
        "dataset_last_frame": image_count,
    }
    dump_yaml(out_dir / "dataset.yaml", dataset_meta)

    return {
        "sequence_name": sequence_name,
        "dataset_name": f"mono3d/tum/mono/{sequence_name}",
        "dataset_directory": str(out_dir),
        "dataset_first_frame": 0,
        "dataset_last_frame": image_count,
        "counts": {"images": image_count, "groundtruth": len(gt_index)},
    }


def selected_variants(args: argparse.Namespace) -> list[str]:
    if args.all_variants:
        return list(SVO_TUM_VARIANTS)
    if args.variant:
        return args.variant
    return ["baseline"]


def make_experiment(
    base: dict[str, Any], prepared: list[dict[str, Any]], variant_name: str
) -> dict[str, Any]:
    variant = SVO_TUM_VARIANTS[variant_name]
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
    settings.update(variant["settings"])

    experiment_name = str(variant["experiment_name"])
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
    tum_root = resolve_path(args.tum_root)
    selected = discover_sequences(tum_root, args.sequence)
    variants = selected_variants(args)

    prepared_payload = [prepare_sequence(path, args.calib_mode) for path in selected]
    payload: dict[str, Any] = {
        "tum_root": str(tum_root),
        "calib_mode": args.calib_mode,
        "variants": variants,
        "prepared_datasets": prepared_payload,
    }

    if args.write_configs:
        base_exp = load_yaml(BASE_EXP_SOURCE)
        generated_paths: list[str] = []
        generated_meta: list[dict[str, str]] = []
        for variant_name in variants:
            mono_exp = make_experiment(base_exp, prepared_payload, variant_name)
            experiment_name = str(SVO_TUM_VARIANTS[variant_name]["experiment_name"])
            mono_path = SVO_EXP_ROOT / f"{experiment_name}.yaml"
            dump_yaml(mono_path, mono_exp)
            generated_paths.append(str(mono_path))
            generated_meta.append(
                {
                    "variant": variant_name,
                    "experiment_name": experiment_name,
                    "path": str(mono_path),
                    "description": str(SVO_TUM_VARIANTS[variant_name]["description"]),
                }
            )
        payload["generated_experiments"] = generated_paths
        payload["generated_experiment_meta"] = generated_meta

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
