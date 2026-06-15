#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "7sense"
DEFAULT_PREPARED_ROOT = REPO_ROOT / "datasets" / "7sense_prepared"
DEFAULT_DERIVED_ROOT = REPO_ROOT / "datasets" / "7sense_prepared_derived"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "7sense"
RUN_DA3_SCRIPT = REPO_ROOT / "scripts" / "run_da3_streaming.sh"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "da3_streaming" / "weights"
DEFAULT_DA3_MODEL_DIR = Path(
    "/home/qcar/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DA3-Streaming on prepared 7sense RGB sequences.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append", default=[], help="Scene name, e.g. chess")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Specific sequence identifier, e.g. chess/seq-01 or redkitchen_seq-14",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--fps", nargs="+", default=["2", "3", "5"], help="Prepared FPS labels to run.")
    parser.add_argument(
        "--frame-stride",
        nargs="+",
        type=int,
        default=[1],
        help="Keep every Nth frame from the prepared folder.",
    )
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--loop-enable", dest="loop_enable", action="store_true", default=True)
    parser.add_argument("--no-loop-enable", dest="loop_enable", action="store_false")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--da3-model-dir", type=Path, default=DEFAULT_DA3_MODEL_DIR)
    parser.add_argument("--conda-env", default="da3stream")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    return parser.parse_args()


def split_file_for(scene_dir: Path, split: str) -> Path:
    if split == "train":
        return scene_dir / "TrainSplit.txt"
    if split == "test":
        return scene_dir / "TestSplit.txt"
    raise ValueError(f"unsupported split: {split}")


def split_entries(scene_dir: Path, split: str) -> list[str]:
    import re

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
    import re

    if "/" in value:
        scene, seq_name = value.split("/", 1)
        return scene, seq_name
    match = re.match(r"(.+)_seq-(\d+)$", value)
    if match:
        return match.group(1), f"seq-{int(match.group(2)):02d}"
    raise ValueError(f"Sequence filter must look like scene/seq-01 or scene_seq-01, got: {value}")


def selected_sequence_ids(dataset_root: Path, scenes: list[str], sequence_filters: list[str], split: str) -> list[str]:
    requested_scenes = set(scenes)
    selected: list[str] = []
    if sequence_filters:
        for item in sequence_filters:
            scene_name, seq_name = parse_sequence_filter(item)
            selected.append(f"{scene_name}_{seq_name}")
        return sorted(selected)

    for scene_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if requested_scenes and scene_dir.name not in requested_scenes:
            continue
        for seq_name in split_entries(scene_dir, split):
            selected.append(f"{scene_dir.name}_{seq_name}")
    if not selected:
        raise FileNotFoundError("No 7sense sequences matched the requested filters.")
    return selected


def timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_stride_view(source_dir: Path, derived_root: Path, stride: int) -> tuple[Path, int]:
    if stride <= 0:
        raise ValueError(f"frame stride must be positive, got {stride}")
    frame_paths = sorted(source_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No prepared frames found in {source_dir}")
    selected = frame_paths[::stride]
    if stride == 1:
        return source_dir, len(selected)

    derived_dir = derived_root / f"{source_dir.name}_stride{stride}"
    derived_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(derived_dir.glob("frame_*.png"))
    if len(existing) != len(selected):
        for old_path in existing:
            old_path.unlink()
        for index, frame_path in enumerate(selected, start=1):
            link_path = derived_dir / f"frame_{index:06d}.png"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(frame_path.resolve())
    return derived_dir, len(selected)


def build_config_text(weights_dir: Path, da3_model_dir: Path, chunk_size: int, overlap: int, loop_enable: bool) -> str:
    da3 = da3_model_dir / "model.safetensors"
    da3_config = da3_model_dir / "config.json"
    salad = weights_dir / "dino_salad.ckpt"
    return f"""Weights:
  DA3: '{da3}'
  DA3_CONFIG: '{da3_config}'
  SALAD: '{salad}'

Model:
  chunk_size: {chunk_size}
  overlap: {overlap}

  loop_chunk_size: 8
  loop_enable: {'True' if loop_enable else 'False'}
  useDBoW: False
  delete_temp_files: True

  align_lib: 'triton'
  align_method: 'sim3'
  scale_compute_method: 'auto'
  align_type: 'dense'

  ref_view_strategy: 'middle'
  ref_view_strategy_loop: 'middle'
  depth_threshold: 15

  save_depth_conf_result: True
  save_debug_info: False

  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 5000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
    sample_ratio: 0.15
    conf_threshold_coef: 0.75

Loop:
  SALAD:
    image_size: [224, 224]
    batch_size: 8
    similarity_threshold: 0.85
    top_k: 5
    use_nms: True
    nms_threshold: 35

  SIM3_Optimizer:
    lang_version: 'python'
    max_iterations: 30
    lambda_init: 1e-6
"""


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    derived_root = args.derived_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env["DA3_CONDA_ENV"] = args.conda_env
    child_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    run_token = timestamp_token()
    plan_rows = []

    for sequence_id in selected_sequence_ids(args.dataset_root.resolve(), args.scene, args.sequence, args.split):
        for fps in args.fps:
            prepared_dir = prepared_root / f"{sequence_id}_fps{fps.replace('.', 'p')}"
            if not prepared_dir.exists():
                raise FileNotFoundError(f"Missing prepared dir: {prepared_dir}")
            for stride in args.frame_stride:
                image_dir, frame_count = ensure_stride_view(prepared_dir, derived_root, stride)
                stride_suffix = "" if stride == 1 else f"_stride{stride}"
                run_dir = output_root / sequence_id / f"{run_token}_fps{fps.replace('.', 'p')}{stride_suffix}_da3_streaming"
                plan_rows.append(
                    {
                        "sequence": sequence_id,
                        "fps": fps,
                        "frame_stride": stride,
                        "frame_count": frame_count,
                        "prepared_dir": str(image_dir),
                        "run_dir": str(run_dir),
                    }
                )
                if not args.run:
                    continue
                if run_dir.exists() and any(run_dir.iterdir()) and not args.force_run:
                    print(f"[skip] existing run dir {run_dir}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path = run_dir / "config.yaml"
                config_path.write_text(
                    build_config_text(
                        args.weights_dir.resolve(),
                        args.da3_model_dir.resolve(),
                        args.chunk_size,
                        args.overlap,
                        args.loop_enable,
                    ),
                    encoding="utf-8",
                )
                command = [
                    "bash",
                    str(RUN_DA3_SCRIPT),
                    "--image-dir",
                    str(image_dir),
                    "--output-dir",
                    str(run_dir),
                    "--config",
                    str(config_path),
                ]
                print(f"[run] {sequence_id} fps{fps} stride{stride} | frames={frame_count}")
                subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_env)

    plan_path = output_root / f"7sense_da3_plan_{run_token}.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "fps", "frame_stride", "frame_count", "prepared_dir", "run_dir"],
        )
        writer.writeheader()
        writer.writerows(plan_rows)
    print(f"[plan] {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
