#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EUROC_ROOT = REPO_ROOT / "datasets" / "euroc"
DEFAULT_PREPARED_ROOT = REPO_ROOT / "datasets" / "euroc_prepared"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "euroc"
RUN_DA3_SCRIPT = REPO_ROOT / "scripts" / "run_da3_streaming.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_euroc_da3.py"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "da3_streaming" / "weights"
DEFAULT_DA3_MODEL_DIR = Path(
    "/home/qcar/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and evaluate DA3-Streaming on EuRoC cam0 sequences.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EUROC_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence", action="append", default=[], help="Specific EuRoC sequence(s).")
    parser.add_argument("--config", type=Path, help="Optional DA3 config override.")
    parser.add_argument("--fps", type=float, default=5.0, help="Prepared EuRoC frame rate. Defaults to 5.")
    parser.add_argument("--chunk-size", type=int, default=30, help="DA3 chunk size. Defaults to 30.")
    parser.add_argument("--overlap", type=int, default=15, help="DA3 chunk overlap. Defaults to 15.")
    parser.add_argument("--loop-enable", dest="loop_enable", action="store_true", default=True, help="Enable DA3 loop closure.")
    parser.add_argument("--no-loop-enable", dest="loop_enable", action="store_false", help="Disable DA3 loop closure.")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR, help="Directory containing dino_salad.ckpt.")
    parser.add_argument(
        "--da3-model-dir",
        type=Path,
        default=DEFAULT_DA3_MODEL_DIR,
        help="Directory containing matching DA3-LARGE model.safetensors and config.json.",
    )
    parser.add_argument(
        "--conda-env",
        default="da3stream",
        help="Conda environment used by scripts/run_da3_streaming.sh. Defaults to da3stream.",
    )
    parser.add_argument("--run", action="store_true", help="Actually run DA3 before evaluation.")
    parser.add_argument("--force-run", action="store_true", help="Rerun even if the output folder already exists.")
    return parser.parse_args()


def selected_sequences(dataset_root: Path, names: list[str]) -> list[Path]:
    if not names:
        return sorted(path for path in dataset_root.iterdir() if (path / "mav0" / "cam0" / "data").exists())
    resolved = []
    for name in names:
        seq_dir = dataset_root / name
        if not (seq_dir / "mav0" / "cam0" / "data").exists():
            raise FileNotFoundError(f"Missing EuRoC sequence: {seq_dir}")
        resolved.append(seq_dir)
    return sorted(resolved)


def timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fps_label(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else str(fps).replace(".", "p")


def ensure_absolute_symlink(link_path: Path, target_path: Path) -> None:
    resolved_target = target_path.resolve()
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == resolved_target:
            return
        if link_path.is_dir() and not link_path.is_symlink():
            for child in link_path.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    raise RuntimeError(f"Refusing to remove nested directory from prepared frame dir: {child}")
            link_path.rmdir()
        else:
            link_path.unlink()
    link_path.symlink_to(resolved_target)


def prepare_sequence_frames(sequence_dir: Path, prepared_root: Path, fps: float) -> Path:
    image_dir = sequence_dir / "mav0" / "cam0" / "data"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing EuRoC cam0 directory: {image_dir}")

    prepared_dir = prepared_root / f"{sequence_dir.name}_fps{fps_label(fps)}"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(image_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {image_dir}")

    selected: list[Path] = []
    min_delta = 1.0 / fps
    last_ts = None
    for frame_path in frame_paths:
        timestamp = int(frame_path.stem) / 1_000_000_000.0
        if last_ts is None or (timestamp - last_ts) >= (min_delta - 1e-9):
            selected.append(frame_path)
            last_ts = timestamp

    if not selected:
        raise RuntimeError(f"No frames selected for {sequence_dir.name} at fps={fps}")

    existing_links = list(prepared_dir.glob("*.png"))
    if len(existing_links) != len(selected):
        for old in existing_links:
            old.unlink()
        for src in selected:
            ensure_absolute_symlink(prepared_dir / src.name, src)

    return prepared_dir


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


def write_run_config(run_dir: Path, weights_dir: Path, da3_model_dir: Path, chunk_size: int, overlap: int, loop_enable: bool) -> Path:
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        build_config_text(weights_dir, da3_model_dir, chunk_size, overlap, loop_enable),
        encoding="utf-8",
    )
    return config_path


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    weights_dir = args.weights_dir.resolve()
    da3_model_dir = args.da3_model_dir.resolve()
    child_env = dict(os.environ)
    child_env["DA3_CONDA_ENV"] = args.conda_env
    child_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

    plan_rows = []
    run_token = timestamp_token()
    for seq_dir in selected_sequences(dataset_root, args.sequence):
        sequence = seq_dir.name
        image_dir = prepare_sequence_frames(seq_dir, prepared_root, args.fps)
        run_dir = output_root / sequence / f"{run_token}_fps{fps_label(args.fps)}_da3_streaming"
        plan_rows.append(
            {
                "sequence": sequence,
                "image_dir": str(image_dir),
                "run_dir": str(run_dir),
                "fps": args.fps,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "loop_enable": args.loop_enable,
            }
        )

        if args.run:
            if run_dir.exists() and not args.force_run:
                print(f"[skip] existing run dir: {run_dir}")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path = args.config.resolve() if args.config else write_run_config(
                    run_dir,
                    weights_dir,
                    da3_model_dir,
                    args.chunk_size,
                    args.overlap,
                    args.loop_enable,
                )
                command = ["bash", str(RUN_DA3_SCRIPT), "--image-dir", str(image_dir), "--output-dir", str(run_dir)]
                command.extend(["--config", str(config_path)])
                print(f"[run] {sequence}")
                subprocess.run(command, check=True, cwd=REPO_ROOT, env=child_env)

        if (run_dir / "camera_poses.txt").exists():
            eval_command = [
                "python3",
                str(EVAL_SCRIPT),
                "--run-dir",
                str(run_dir),
                "--sequence-dir",
                str(seq_dir),
                "--prepared-dir",
                str(image_dir),
            ]
            print(f"[eval] {sequence}")
            subprocess.run(eval_command, check=True, cwd=REPO_ROOT, env=child_env)
        else:
            print(f"[warn] no camera_poses.txt for {sequence}, skipped evaluation")

    plan_path = output_root / f"plan_{run_token}.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "image_dir", "run_dir", "fps", "chunk_size", "overlap", "loop_enable"])
        writer.writeheader()
        writer.writerows(plan_rows)
    print(f"plan: {plan_path}")
    print("next: python3 scripts/plot_euroc_svo_vs_da3.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
