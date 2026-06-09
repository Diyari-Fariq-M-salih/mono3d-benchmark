#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SANDBOX_ROOT = Path("/home/qcar/Documents/Diyari_M_salih_2026/da3_official_sandbox")
DEFAULT_LOCAL_STREAMING_ROOT = (
    REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "da3_streaming"
)
DEFAULT_GPU_MONITOR = (
    DEFAULT_SANDBOX_ROOT / "Depth-Anything-3" / "da3_streaming" / "monitor_gpu.py"
)
DEFAULT_FRAMES_ROOT = DEFAULT_SANDBOX_ROOT / "hall_da3_large_matrix" / "frames"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "hall_da3_large_matrix"
)
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "methods" / "depth_anything_3" / "upstream" / "da3_streaming" / "weights"
DEFAULT_DA3_MODEL_DIR = Path(
    "/home/qcar/.cache/huggingface/hub/models--depth-anything--DA3-LARGE/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
)


@dataclass(frozen=True)
class RunSpec:
    source_name: str
    fps: str
    chunk_size: int
    overlap: int
    loop_enable: bool

    @property
    def base_name(self) -> str:
        stem = Path(self.source_name).stem
        loop_tag = "loop" if self.loop_enable else "noloop"
        return f"{stem}__fps{self.fps}__chunk{self.chunk_size}__overlap{self.overlap}__{loop_tag}"

    @property
    def frame_dir_name(self) -> str:
        stem = Path(self.source_name).stem
        fps_label = str(self.fps).replace(".", "p")
        return f"{stem}_fps{fps_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the saved hall DA3-LARGE matrix settings from the official sandbox."
    )
    parser.add_argument(
        "--frame-bases",
        nargs="+",
        default=["color_hall"],
        help="Existing frame-set bases, e.g. color_hall or grey_hall.",
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        default=["2", "3", "5"],
        help="FPS labels that match the saved frame directories.",
    )
    parser.add_argument(
        "--chunk-overlap",
        nargs="+",
        default=["10:5", "20:10", "30:15"],
        help="Chunk/overlap pairs like 10:5 20:10 30:15.",
    )
    parser.add_argument(
        "--conda-env",
        default="da3stream",
        help="Conda environment name used for DA3-Streaming.",
    )
    parser.add_argument(
        "--frames-root",
        default=str(DEFAULT_FRAMES_ROOT),
        help="Root folder that already contains extracted hall frame directories.",
    )
    parser.add_argument(
        "--streaming-root",
        default=str(DEFAULT_LOCAL_STREAMING_ROOT),
        help="DA3-Streaming checkout root to execute.",
    )
    parser.add_argument(
        "--gpu-monitor",
        default=str(DEFAULT_GPU_MONITOR),
        help="Path to monitor_gpu.py used to wrap each run.",
    )
    parser.add_argument(
        "--no-gpu-monitor",
        action="store_true",
        help="Run DA3-Streaming directly without wrapping monitor_gpu.py.",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Streaming weight directory containing dino_salad.ckpt.",
    )
    parser.add_argument(
        "--da3-model-dir",
        default=str(DEFAULT_DA3_MODEL_DIR),
        help="Directory containing the matching DA3-LARGE model.safetensors and config.json.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder where plans and runs will be written.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Write the run plan CSV only.")
    parser.add_argument("--run", action="store_true", help="Run the planned experiments.")
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run experiments even if metadata.json already exists.",
    )
    parser.add_argument(
        "--loop-enable",
        action="store_true",
        help="Generate configs with loop closure enabled.",
    )
    return parser.parse_args()


def parse_chunk_overlap(values: list[str]) -> list[tuple[int, int]]:
    pairs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid chunk/overlap pair: {value}")
        chunk_str, overlap_str = value.split(":", 1)
        chunk_size = int(chunk_str)
        overlap = int(overlap_str)
        if overlap >= chunk_size:
            raise ValueError(f"Overlap must be smaller than chunk size: {value}")
        pairs.append((chunk_size, overlap))
    return pairs


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    pairs = parse_chunk_overlap(args.chunk_overlap)
    specs = []
    for source_name in args.frame_bases:
        for fps in args.fps:
            for chunk_size, overlap in pairs:
                specs.append(
                    RunSpec(
                        source_name=source_name,
                        fps=str(fps),
                        chunk_size=chunk_size,
                        overlap=overlap,
                        loop_enable=args.loop_enable,
                    )
                )
    return specs


def ensure_paths(output_root: Path) -> tuple[Path, Path]:
    runs_root = output_root / "runs"
    plans_root = output_root / "plans"
    runs_root.mkdir(parents=True, exist_ok=True)
    plans_root.mkdir(parents=True, exist_ok=True)
    return runs_root, plans_root


def write_plan(specs: list[RunSpec], plans_root: Path, frames_root: Path, runs_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plan_path = plans_root / f"hall_da3_large_plan_{timestamp}.csv"
    with plan_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_base",
                "fps",
                "chunk_size",
                "overlap",
                "loop_enable",
                "frame_dir",
                "run_dir",
            ],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "frame_base": spec.source_name,
                    "fps": spec.fps,
                    "chunk_size": spec.chunk_size,
                    "overlap": spec.overlap,
                    "loop_enable": spec.loop_enable,
                    "frame_dir": str(frames_root / spec.frame_dir_name),
                    "run_dir": str(runs_root / spec.base_name),
                }
            )
    return plan_path


def count_frames(frame_dir: Path) -> int:
    return len(list(frame_dir.glob("*.png")))


def resolve_existing_frame_dir(frames_root: Path, frame_base: str, fps: str) -> Path:
    frame_dir = frames_root / f"{Path(frame_base).stem}_fps{str(fps).replace('.', 'p')}"
    if not frame_dir.exists():
        raise FileNotFoundError(f"Expected existing frame dir not found: {frame_dir}")
    existing_count = count_frames(frame_dir)
    if existing_count == 0:
        raise FileNotFoundError(f"Existing frame dir has no PNG frames: {frame_dir}")
    return frame_dir


def config_text(
    weights_dir: Path, da3_model_dir: Path, chunk_size: int, overlap: int, loop_enable: bool
) -> str:
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


def write_run_config(
    run_dir: Path,
    weights_dir: Path,
    da3_model_dir: Path,
    chunk_size: int,
    overlap: int,
    loop_enable: bool,
) -> Path:
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        config_text(weights_dir, da3_model_dir, chunk_size, overlap, loop_enable),
        encoding="utf-8",
    )
    return config_path


def write_metadata(
    run_dir: Path,
    spec: RunSpec,
    frame_dir: Path,
    config_path: Path,
    frame_count: int,
    streaming_root: Path,
    extra: dict,
) -> None:
    payload = {
        "frame_base": spec.source_name,
        "fps": spec.fps,
        "chunk_size": spec.chunk_size,
        "overlap": spec.overlap,
        "loop_enable": spec.loop_enable,
        "frame_dir": str(frame_dir),
        "frame_count": frame_count,
        "config_path": str(config_path),
        "streaming_root": str(streaming_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    (run_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_existing_status(metadata_path: Path) -> tuple[str | None, int | None]:
    if not metadata_path.exists():
        return None, None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return payload.get("status"), payload.get("returncode")


def run_spec(
    spec: RunSpec,
    conda_env: str,
    frames_root: Path,
    runs_root: Path,
    streaming_root: Path,
    gpu_monitor: Path,
    weights_dir: Path,
    da3_model_dir: Path,
    force_run: bool,
    no_gpu_monitor: bool,
) -> None:
    frame_dir = resolve_existing_frame_dir(frames_root, spec.source_name, spec.fps)

    run_dir = runs_root / spec.base_name
    metadata_path = run_dir / "metadata.json"
    existing_status, existing_returncode = read_existing_status(metadata_path)
    if metadata_path.exists() and not force_run:
        if existing_status == "completed" and existing_returncode == 0:
            print(f"[run] Skipping successful run {run_dir}")
            return
        print(f"[run] Re-running prior non-successful run {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_run_config(
        run_dir,
        weights_dir,
        da3_model_dir,
        spec.chunk_size,
        spec.overlap,
        spec.loop_enable,
    )
    frame_count = count_frames(frame_dir)

    da3_command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "da3_streaming.py",
        "--image_dir",
        str(frame_dir),
        "--config",
        str(config_path),
        "--output_dir",
        str(run_dir),
    ]
    wrapped_command = da3_command
    if not no_gpu_monitor:
        wrapped_command = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(gpu_monitor),
            "--output-csv",
            str(run_dir / "gpu_usage.csv"),
            "--plot",
            str(run_dir / "gpu_usage.png"),
            "--",
            *da3_command,
        ]

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    write_metadata(
        run_dir,
        spec,
        frame_dir,
        config_path,
        frame_count,
        streaming_root,
        {
            "status": "running",
            "command": wrapped_command,
        },
    )
    print(f"[run] {spec.base_name}")
    print(f"[run] Frames: {frame_count} | Config: {config_path}")

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["PYTHONPATH"] = str(streaming_root.parent / "src")

    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.run(
            wrapped_command,
            cwd=streaming_root,
            env=env,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )

    write_metadata(
        run_dir,
        spec,
        frame_dir,
        config_path,
        frame_count,
        streaming_root,
        {
            "status": "completed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "command": wrapped_command,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "gpu_csv": str(run_dir / "gpu_usage.csv"),
            "gpu_plot": str(run_dir / "gpu_usage.png"),
        },
    )
    if process.returncode != 0:
        raise SystemExit(f"Run failed for {spec.base_name}. See {stderr_path}")


def main() -> int:
    args = parse_args()
    if not args.plan_only and not args.run:
        print("Choose one of --plan-only or --run", file=sys.stderr)
        return 2

    frames_root = Path(args.frames_root)
    if not frames_root.is_absolute():
        frames_root = (REPO_ROOT / frames_root).resolve()
    streaming_root = Path(args.streaming_root)
    if not streaming_root.is_absolute():
        streaming_root = (REPO_ROOT / streaming_root).resolve()
    gpu_monitor = Path(args.gpu_monitor)
    if not gpu_monitor.is_absolute():
        gpu_monitor = (REPO_ROOT / gpu_monitor).resolve()
    weights_dir = Path(args.weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = (REPO_ROOT / weights_dir).resolve()
    da3_model_dir = Path(args.da3_model_dir)
    if not da3_model_dir.is_absolute():
        da3_model_dir = (REPO_ROOT / da3_model_dir).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    runs_root, plans_root = ensure_paths(output_root)
    specs = build_specs(args)
    plan_path = write_plan(specs, plans_root, frames_root, runs_root)
    print(f"[plan] {plan_path}")

    if args.plan_only:
        return 0

    if not streaming_root.exists():
        raise FileNotFoundError(f"Streaming root not found: {streaming_root}")
    if not args.no_gpu_monitor and not gpu_monitor.exists():
        raise FileNotFoundError(f"GPU monitor not found: {gpu_monitor}")
    for name in ["dino_salad.ckpt"]:
        candidate = weights_dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing weight file: {candidate}")
    for name in ["model.safetensors", "config.json"]:
        candidate = da3_model_dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing DA3-LARGE model file: {candidate}")

    for spec in specs:
        run_spec(
            spec=spec,
            conda_env=args.conda_env,
            frames_root=frames_root,
            runs_root=runs_root,
            streaming_root=streaming_root,
            gpu_monitor=gpu_monitor,
            weights_dir=weights_dir,
            da3_model_dir=da3_model_dir,
            force_run=args.force_run,
            no_gpu_monitor=args.no_gpu_monitor,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
