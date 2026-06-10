#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = (
    REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "hall_da3_large_matrix" / "runs"
)
DEFAULT_REPORTS_ROOT = (
    REPO_ROOT / "reports" / "evaluation" / "da3_gpu_summary"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DA3 GPU telemetry across run folders and render comparison plots."
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root directory containing one subdirectory per DA3 run.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to a timestamped folder under reports/evaluation/da3_gpu_summary.",
    )
    return parser.parse_args()


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metadata(run_dir: Path) -> dict | None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_gpu_stats(csv_path: Path) -> dict[str, float | int] | None:
    if not csv_path.exists():
        return None

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            util = safe_float(raw_row.get("utilization.gpu"))
            mem_used = safe_float(raw_row.get("memory.used"))
            mem_total = safe_float(raw_row.get("memory.total"))
            temp = safe_float(raw_row.get("temperature.gpu"))
            power = safe_float(raw_row.get("power.draw"))
            elapsed = safe_float(raw_row.get("elapsed_s"))
            if None in (util, mem_used, mem_total, temp, power, elapsed):
                continue
            rows.append(
                {
                    "util": util,
                    "mem_used": mem_used,
                    "mem_total": mem_total,
                    "temp": temp,
                    "power": power,
                    "elapsed": elapsed,
                }
            )

    if not rows:
        return None

    utils = [row["util"] for row in rows]
    mem_used = [row["mem_used"] for row in rows]
    mem_total = [row["mem_total"] for row in rows]
    temps = [row["temp"] for row in rows]
    powers = [row["power"] for row in rows]
    elapsed = [row["elapsed"] for row in rows]
    duration = max(elapsed) - min(elapsed) if len(elapsed) > 1 else elapsed[0]

    return {
        "gpu_samples": len(rows),
        "duration_s": duration,
        "gpu_mean_util_percent": sum(utils) / len(utils),
        "gpu_peak_util_percent": max(utils),
        "gpu_mean_mem_used_mb": sum(mem_used) / len(mem_used),
        "gpu_peak_mem_used_mb": max(mem_used),
        "gpu_mean_mem_total_mb": sum(mem_total) / len(mem_total),
        "gpu_mean_mem_used_percent": (sum(mem_used) / len(mem_used)) / (sum(mem_total) / len(mem_total)) * 100.0,
        "gpu_peak_mem_used_percent": max(
            used / total * 100.0 for used, total in zip(mem_used, mem_total) if total > 0.0
        ),
        "gpu_mean_power_w": sum(powers) / len(powers),
        "gpu_peak_power_w": max(powers),
        "gpu_mean_temp_c": sum(temps) / len(temps),
        "gpu_peak_temp_c": max(temps),
    }


def collect_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        metadata = load_metadata(run_dir)
        if metadata is None:
            continue
        if metadata.get("status") != "completed" or metadata.get("returncode") != 0:
            continue

        gpu_csv = Path(metadata.get("gpu_csv", run_dir / "gpu_usage.csv"))
        if not gpu_csv.is_absolute():
            gpu_csv = (run_dir / gpu_csv).resolve()
        stats = load_gpu_stats(gpu_csv)
        if stats is None:
            continue

        frame_base = str(metadata.get("frame_base", "unknown"))
        fps = str(metadata.get("fps", ""))
        chunk_size = int(metadata.get("chunk_size", 0))
        overlap = int(metadata.get("overlap", 0))
        loop_enable = bool(metadata.get("loop_enable", False))

        row: dict[str, object] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "frame_base": frame_base,
            "fps": fps,
            "fps_value": safe_float(fps) if safe_float(fps) is not None else math.nan,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_overlap_label": f"{chunk_size}/{overlap}",
            "loop_enable": loop_enable,
        }
        row.update(stats)
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "run_name",
        "run_dir",
        "frame_base",
        "fps",
        "fps_value",
        "chunk_size",
        "overlap",
        "chunk_overlap_label",
        "loop_enable",
        "gpu_samples",
        "duration_s",
        "gpu_mean_util_percent",
        "gpu_peak_util_percent",
        "gpu_mean_mem_used_mb",
        "gpu_peak_mem_used_mb",
        "gpu_mean_mem_total_mb",
        "gpu_mean_mem_used_percent",
        "gpu_peak_mem_used_percent",
        "gpu_mean_power_w",
        "gpu_peak_power_w",
        "gpu_mean_temp_c",
        "gpu_peak_temp_c",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_scene_heatmaps(rows: list[dict[str, object]], output_dir: Path) -> list[str]:
    metrics = [
        ("gpu_mean_util_percent", "Mean GPU Utilization (%)", "YlOrRd", "scene_heatmaps_mean_util.png"),
        ("gpu_mean_power_w", "Mean GPU Power (W)", "YlOrBr", "scene_heatmaps_mean_power.png"),
        ("gpu_peak_mem_used_mb", "Peak GPU Memory (MB)", "GnBu", "scene_heatmaps_peak_mem.png"),
        ("duration_s", "Runtime (s)", "PuBu", "scene_heatmaps_runtime.png"),
    ]

    scenes = sorted({str(row["frame_base"]) for row in rows})
    fps_labels = sorted({str(row["fps"]) for row in rows}, key=lambda value: float(value))
    chunk_labels = sorted(
        {str(row["chunk_overlap_label"]) for row in rows},
        key=lambda value: tuple(int(part) for part in value.split("/")),
    )

    saved: list[str] = []
    for metric_key, title, cmap, filename in metrics:
        fig, axes = plt.subplots(
            len(scenes),
            1,
            figsize=(10, max(3.0 * len(scenes), 4.0)),
            constrained_layout=True,
        )
        if len(scenes) == 1:
            axes = [axes]

        all_values = [
            float(row[metric_key])
            for row in rows
            if row.get(metric_key) is not None and not math.isnan(float(row[metric_key]))
        ]
        vmin = min(all_values) if all_values else 0.0
        vmax = max(all_values) if all_values else 1.0
        image = None

        for ax, scene in zip(axes, scenes):
            matrix = []
            scene_rows = [row for row in rows if row["frame_base"] == scene]
            for fps in fps_labels:
                values = []
                for chunk_label in chunk_labels:
                    match = next(
                        (
                            row
                            for row in scene_rows
                            if str(row["fps"]) == fps and str(row["chunk_overlap_label"]) == chunk_label
                        ),
                        None,
                    )
                    values.append(float("nan") if match is None else float(match[metric_key]))
                matrix.append(values)

            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(scene.replace("_", " "))
            ax.set_xticks(range(len(chunk_labels)))
            ax.set_xticklabels(chunk_labels)
            ax.set_yticks(range(len(fps_labels)))
            ax.set_yticklabels(fps_labels)
            ax.set_xlabel("chunk/overlap")
            ax.set_ylabel("fps")

            for row_index, fps in enumerate(fps_labels):
                for col_index, chunk_label in enumerate(chunk_labels):
                    value = matrix[row_index][col_index]
                    if math.isnan(value):
                        continue
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

        if image is not None:
            fig.colorbar(image, ax=axes, shrink=0.9, label=title)
        fig.suptitle(f"DA3 GPU Summary by Scene: {title}", fontsize=15)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved.append(str(out_path))

    return saved


def plot_runtime_vs_memory(rows: list[dict[str, object]], output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["frame_base"])].append(row)

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    for index, (scene, scene_rows) in enumerate(sorted(grouped.items())):
        xs = [float(row["duration_s"]) for row in scene_rows]
        ys = [float(row["gpu_peak_mem_used_mb"]) for row in scene_rows]
        sizes = [40.0 + float(row["gpu_mean_util_percent"]) * 1.8 for row in scene_rows]
        labels = [f"{row['fps']} fps | {row['chunk_overlap_label']}" for row in scene_rows]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.75,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            label=scene.replace("_", " "),
        )
        for x, y, text in zip(xs, ys, labels):
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_title("DA3 Runtime vs Peak GPU Memory")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path = output_dir / "runtime_vs_peak_mem_scatter.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def plot_parameter_aggregates(rows: list[dict[str, object]], output_dir: Path) -> str:
    chunk_labels = sorted(
        {str(row["chunk_overlap_label"]) for row in rows},
        key=lambda value: tuple(int(part) for part in value.split("/")),
    )
    mean_power = []
    peak_mem = []
    mean_util = []
    for label in chunk_labels:
        matches = [row for row in rows if row["chunk_overlap_label"] == label]
        mean_power.append(sum(float(row["gpu_mean_power_w"]) for row in matches) / len(matches))
        peak_mem.append(sum(float(row["gpu_peak_mem_used_mb"]) for row in matches) / len(matches))
        mean_util.append(sum(float(row["gpu_mean_util_percent"]) for row in matches) / len(matches))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].bar(chunk_labels, mean_util, color="#c26a3d")
    axes[0].set_title("Mean GPU Utilization by Chunk/Overlap")
    axes[0].set_ylabel("Utilization (%)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(chunk_labels, mean_power, color="#2f7f72")
    axes[1].set_title("Mean GPU Power by Chunk/Overlap")
    axes[1].set_ylabel("Power (W)")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(chunk_labels, peak_mem, color="#3e78b2")
    axes[2].set_title("Average Peak GPU Memory by Chunk/Overlap")
    axes[2].set_ylabel("Peak memory (MB)")
    axes[2].set_xlabel("chunk/overlap")
    axes[2].grid(True, axis="y", alpha=0.25)

    out_path = output_dir / "chunk_overlap_aggregate_bars.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def write_manifest(
    rows: list[dict[str, object]], plot_paths: list[str], output_dir: Path, runs_root: Path
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "run_count": len(rows),
        "scene_count": len({str(row["frame_base"]) for row in rows}),
        "plots": plot_paths,
        "summary_csv": str(output_dir / "summary.csv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(output_arg: str) -> Path:
    if output_arg:
        output_dir = Path(output_arg)
        if not output_dir.is_absolute():
            output_dir = (REPO_ROOT / output_dir).resolve()
        return output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORTS_ROOT / timestamp


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (REPO_ROOT / runs_root).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(runs_root)
    if not rows:
        raise SystemExit("No completed DA3 runs with GPU telemetry were found.")

    rows.sort(
        key=lambda row: (
            str(row["frame_base"]),
            float(row["fps_value"]),
            int(row["chunk_size"]),
            int(row["overlap"]),
        )
    )
    write_summary_csv(rows, output_dir / "summary.csv")

    plot_paths: list[str] = []
    plot_paths.extend(plot_scene_heatmaps(rows, output_dir))
    plot_paths.append(plot_runtime_vs_memory(rows, output_dir))
    plot_paths.append(plot_parameter_aggregates(rows, output_dir))
    write_manifest(rows, plot_paths, output_dir, runs_root)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
