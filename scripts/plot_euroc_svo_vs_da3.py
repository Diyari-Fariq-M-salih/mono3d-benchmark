#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SVO_SUMMARY = REPO_ROOT / "reports" / "evaluation" / "svo_full11_euroc_mono_vs_mono_imu_20260601" / "summary.csv"
DEFAULT_DA3_RUNS_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "euroc"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EuRoC SVO mono / mono+imu results against DA3 trajectory evaluation."
    )
    parser.add_argument("--svo-summary", type=Path, default=DEFAULT_SVO_SUMMARY)
    parser.add_argument("--da3-runs-root", type=Path, default=DEFAULT_DA3_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Defaults to reports/evaluation/euroc_svo_vs_da3_<latest>.") 
    return parser.parse_args()


def load_svo_summary(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_sequence: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_sequence[row["sequence"]][row["mode"]] = row
    return by_sequence


def load_best_da3(runs_root: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for summary_path in sorted(runs_root.glob("*/**/eval_euroc/trajectory_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        seq = payload["sequence"]
        sim3 = payload.get("sim3", {}).get("rmse_m")
        if sim3 is None:
            continue
        current = best.get(seq)
        if current is None or float(sim3) < float(current["summary"]["sim3"]["rmse_m"]):
            best[seq] = {
                "summary": payload,
                "run_dir": str(summary_path.parents[1]),
                "run_name": summary_path.parents[1].name,
            }
    if not best:
        raise FileNotFoundError(f"No EuRoC DA3 evaluation summaries found under {runs_root}")
    return best


def value_or_blank(value):
    return "" if value is None else value


def latest_token(runs_root: Path) -> str:
    tokens = sorted(
        path.name
        for path in runs_root.glob("*/*")
        if path.is_dir() and path.name.endswith("_da3_streaming")
    )
    return tokens[-1] if tokens else "latest"


def plot_grouped_bars(sequences: list[str], mono: list[float], mono_imu: list[float], da3: list[float], out_path: Path) -> None:
    indices = list(range(len(sequences)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(11, len(sequences) * 1.2), 5.8))
    ax.bar([i - width for i in indices], mono, width=width, label="SVO mono", color="#b55d3d")
    ax.bar(indices, mono_imu, width=width, label="SVO mono+imu", color="#2f7f6f")
    ax.bar([i + width for i in indices], da3, width=width, label="DA3", color="#3b6cb7")
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel("Sim3 RMSE (m)")
    ax.set_title("EuRoC Trajectory Shape Error After Sim3 Alignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    svo_summary = load_svo_summary(args.svo_summary.resolve())
    da3_best = load_best_da3(args.da3_runs_root.resolve())

    common = sorted(set(svo_summary) & set(da3_best))
    if not common:
        raise RuntimeError("No overlapping EuRoC sequences found between SVO summary and DA3 runs")

    output_dir = args.output_dir.resolve() if args.output_dir else (DEFAULT_REPORT_ROOT / f"euroc_svo_vs_da3_{latest_token(args.da3_runs_root.resolve())}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    mono_vals = []
    mono_imu_vals = []
    da3_vals = []
    for sequence in common:
        mono = svo_summary[sequence].get("mono")
        mono_imu = svo_summary[sequence].get("mono-imu")
        da3 = da3_best[sequence]["summary"]
        mono_sim3 = float(mono["sim3_rmse_m"]) if mono else None
        mono_imu_sim3 = float(mono_imu["sim3_rmse_m"]) if mono_imu else None
        da3_sim3 = float(da3["sim3"]["rmse_m"])
        rows.append(
            {
                "sequence": sequence,
                "svo_mono_sim3_rmse_m": value_or_blank(mono_sim3),
                "svo_mono_imu_sim3_rmse_m": value_or_blank(mono_imu_sim3),
                "da3_sim3_rmse_m": da3_sim3,
                "da3_match_count": da3["match_count"],
                "da3_path_ratio_est_over_gt": value_or_blank(da3.get("path_ratio_est_over_gt")),
                "da3_run_name": da3_best[sequence]["run_name"],
                "da3_run_dir": da3_best[sequence]["run_dir"],
            }
        )
        mono_vals.append(mono_sim3)
        mono_imu_vals.append(mono_imu_sim3)
        da3_vals.append(da3_sim3)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_grouped_bars(common, mono_vals, mono_imu_vals, da3_vals, output_dir / "sim3_rmse.png")
    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"plot: {output_dir / 'sim3_rmse.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
