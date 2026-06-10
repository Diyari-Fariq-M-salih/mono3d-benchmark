#!/usr/bin/env python3
"""Aggregate TUM SVO-variant vs DA3 comparison reports for report writing."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiple TUM SVO-vs-DA3 comparison outputs."
    )
    parser.add_argument(
        "--report-dir",
        action="append",
        type=Path,
        default=[],
        help="Specific tum_svo_vs_da3_* report directory to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--prefix",
        default="tum_svo_vs_da3_",
        help="Directory prefix to scan under reports/evaluation when --report-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the aggregate CSV and plots. Defaults to reports/evaluation/tum_svo_variants_summary_<timestamp>.",
    )
    return parser.parse_args()


def find_report_dirs(args: argparse.Namespace) -> list[Path]:
    if args.report_dir:
        return [path.resolve() for path in args.report_dir]
    return sorted(
        path
        for path in REPORT_ROOT.iterdir()
        if path.is_dir() and path.name.startswith(args.prefix) and (path / "summary.csv").exists()
    )


def infer_variant_name(report_dir: Path) -> str:
    name = report_dir.name
    if not name.startswith("tum_svo_vs_da3_"):
        return name
    remainder = name[len("tum_svo_vs_da3_") :]
    if "_202" in remainder:
        return remainder.split("_202", 1)[0].rstrip("_")
    return remainder


def load_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def compute_variant_summary(report_dir: Path) -> dict[str, object]:
    rows = load_rows(report_dir / "summary.csv")
    comparable = [row for row in rows if row.get("svo_status") == "ok" and row.get("svo_sim3_rmse_m", "").strip()]
    svo_values = [to_float(row["svo_sim3_rmse_m"]) for row in comparable]
    da3_values = [to_float(row["da3_sim3_rmse_m"]) for row in comparable]
    svo_values = [value for value in svo_values if value is not None]
    da3_values = [value for value in da3_values if value is not None]
    svo_wins = sum(1 for row in rows if row.get("winner_by_sim3") == "SVO")
    da3_wins = sum(1 for row in rows if row.get("winner_by_sim3") == "DA3")
    return {
        "variant": infer_variant_name(report_dir),
        "report_dir": str(report_dir),
        "sequence_count": len(rows),
        "comparable_sequence_count": len(comparable),
        "svo_mean_sim3_rmse_m": (sum(svo_values) / len(svo_values)) if svo_values else None,
        "da3_mean_sim3_rmse_m": (sum(da3_values) / len(da3_values)) if da3_values else None,
        "mean_sim3_ratio_svo_over_da3": ((sum(svo_values) / len(svo_values)) / (sum(da3_values) / len(da3_values)))
        if svo_values and da3_values and sum(da3_values) > 0
        else None,
        "svo_win_count": svo_wins,
        "da3_win_count": da3_wins,
    }


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "report_dir",
        "sequence_count",
        "comparable_sequence_count",
        "svo_mean_sim3_rmse_m",
        "da3_mean_sim3_rmse_m",
        "mean_sim3_ratio_svo_over_da3",
        "svo_win_count",
        "da3_win_count",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    report_dirs = find_report_dirs(args)
    if not report_dirs:
        raise FileNotFoundError("No TUM SVO-vs-DA3 comparison reports found.")

    summaries = [compute_variant_summary(path) for path in report_dirs]
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        import datetime as dt

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPORT_ROOT / f"tum_svo_variants_summary_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(summaries, output_dir / "summary.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps({"output_dir": str(output_dir), "variants": summaries}, indent=2))
        return 0

    variants = [str(item["variant"]) for item in summaries]
    svo_means = [item["svo_mean_sim3_rmse_m"] for item in summaries]
    da3_means = [item["da3_mean_sim3_rmse_m"] for item in summaries]

    filtered = [
        (variant, float(svo), float(da3))
        for variant, svo, da3 in zip(variants, svo_means, da3_means)
        if svo is not None and da3 is not None
    ]
    if filtered:
        indices = list(range(len(filtered)))
        width = 0.38
        fig_width = max(9, len(filtered) * 1.3)
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))
        ax.bar([i - width / 2 for i in indices], [row[1] for row in filtered], width=width, label="SVO", color="#b55d3d")
        ax.bar([i + width / 2 for i in indices], [row[2] for row in filtered], width=width, label="DA3", color="#2f7f6f")
        ax.set_xticks(indices)
        ax.set_xticklabels([row[0] for row in filtered], rotation=20, ha="right")
        ax.set_ylabel("Mean Sim3 RMSE (m)")
        ax.set_title("Mean TUM Sim3 Error by SVO Variant vs DA3")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "mean_sim3_rmse.png", dpi=180)
        plt.close(fig)

    print(json.dumps({"output_dir": str(output_dir), "variants": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
