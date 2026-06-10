#!/usr/bin/env python3
"""Compare latest TUM SVO trajectories against the best DA3 TUM trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
SVO_BENCH_ROOT = REPO_ROOT / "outputs" / "logs" / "svo_benchmarks" / "mono3d_tum_mono"
DA3_RUNS_ROOT = (
    REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "hall_da3_large_matrix" / "runs"
)
REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"
SVO_SANITY_SCRIPT = REPO_ROOT / "evaluation" / "odometry" / "svo_sanity_report.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latest TUM SVO mono results against best DA3 trajectory results."
    )
    parser.add_argument(
        "--svo-dir",
        type=Path,
        help="Specific SVO TUM run directory. Defaults to the latest mono3d_tum_mono run.",
    )
    parser.add_argument(
        "--da3-runs-root",
        type=Path,
        default=DA3_RUNS_ROOT,
        help="Directory containing DA3 TUM run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for plots and CSV output. Defaults to reports/evaluation/tum_svo_vs_da3_<run>.",
    )
    return parser.parse_args()


def normalize_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    path_text = str(path)
    if path_text.startswith("/workspace/"):
        return REPO_ROOT / path_text[len("/workspace/") :]
    if "/datasets/" in path_text:
        suffix = path_text.split("/datasets/", 1)[1]
        candidate = REPO_ROOT / "datasets" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    if "/outputs/" in path_text:
        suffix = path_text.split("/outputs/", 1)[1]
        candidate = REPO_ROOT / "outputs" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    if "/reports/" in path_text:
        suffix = path_text.split("/reports/", 1)[1]
        candidate = REPO_ROOT / "reports" / suffix
        if candidate.exists() or not path.exists():
            return candidate
    repo_name = REPO_ROOT.name
    marker = f"/{repo_name}/"
    if marker in path_text:
        suffix = path_text.split(marker, 1)[1]
        return REPO_ROOT / suffix
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def latest_svo_run() -> Path:
    runs = sorted(path for path in SVO_BENCH_ROOT.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No SVO TUM runs found under {SVO_BENCH_ROOT}")
    return runs[-1]


def normalize_trace_dir(path_value: str | Path) -> Path:
    return normalize_repo_path(path_value)


def load_svo_reports(run_dir: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for seq_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        report_path = seq_dir / "sanity_report.json"
        if report_path.exists():
            reports[seq_dir.name] = json.loads(report_path.read_text(encoding="utf-8"))
            continue
        result = subprocess.run(
            ["python3", str(SVO_SANITY_SCRIPT), str(seq_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        reports[seq_dir.name] = json.loads(result.stdout)
    if not reports:
        raise FileNotFoundError(f"No SVO reports found under {run_dir}")
    return reports


def load_best_da3_reports(runs_root: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "eval_tum" / "trajectory_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sequence = run_dir.name.split("__fps", 1)[0]
        sim3_rmse = summary.get("sim3", {}).get("rmse_m")
        if sim3_rmse is None:
            continue
        current = best.get(sequence)
        if current is None or sim3_rmse < current["trajectory_summary"]["sim3"]["rmse_m"]:
            best[sequence] = {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "trajectory_summary": summary,
            }
    if not best:
        raise FileNotFoundError(f"No DA3 trajectory summaries found under {runs_root}")
    return best


def load_svo_pose_series(trace_dir: Path):
    import numpy as np

    trace_dir = normalize_trace_dir(trace_dir)
    gt = np.loadtxt(trace_dir / "stamped_groundtruth.txt")
    est = np.loadtxt(trace_dir / "stamped_traj_estimate.txt")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    if est.ndim == 1:
        est = est.reshape(1, -1)
    return gt, est


def load_da3_pose_series(run_dir: Path):
    import numpy as np

    est = np.loadtxt(run_dir / "eval_tum" / "estimate_raw_tum.txt")
    gt = np.loadtxt(run_dir / "eval_tum" / "estimate_raw_tum.txt")
    if est.ndim == 1:
        est = est.reshape(1, -1)
    gt_path = run_dir / "eval_tum" / "trajectory_summary.json"
    summary = json.loads(gt_path.read_text(encoding="utf-8"))
    tum_sequence_dir = normalize_repo_path(summary["tum_sequence_dir"])
    gt = np.loadtxt(tum_sequence_dir / "groundtruth.txt")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    return gt, est


def time_match_positions(gt, est, max_dt: float = 0.01):
    import numpy as np

    gt_t = gt[:, 0]
    gt_p = gt[:, 1:4]
    est_t = est[:, 0]
    est_p = est[:, 1:4]

    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = idx - 1
    right = idx
    use_right = np.abs(gt_t[right] - est_t) < np.abs(gt_t[left] - est_t)
    match_idx = np.where(use_right, right, left)
    matched_gt_t = gt_t[match_idx]
    matched_gt_p = gt_p[match_idx]

    time_err = np.abs(matched_gt_t - est_t)
    mask = time_err <= max_dt
    return matched_gt_p[mask], est_p[mask]


def sim3_align(est, gt):
    import numpy as np

    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_centered = est - mu_est
    gt_centered = gt - mu_gt
    var_est = (est_centered**2).sum() / len(est)
    cov = est_centered.T @ gt_centered / len(est)
    u, singular_values, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    rot = vt.T @ sign @ u.T
    scale = float((np.diag(singular_values) @ sign).trace() / var_est)
    trans = mu_gt - scale * (rot @ mu_est)
    return (scale * (rot @ est.T)).T + trans


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "svo_status",
        "svo_tracking_ratio",
        "svo_trajectory_ratio",
        "svo_se3_rmse_m",
        "svo_sim3_rmse_m",
        "svo_sim3_scale",
        "svo_path_ratio_est_over_gt",
        "da3_run_name",
        "da3_match_count",
        "da3_se3_rmse_m",
        "da3_sim3_rmse_m",
        "da3_sim3_scale",
        "da3_path_ratio_est_over_gt",
        "winner_by_sim3",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bars(plt, sequences, svo_values, da3_values, ylabel: str, title: str, out_path: Path) -> None:
    filtered = [
        (seq, s, d)
        for seq, s, d in zip(sequences, svo_values, da3_values)
        if s is not None and d is not None
    ]
    if not filtered:
        return
    seqs = [item[0] for item in filtered]
    svo = [float(item[1]) for item in filtered]
    da3 = [float(item[2]) for item in filtered]
    indices = list(range(len(seqs)))
    width = 0.38
    fig_width = max(10, len(seqs) * 1.15)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    ax.bar([i - width / 2 for i in indices], svo, width=width, label="SVO", color="#b55d3d")
    ax.bar([i + width / 2 for i in indices], da3, width=width, label="DA3", color="#2f7f6f")
    ax.set_xticks(indices)
    ax.set_xticklabels(seqs, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_views(plt, sequence: str, gt_pos, svo_pos, da3_pos, out_top: Path, out_side: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    if svo_pos is not None:
        ax.plot(svo_pos[:, 0], svo_pos[:, 1], label="SVO", color="#b55d3d", linewidth=1.5, alpha=0.9)
    if da3_pos is not None:
        ax.plot(da3_pos[:, 0], da3_pos[:, 1], label="DA3", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Top View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_top, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], label="ground truth", color="#111111", linewidth=2.0)
    if svo_pos is not None:
        ax.plot(svo_pos[:, 0], svo_pos[:, 2], label="SVO", color="#b55d3d", linewidth=1.5, alpha=0.9)
    if da3_pos is not None:
        ax.plot(da3_pos[:, 0], da3_pos[:, 2], label="DA3", color="#2f7f6f", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    svo_dir = args.svo_dir.resolve() if args.svo_dir else latest_svo_run()
    da3_runs_root = args.da3_runs_root.resolve() if args.da3_runs_root else DA3_RUNS_ROOT
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else REPORT_ROOT / f"tum_svo_vs_da3_{svo_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    svo_reports = load_svo_reports(svo_dir)
    da3_best = load_best_da3_reports(da3_runs_root)
    sequences = sorted(set(svo_reports) & set(da3_best))
    if not sequences:
        raise RuntimeError("No overlapping SVO and DA3 TUM sequences found.")

    rows: list[dict[str, object]] = []
    for sequence in sequences:
        svo = svo_reports[sequence]
        da3 = da3_best[sequence]
        svo_sim3 = svo["sim3"]["rmse_m"]
        da3_sim3 = da3["trajectory_summary"]["sim3"]["rmse_m"]
        if svo_sim3 is None and da3_sim3 is None:
            winner = ""
        elif svo_sim3 is None:
            winner = "DA3"
        elif da3_sim3 is None:
            winner = "SVO"
        else:
            winner = "SVO" if float(svo_sim3) < float(da3_sim3) else "DA3"
        rows.append(
            {
                "sequence": sequence,
                "svo_status": svo.get("status", "ok"),
                "svo_tracking_ratio": svo.get("tracking_ratio"),
                "svo_trajectory_ratio": svo.get("trajectory_ratio"),
                "svo_se3_rmse_m": svo["se3"]["rmse_m"],
                "svo_sim3_rmse_m": svo["sim3"]["rmse_m"],
                "svo_sim3_scale": svo["scale_path"]["sim3_scale"],
                "svo_path_ratio_est_over_gt": svo["scale_path"]["path_ratio_est_over_gt"],
                "da3_run_name": da3["run_name"],
                "da3_match_count": da3["trajectory_summary"]["match_count"],
                "da3_se3_rmse_m": da3["trajectory_summary"]["se3"]["rmse_m"],
                "da3_sim3_rmse_m": da3["trajectory_summary"]["sim3"]["rmse_m"],
                "da3_sim3_scale": da3["trajectory_summary"]["sim3_scale"],
                "da3_path_ratio_est_over_gt": da3["trajectory_summary"]["path_ratio_est_over_gt"],
                "winner_by_sim3": winner,
            }
        )
    write_csv(rows, output_dir / "summary.csv")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps({"svo_dir": str(svo_dir), "output_dir": str(output_dir), "summary_csv": str(output_dir / "summary.csv")}, indent=2))
        return 0

    plot_grouped_bars(
        plt,
        sequences,
        [svo_reports[seq]["sim3"]["rmse_m"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["sim3"]["rmse_m"] for seq in sequences],
        ylabel="Sim3 RMSE (m)",
        title="TUM Trajectory Shape Error: SVO vs Best DA3",
        out_path=output_dir / "sim3_rmse.png",
    )
    plot_grouped_bars(
        plt,
        sequences,
        [svo_reports[seq]["se3"]["rmse_m"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["se3"]["rmse_m"] for seq in sequences],
        ylabel="SE3 RMSE (m)",
        title="TUM Absolute Error: SVO vs Best DA3",
        out_path=output_dir / "se3_rmse.png",
    )
    plot_grouped_bars(
        plt,
        sequences,
        [svo_reports[seq]["scale_path"]["path_ratio_est_over_gt"] for seq in sequences],
        [da3_best[seq]["trajectory_summary"]["path_ratio_est_over_gt"] for seq in sequences],
        ylabel="Estimated / GT Path Ratio",
        title="TUM Scale Consistency: SVO vs Best DA3",
        out_path=output_dir / "path_ratio.png",
    )

    for sequence in sequences:
        gt_ref = None
        svo_aligned = None
        da3_aligned = None

        svo = svo_reports[sequence]
        if svo.get("status", "ok") == "ok":
            gt_svo, est_svo = load_svo_pose_series(normalize_trace_dir(svo["trace_dir"]))
            gt_match_svo, est_match_svo = time_match_positions(gt_svo, est_svo)
            if len(gt_match_svo) >= 2 and len(est_match_svo) >= 2:
                gt_ref = gt_match_svo
                svo_aligned = sim3_align(est_match_svo, gt_match_svo)

        da3_run_dir = Path(da3_best[sequence]["run_dir"])
        gt_da3, est_da3 = load_da3_pose_series(da3_run_dir)
        gt_match_da3, est_match_da3 = time_match_positions(gt_da3, est_da3)
        if len(gt_match_da3) >= 2 and len(est_match_da3) >= 2:
            if gt_ref is None or len(gt_match_da3) > len(gt_ref):
                gt_ref = gt_match_da3
            da3_aligned = sim3_align(est_match_da3, gt_match_da3)

        if gt_ref is None:
            continue

        plot_trajectory_views(
            plt,
            sequence=sequence,
            gt_pos=gt_ref,
            svo_pos=svo_aligned,
            da3_pos=da3_aligned,
            out_top=output_dir / f"{sequence}_trajectory_top.png",
            out_side=output_dir / f"{sequence}_trajectory_side.png",
        )

    print(
        json.dumps(
            {
                "svo_dir": str(svo_dir),
                "output_dir": str(output_dir),
                "summary_csv": str(output_dir / "summary.csv"),
                "sequences": sequences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
