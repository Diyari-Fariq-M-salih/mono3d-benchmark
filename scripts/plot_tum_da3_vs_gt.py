#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_RUNS_ROOT = REPO_ROOT / "outputs" / "reconstructions" / "depth_anything_3" / "hall_da3_large_matrix" / "runs"
REPORT_ROOT = REPO_ROOT / "reports" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone TUM DA3-vs-GT report from best DA3 runs.")
    parser.add_argument("--da3-runs-root", type=Path, default=DA3_RUNS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_ROOT / "tum_da3_vs_gt_20260610_091041_mono3d_tum_mono",
    )
    return parser.parse_args()


def load_tum_trajectory(path: Path):
    import numpy as np

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def time_match_positions(gt, est, max_dt: float = 0.02):
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


def plot_single_bars(sequences, values, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(sequences) * 1.2), 5.8))
    indices = list(range(len(sequences)))
    ax.bar(indices, values, color="#3b6cb7")
    ax.set_xticks(indices)
    ax.set_xticklabels(sequences, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_views(sequence: str, gt_pos, da3_pos, out_top: Path, out_side: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth", color="#111111", linewidth=2.0)
    ax.plot(da3_pos[:, 0], da3_pos[:, 1], label="DA3", color="#3b6cb7", linewidth=1.5, alpha=0.9)
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
    ax.plot(da3_pos[:, 0], da3_pos[:, 2], label="DA3", color="#3b6cb7", linewidth=1.5, alpha=0.9)
    ax.set_title(f"{sequence} Side View")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_side, dpi=180)
    plt.close(fig)


def load_best_da3(runs_root: Path) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        summary_path = run_dir / "eval_tum" / "trajectory_summary.json"
        est_path = run_dir / "eval_tum" / "estimate_raw_tum.txt"
        if not summary_path.exists() or not est_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        sequence = run_dir.name.split("__fps", 1)[0]
        sim3_rmse = payload.get("sim3", {}).get("rmse_m")
        if sim3_rmse is None:
            continue
        current = best.get(sequence)
        if current is None or float(sim3_rmse) < float(current["trajectory_summary"]["sim3"]["rmse_m"]):
            best[sequence] = {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "trajectory_summary": payload,
            }
    if not best:
        raise FileNotFoundError(f"No evaluable TUM DA3 runs found under {runs_root}")
    return best


def main() -> int:
    args = parse_args()
    runs_root = args.da3_runs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_by_sequence = load_best_da3(runs_root)
    sequences = sorted(best_by_sequence)

    rows = []
    for sequence in sequences:
        best = best_by_sequence[sequence]
        traj = best["trajectory_summary"]
        rows.append(
            {
                "sequence": sequence,
                "run_name": best["run_name"],
                "match_count": traj["match_count"],
                "se3_rmse_m": traj["se3"]["rmse_m"],
                "sim3_rmse_m": traj["sim3"]["rmse_m"],
                "sim3_scale": traj["sim3_scale"],
                "path_ratio_est_over_gt": traj["path_ratio_est_over_gt"],
            }
        )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_single_bars(
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["sim3"]["rmse_m"] for seq in sequences],
        "Sim3 RMSE (m)",
        "TUM DA3 vs GT: Best Sim3 Error Per Sequence",
        output_dir / "sim3_rmse_best.png",
    )
    plot_single_bars(
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["se3"]["rmse_m"] for seq in sequences],
        "SE3 RMSE (m)",
        "TUM DA3 vs GT: Best Absolute Error Per Sequence",
        output_dir / "se3_rmse_best.png",
    )
    plot_single_bars(
        sequences,
        [best_by_sequence[seq]["trajectory_summary"]["path_ratio_est_over_gt"] for seq in sequences],
        "Estimated / GT Path Ratio",
        "TUM DA3 vs GT: Best Path-Length Consistency",
        output_dir / "path_ratio_best.png",
    )

    for sequence in sequences:
        best = best_by_sequence[sequence]
        run_dir = Path(best["run_dir"])
        est = load_tum_trajectory(run_dir / "eval_tum" / "estimate_raw_tum.txt")
        gt = load_tum_trajectory(Path(best["trajectory_summary"]["tum_sequence_dir"]) / "groundtruth.txt")
        gt_pos, est_pos = time_match_positions(gt, est)
        if len(gt_pos) < 2 or len(est_pos) < 2:
            continue
        da3_aligned = sim3_align(est_pos, gt_pos)
        plot_trajectory_views(
            sequence,
            gt_pos,
            da3_aligned,
            output_dir / f"{sequence}_trajectory_top.png",
            output_dir / f"{sequence}_trajectory_side.png",
        )

    (output_dir / "best_by_sequence.json").write_text(json.dumps(best_by_sequence, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "sequences": sequences}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
