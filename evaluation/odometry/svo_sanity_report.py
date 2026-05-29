#!/usr/bin/env python3
"""Compute a lightweight ground-truth sanity report for SVO outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SVO trajectory sanity against ground truth.")
    parser.add_argument("trace_dir", type=Path, help="Path to one SVO trace directory")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional file to write the JSON report to",
    )
    return parser.parse_args()


def load_status_counts(status_path: Path) -> dict[str, int]:
    counts = {
        "frames_total": 0,
        "tracking_frames": 0,
        "initializing_frames": 0,
        "relocalization_frames": 0,
        "paused_frames": 0,
        "good_quality_frames": 0,
        "insufficient_quality_frames": 0,
        "failure_updates": 0,
        "kf_updates": 0,
        "default_updates": 0,
    }

    with status_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            counts["frames_total"] += 1
            stage, quality, update = parts[2], parts[3], parts[4]
            if stage == "Tracking":
                counts["tracking_frames"] += 1
            elif stage == "Initializing":
                counts["initializing_frames"] += 1
            elif stage == "Reloc":
                counts["relocalization_frames"] += 1
            elif stage == "Paused":
                counts["paused_frames"] += 1

            if quality == "Good":
                counts["good_quality_frames"] += 1
            elif quality == "Insufficient":
                counts["insufficient_quality_frames"] += 1

            if update == "Failure":
                counts["failure_updates"] += 1
            elif update == "KF":
                counts["kf_updates"] += 1
            elif update == "Default":
                counts["default_updates"] += 1
    return counts


def rigid_align(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_centered = est - mu_est
    gt_centered = gt - mu_gt
    cov = est_centered.T @ gt_centered / len(est)
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = mu_gt - rot @ mu_est
    return rot, trans


def sim3_align(est: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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
    scale = np.trace(np.diag(singular_values) @ sign) / var_est
    trans = mu_gt - scale * (rot @ mu_est)
    return float(scale), rot, trans


def error_stats(err: np.ndarray) -> dict[str, float]:
    return {
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mean_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "max_m": float(np.max(err)),
    }


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()

    status_path = trace_dir / "status.txt"
    gt_path = trace_dir / "stamped_groundtruth.txt"
    est_path = trace_dir / "stamped_traj_estimate.txt"

    counts = load_status_counts(status_path)
    gt = np.loadtxt(gt_path)
    est = np.loadtxt(est_path)

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
    time_mask = time_err <= 0.01
    est_t = est_t[time_mask]
    est_p = est_p[time_mask]
    matched_gt_p = matched_gt_p[time_mask]
    time_err = time_err[time_mask]

    rot_se3, trans_se3 = rigid_align(est_p, matched_gt_p)
    aligned_se3 = (rot_se3 @ est_p.T).T + trans_se3
    err_se3 = np.linalg.norm(aligned_se3 - matched_gt_p, axis=1)

    scale_sim3, rot_sim3, trans_sim3 = sim3_align(est_p, matched_gt_p)
    aligned_sim3 = (scale_sim3 * (rot_sim3 @ est_p.T)).T + trans_sim3
    err_sim3 = np.linalg.norm(aligned_sim3 - matched_gt_p, axis=1)

    est_path_len = float(np.linalg.norm(np.diff(est_p, axis=0), axis=1).sum())
    gt_path_len = float(np.linalg.norm(np.diff(matched_gt_p, axis=0), axis=1).sum())

    report = {
        "trace_dir": str(trace_dir),
        **counts,
        "tracking_ratio": counts["tracking_frames"] / counts["frames_total"] if counts["frames_total"] else 0.0,
        "trajectory_rows": int(est.shape[0]),
        "trajectory_ratio": est.shape[0] / counts["frames_total"] if counts["frames_total"] else 0.0,
        "timestamp_match": {
            "matched_rows": int(est_t.shape[0]),
            "median_abs_dt_s": float(np.median(time_err)),
            "max_abs_dt_s": float(np.max(time_err)),
        },
        "se3": error_stats(err_se3),
        "sim3": error_stats(err_sim3),
        "scale_path": {
            "sim3_scale": scale_sim3,
            "est_path_m": est_path_len,
            "gt_path_m": gt_path_len,
            "path_ratio_est_over_gt": est_path_len / gt_path_len if gt_path_len > 0 else 0.0,
        },
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
