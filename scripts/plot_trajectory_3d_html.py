#!/usr/bin/env python3
"""Generate an interactive HTML 3D viewer for SVO trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive 3D trajectory HTML viewer.")
    parser.add_argument("estimate", type=Path, help="Estimated trajectory file.")
    parser.add_argument("--groundtruth", type=Path, required=True, help="Ground-truth trajectory file.")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--title", default="3D Trajectory Viewer", help="Viewer title.")
    parser.add_argument(
        "--sim3-only",
        action="store_true",
        help="Write a viewer with only ground truth and Sim(3)-aligned estimate.",
    )
    return parser.parse_args()


def load_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 8:
        return data[:, 0], data[:, 1:4]
    if data.shape[1] >= 9:
        return data[:, 1], data[:, 2:5]
    raise ValueError(f"Unsupported trajectory format in {path}")


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
    scale = float(np.trace(np.diag(singular_values) @ sign) / var_est)
    trans = mu_gt - scale * (rot @ mu_est)
    return scale, rot, trans


def matched_positions(est_t: np.ndarray, est_p: np.ndarray, gt_t: np.ndarray, gt_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = idx - 1
    right = idx
    use_right = np.abs(gt_t[right] - est_t) < np.abs(gt_t[left] - est_t)
    match_idx = np.where(use_right, right, left)
    matched_gt_t = gt_t[match_idx]
    matched_gt_p = gt_p[match_idx]
    time_err = np.abs(matched_gt_t - est_t)
    mask = time_err <= 0.01
    return est_p[mask], matched_gt_p[mask]


def trace3d(points: np.ndarray, name: str, color: str, width: float, dash: str | None = None, visible: bool = True) -> go.Scatter3d:
    line = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        name=name,
        line=line,
        visible=True if visible else "legendonly",
    )


def main() -> int:
    args = parse_args()
    est_t, est_p = load_positions(args.estimate)
    gt_t, gt_p = load_positions(args.groundtruth)
    est_match, gt_match = matched_positions(est_t, est_p, gt_t, gt_p)

    rot_se3, trans_se3 = rigid_align(est_match, gt_match)
    est_se3 = (rot_se3 @ est_match.T).T + trans_se3

    scale_sim3, rot_sim3, trans_sim3 = sim3_align(est_match, gt_match)
    est_sim3 = (scale_sim3 * (rot_sim3 @ est_match.T)).T + trans_sim3

    fig = go.Figure()
    fig.add_trace(trace3d(gt_match, "Ground Truth", "#444444", 5))
    fig.add_trace(trace3d(est_sim3, f"Estimate Sim3 Aligned (scale={scale_sim3:.3f})", "#2f7f6f", 5))
    if not args.sim3_only:
        fig.add_trace(trace3d(est_match, "Estimate Raw", "#b55d3d", 4, visible=False))
        fig.add_trace(trace3d(est_se3, "Estimate SE3 Aligned", "#1f77b4", 4, dash="dot", visible=False))

    fig.update_layout(
        title=args.title,
        template="plotly_white",
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
