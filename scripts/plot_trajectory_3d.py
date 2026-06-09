#!/usr/bin/env python3
"""Plot a 3D trajectory from an SVO trajectory text file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 3D trajectory from a text file.")
    parser.add_argument("traj_file", type=Path, help="Trajectory text file to plot.")
    parser.add_argument(
        "--groundtruth",
        type=Path,
        help="Optional ground-truth trajectory file to overlay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to <traj_file stem>_3d.png.",
    )
    parser.add_argument(
        "--title",
        default="3D Trajectory",
        help="Plot title.",
    )
    return parser.parse_args()


def load_positions(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 8:
        return data[:, 1:4]
    if data.shape[1] >= 9:
        return data[:, 2:5]
    raise ValueError(f"Unsupported trajectory format in {path}")


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def main() -> int:
    args = parse_args()
    traj = load_positions(args.traj_file)
    gt = load_positions(args.groundtruth) if args.groundtruth else None

    out_path = args.output or args.traj_file.with_name(f"{args.traj_file.stem}_3d.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#b55d3d", linewidth=1.6, label="estimate")
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color="#1f4f99", s=30, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color="#2f7f6f", s=30, label="end")

    all_points = traj
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="#444444", linewidth=1.1, alpha=0.8, label="ground truth")
        all_points = np.vstack([traj, gt])

    set_equal_axes(ax, all_points)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(args.title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
