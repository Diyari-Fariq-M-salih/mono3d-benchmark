# Installation Notes

This folder tracks how each dataset and method should be installed in a reproducible way.

## Principles

- Use one clean setup path per method.
- Prefer `venv` for Python-heavy methods.
- Prefer Docker or ROS workspaces when host installation becomes painful.
- Keep original repositories under `methods/`.
- Keep datasets under `datasets/`.
- Record every install decision before benchmarking.
- Install the minimal working dependency set first, then add optional extras only if benchmark features require them.
- Avoid duplicate clones, duplicate datasets, and avoid installing full optional bundles unless they are needed.

## Current Environment Constraints

- `python3` is available
- `venv` is available
- `git` is available
- `conda` is not installed on this machine
- `docker` exists, but needs execution outside the sandbox to run normally
- hardware from `reports/pc_specs.txt`: about `31 GiB` RAM and an `RTX 4070` with about `12 GiB` VRAM

## Hardware-Aware Decisions

- Keep batch size, frame count, and checkpoint size conservative for `Depth Anything 3` and `DA3-Streaming`.
- Prefer one clean Python environment for `Depth Anything 3` rather than maintaining multiple competing reconstruction stacks.
- Use `SVO Pro Open` as the only active odometry method implementation path for the current phase.

## Planned Installation Order

1. Benchmark utility venv
2. EuRoC dataset
3. SVO Pro Open
4. Depth Anything 3
5. DA3-Streaming experiments in the same upstream repo
6. ETH3D later if reconstruction evaluation needs it

## Dataset Download Policy

- Use `scripts/download_datasets.py` with named presets from `datasets/download_manifest.json`.
- For manual and repo-style download flows, use the explicit shell scripts in `scripts_downloading/`.
- For now, prioritize `EuRoC`. `ETH3D` can stay deferred until the `SVO + DA3` pipeline is stable.
- EuRoC archives can be extracted directly after download.
- ETH3D uses `.7z` archives and this machine now has `7z` available, so downloaded archives can be normalized into extracted scene folders immediately.
- EuRoC is currently documented as a manual-download step in `scripts_downloading/euroc_manual_download.md`.
- Current backup state:
- `datasets/euroc/` needs to be restored on this machine.
- `datasets/eth3d/` is intentionally deferred for now.

## Deferred Items

- `ETH3D` is deferred until the active `SVO + DA3` integration path is working.
- Additional reconstruction baselines are intentionally removed from the active plan to keep the internship scope focused.

## Active Odometry Path

- `SVO Pro Open` is the only active odometry method for the current phase.
- Use the container image `mono3d-benchmark/svo:noetic`.
- Use `scripts/setup_svo_workspace.sh` and `scripts/build_svo.sh` inside that container for the reproducible workspace flow.
- Use `scripts/enter_svo_container.sh` for interactive runs; it now starts the container with NVIDIA runtime access and opens at `/workspace`.
- The current benchmarked EuRoC odometry sweep is a full 11-sequence `mono` vs `mono-imu` comparison with CPU, memory, and GPU logging enabled on fresh runs.

## Tracking

Use `environment/install_notes/install_matrix.md` to log status and issues for each item.
