# Monocular 3D Reconstruction Benchmark

This repository hosts a comparative study of monocular odometry and dense or streaming 3D reconstruction methods.

The benchmark is intentionally split into two tracks:

1. Odometry and trajectory estimation
2. Dense or online 3D reconstruction

These tracks overlap, but they do not answer the same research question and should not be collapsed into a single score.

## Research Questions

### Odometry

- Does adding IMU improve monocular tracking enough to justify the extra system complexity?
- How much do drift, scale stability, and tracking robustness improve when moving from monocular to mono-inertial pipelines?

### Reconstruction

- Which methods provide the most accurate and useful geometry from monocular video?
- Which methods can operate online or in a streaming setting?
- Do streaming or long-context methods solve the chunk consistency issues seen with local feed-forward reconstruction?

## Method Groups

### Odometry and SLAM

- `ORB-SLAM3`
- `DSO`
- `SVO`
- `SVO + IMU` if supported by the installed version

### Dense and Streaming Reconstruction

- `Depth Anything 3`
- `DA3-Streaming`
- `LiteVGGT`
- `MapAnything`
- `STAC-3R`

## Datasets

### Odometry

Primary benchmark: `EuRoC MAV`

Suggested first sequences:

- `MH_01_easy`
- `MH_03_medium`
- `V1_02_medium`

### Reconstruction

Primary benchmark: `ETH3D`

Suggested first pass:

- One small indoor scene
- One larger indoor scene

## First Milestone

The first benchmark milestone is a minimal odometry baseline on EuRoC:

1. Run `ORB-SLAM3` in monocular mode
2. Run `ORB-SLAM3` in mono-inertial mode
3. Run `SVO` in monocular mode
4. Run `SVO + IMU` if available
5. Run `DSO`
6. Evaluate trajectories with `evo`

## Output Conventions

Trajectory output:

```text
timestamp tx ty tz qx qy qz qw
```

Stored under:

```text
outputs/trajectories/{method}/{dataset}/{sequence}.txt
```

Reconstruction output:

- `PLY` point clouds
- `PLY` meshes when available
- optional `NPZ` side outputs

Stored under:

```text
outputs/reconstructions/{method}/{dataset}/{sequence}/
```

Run metadata should be recorded in machine-readable form under `outputs/logs/`.

## Metrics

### Odometry

- `ATE RMSE`
- `RPE translation`
- `RPE rotation`
- `scale error`
- `tracking lost count`
- `initialization success`
- `FPS`
- runtime
- memory usage

### Reconstruction

- `accuracy`
- `completeness`
- `F-score`
- `Chamfer distance`
- runtime
- GPU memory
- maximum sequence length before failure

## Repository Layout

```text
environment/
datasets/
methods/
wrappers/
evaluation/
outputs/
reports/
scripts/
```

## Current Status

This repository currently contains:

- project scaffolding
- starter wrapper and evaluation scripts
- the comparative-study handoff document

The next practical step is to wire the first real odometry method into `wrappers/run_orb_slam3.py` and standardize EuRoC input preparation.
