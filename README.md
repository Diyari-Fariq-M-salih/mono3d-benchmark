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

1. Run `SVO` in monocular mode
2. Run `SVO + IMU` if available
3. Evaluate trajectories with `evo`

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
- installation registry and environment bootstrap helpers
- the comparative-study handoff document

Current hardware notes from `reports/pc_specs.txt`:

- system memory: about `31 GiB`
- GPU: `NVIDIA GeForce RTX 4070`
- GPU memory: about `12 GiB`

Because of that constraint, the benchmark should prefer `LiteVGGT`-style setups and smaller checkpoints over full-size `VGGT` defaults when possible.

Environment strategy for reproducibility:

- keep installs scripted and registry-driven
- prefer minimal dependency installs before optional extras
- avoid duplicate dataset copies
- use isolated environments when needed, but avoid unnecessary heavyweight extras
- prefer method-specific repos that are actively aligned with the intended benchmark setup
- for classical C++ odometry, use `SVO Pro Open` in its own ROS-oriented environment

Current implementation note:

- `SVO Pro Open` is now the only active odometry method path.

The next practical step is to stage the `SVO Pro Open` environment and wire the first real odometry method into `wrappers/run_svo.py`.
