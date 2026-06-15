# Monocular 3D Reconstruction Benchmark

This repository hosts the current `SVO` and `Depth Anything 3` comparative study. The active direction is now narrower and more explicit:

1. use `SVO` and `SVO + IMU` for trajectory anchoring and evaluation,
2. use `Depth Anything 3 (DA3)` and `DA3-Streaming` for learned dense reconstruction,
3. evaluate trajectory quantitatively where ground truth exists,
4. treat dense reconstruction as qualitative unless dense geometry ground truth is available.

## Active Scope

### Active methods

- `SVO Pro Open`
- `SVO + IMU`
- `Depth Anything 3`
- `DA3-Streaming`

### Active datasets

- `EuRoC` for visual-inertial trajectory benchmarking
- `7sense` for RGB-only qualitative and monocular comparison work already completed
- `OpenLORIS` as the current candidate for an RGB + IMU + GT trajectory benchmark
- `QCar Evry` retained as local project data

### De-emphasized or archival datasets

- `ETH3D`
- `Redwood`
- `TartanAir`
- older ad hoc `custom` experiments

These may still exist on disk, but they are no longer part of the main paper path.

## Main Research Questions

### Trajectory and odometry

- How much does IMU improve `SVO` trajectory quality and stability?
- In which regimes does `SVO + IMU` remain valuable as a geometric anchor even when a learned model exists?

### Learned reconstruction

- How strong is `DA3` as a standalone monocular trajectory and reconstruction baseline?
- How sensitive is learned pose quality to domain shift and sensing regime?
- When trajectory is not enough, what qualitative evidence supports or weakens a fused `SVO + GFM` story?

## Canonical Entry Points

### Setup

- `bash scripts/setup_da3.sh`
- `bash scripts/build_svo.sh`
- `bash scripts/enter_svo_container.sh`

### Run

- `bash scripts/run_svo_euroc_batch.sh`
- `bash scripts/run_svo_7sense_batch.sh`
- `bash scripts/run_svo_qcar_batch.sh`
- `python3 scripts/run_da3_euroc_batch.py --run`
- `python3 scripts/run_da3_7sense_batch.py --run`
- `python3 scripts/run_da3_qcar_batch.py --run`

### Evaluation

- `python3 scripts/evaluate_tum_da3_batch.py`
- `python3 scripts/evaluate_qcar_trajectories.py`
- `python3 scripts/evaluate_7sense_trajectories.py`

## Repository Layout

```text
datasets/                      raw and prepared data
environment/                   docker, ROS, and local env helpers
methods/                       active upstream method repos
outputs/                       raw run artifacts, logs, reconstructions
reports/evaluation/            canonical comparison bundles and runbooks
reports/final_comparative_report/
scripts/                       active orchestration scripts
scripts/archive/               one-off diagnostics and older experiments
```

## Evaluation Conventions

Trajectory outputs are standardized as:

```text
timestamp tx ty tz qx qy qz qw
```

Dense reconstruction outputs are stored under:

```text
outputs/reconstructions/{method}/{dataset}/{sequence}/
```

Machine-readable evaluation bundles live under:

```text
reports/evaluation/
```

## Current Notes

- The only active odometry path is `SVO`.
- The only active learned reconstruction path is `Depth Anything 3`, especially `DA3-Streaming`.
- `EuRoC` remains the main quantitative visual-inertial benchmark already integrated.
- Current paper-facing figures and PDFs live under `reports/final_comparative_report/`.

Useful starting points:

- evaluation index: [reports/evaluation/README.md](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/README.md)
- EuRoC runbook: [reports/evaluation/runbooks/SVO_EuRoC_Runbook.md](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/runbooks/SVO_EuRoC_Runbook.md)
- DA3 runbook: [reports/evaluation/runbooks/DA3_Streaming_Runbook.md](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/runbooks/DA3_Streaming_Runbook.md)
