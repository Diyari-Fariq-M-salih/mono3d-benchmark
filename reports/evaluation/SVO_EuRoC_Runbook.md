# SVO EuRoC Runbook

This document explains how to repeat the full `SVO Pro Open` EuRoC odometry workflow in this repo:

- unpack EuRoC archives
- enter the SVO container
- run `mono`
- run `mono-imu`
- generate sanity reports
- generate comparison plots
- handle the one special-case sequence: `V2_03_difficult`

This runbook assumes you are starting from the repo root:

```bash
cd /home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark
```

## 1. What Is Already Automated

These scripts are the main entry points:

- [scripts/unpack_euroc_all.sh](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/unpack_euroc_all.sh)
- [scripts/enter_svo_container.sh](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/enter_svo_container.sh)
- [scripts/prepare_svo_euroc.py](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/prepare_svo_euroc.py)
- [scripts/run_svo_euroc_batch.sh](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/run_svo_euroc_batch.sh)
- [evaluation/odometry/svo_sanity_report.py](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/evaluation/odometry/svo_sanity_report.py)
- [scripts/plot_svo_sanity_compare.py](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/plot_svo_sanity_compare.py)

What they do:

- `unpack_euroc_all.sh`: expands grouped EuRoC archives into per-sequence folders under `datasets/euroc/`
- `prepare_svo_euroc.py`: converts extracted EuRoC sequences into SVO benchmarking format
- `run_svo_euroc_batch.sh`: runs SVO on one or more EuRoC sequences in `mono`, `mono-imu`, or `both`
- `svo_sanity_report.py`: compares each run against ground truth and writes a JSON summary
- `plot_svo_sanity_compare.py`: compares `mono` vs `mono-imu` across multiple sequences and makes plots

## 2. EuRoC Dataset Layout

Expected extracted layout:

```text
datasets/euroc/
├── MH_01_easy/
│   └── mav0/
├── MH_02_easy/
│   └── mav0/
├── ...
└── V2_03_difficult/
    └── mav0/
```

Current full EuRoC sequence set:

- `MH_01_easy`
- `MH_02_easy`
- `MH_03_medium`
- `MH_04_difficult`
- `MH_05_difficult`
- `V1_01_easy`
- `V1_02_medium`
- `V1_03_difficult`
- `V2_01_easy`
- `V2_02_medium`
- `V2_03_difficult`

## 3. Unpack EuRoC Archives

If you already have extracted sequence folders, you can skip this section.

If you only have the grouped archives such as:

- `datasets/euroc/machine_hall.zip`
- `datasets/euroc/vicon_room1.zip`
- `datasets/euroc/vicon_room2.zip`

run:

```bash
bash scripts/unpack_euroc_all.sh
```

Then verify:

```bash
find datasets/euroc -maxdepth 1 -mindepth 1 -type d | sort
```

## 4. Enter the SVO Container

Start the SVO container:

```bash
bash scripts/enter_svo_container.sh
```

Inside the container:

```bash
cd /workspace
source /opt/ros/noetic/setup.bash
source /workspace/environment/ros_ws/svo/devel/setup.bash
```

If the workspace was already built, this is enough to run SVO.

## 5. Prepare SVO Benchmark Datasets

Prepare all extracted EuRoC sequences for SVO benchmarking:

```bash
python3 scripts/prepare_svo_euroc.py --write-configs
```

Prepare only one sequence:

```bash
python3 scripts/prepare_svo_euroc.py --write-configs --sequence MH_01_easy
```

This creates SVO benchmark-format folders under:

```text
methods/svo/upstream/svo_benchmarking/data/mono3d/euroc/mono/<SEQUENCE>/
```

and experiment YAMLs under:

```text
methods/svo/upstream/svo_benchmarking/experiments/
```

## 6. Run SVO

### Run one sequence in mono

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono --sequence MH_01_easy
```

### Run one sequence in mono+imu

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono-imu --sequence MH_01_easy
```

### Run one sequence in both modes

```bash
bash scripts/run_svo_euroc_batch.sh --mode both --sequence MH_01_easy
```

### Run all prepared EuRoC sequences in mono

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono
```

### Run all prepared EuRoC sequences in mono+imu

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono-imu
```

### Run all prepared EuRoC sequences in both modes

```bash
bash scripts/run_svo_euroc_batch.sh --mode both
```

## 7. Where the Outputs Go

Mono runs:

```text
outputs/logs/svo_benchmarks/mono3d_euroc_mono/<timestamp>_mono3d_euroc_mono/<SEQUENCE>/
```

Mono-IMU runs:

```text
outputs/logs/svo_benchmarks/mono3d_euroc_mono_imu/<timestamp>_mono3d_euroc_mono_imu/<SEQUENCE>/
```

Each per-sequence trace folder contains files such as:

- `status.txt`
- `stamped_traj_estimate.txt`
- `traj_estimate.txt`
- `stamped_groundtruth.txt`
- `groundtruth.txt`
- `speed_bias_estimate.txt` for IMU/backend runs
- `trace_frontend.csv`
- `trace_backend.csv`
- `sanity_report.json`
- `log_cpu_usage.txt`
- `log_memory_usage.txt`
- `log_gpu_usage.txt`

## 8. Automatic CPU, Memory, and GPU Logging

Future runs now record:

- CPU usage: `log_cpu_usage.txt`
- memory usage: `log_memory_usage.txt`
- GPU usage: `log_gpu_usage.txt`

GPU logging uses `nvidia-smi` if available inside the container.

If `nvidia-smi` is not available, `log_gpu_usage.txt` will still be created, but it will contain a note saying GPU logging was skipped.

## 9. Single-Sequence Sanity Report

If you want to recompute a sanity report manually for one trace:

```bash
python3 evaluation/odometry/svo_sanity_report.py \
  outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono/MH_01_easy
```

This compares the estimated trajectory to ground truth and reports:

- tracking ratio
- trajectory coverage ratio
- `SE3` RMSE
- `Sim3` RMSE
- implied scale factor
- estimated / GT path ratio

## 10. Mono vs Mono-IMU Comparison Plots

### Compare the latest runs automatically

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python3 scripts/plot_svo_sanity_compare.py
```

### Compare two specific runs

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python3 scripts/plot_svo_sanity_compare.py \
  --mono-dir outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono \
  --mono-imu-dir outputs/logs/svo_benchmarks/mono3d_euroc_mono_imu/20260601_081241_mono3d_euroc_mono_imu
```

### Write plots into a chosen folder

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python3 scripts/plot_svo_sanity_compare.py \
  --mono-dir outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono \
  --mono-imu-dir outputs/logs/svo_benchmarks/mono3d_euroc_mono_imu/20260601_081241_mono3d_euroc_mono_imu \
  --output-dir reports/evaluation/my_compare_run
```

The comparison script creates:

- `summary.csv`
- `tracking_ratio.png`
- `trajectory_ratio.png`
- `se3_rmse.png`
- `sim3_rmse.png`
- `path_ratio.png`
- per-sequence top-view trajectory plots
- per-sequence side-view trajectory plots

## 11. Full 11-Sequence Comparison

The current full comparison bundle is here:

[reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601)

It uses:

- `mono` from:
  [outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono)
- `mono-imu` from a merged directory:
  [reports/evaluation/mono_imu_merged_20260601](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/mono_imu_merged_20260601)

That merged directory exists because `V2_03_difficult` needed a later rerun after a sequence-specific fix.

## 12. Special Case: `V2_03_difficult`

`V2_03_difficult` needed a special first-frame override.

Why:

- the first image timestamp was about `20 ms` earlier than the first IMU sample
- `mono-imu` backend initialization failed when frame `0` had no IMU history

Current fix:

- in [scripts/prepare_svo_euroc.py](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/scripts/prepare_svo_euroc.py)
- `FIRST_FRAME_OVERRIDES["V2_03"] = 1`

If a future EuRoC-like sequence fails very early in `mono-imu`, inspect:

- first image timestamp
- first IMU timestamp
- whether image `0` occurs before the IMU stream begins

If yes, add a first-frame override for that sequence.

## 13. Re-running `V2_03_difficult`

Inside the container:

```bash
python3 scripts/prepare_svo_euroc.py --write-configs --sequence V2_03_difficult
bash scripts/run_svo_euroc_batch.sh --mode mono-imu --sequence V2_03_difficult
```

Latest successful `mono-imu` `V2_03_difficult` trace:

[outputs/logs/svo_benchmarks/mono3d_euroc_mono_imu/20260601_084938_mono3d_euroc_mono_imu/V2_03_difficult](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/outputs/logs/svo_benchmarks/mono3d_euroc_mono_imu/20260601_084938_mono3d_euroc_mono_imu/V2_03_difficult)

## 14. New EuRoC Datasets in the Future

If you add new EuRoC sequences later:

1. place them under `datasets/euroc/<SEQUENCE>/mav0/...`
2. enter the container
3. regenerate the SVO benchmark data
4. rerun the desired mode

Typical flow:

```bash
bash scripts/enter_svo_container.sh
cd /workspace
source /opt/ros/noetic/setup.bash
source /workspace/environment/ros_ws/svo/devel/setup.bash
python3 scripts/prepare_svo_euroc.py --write-configs
bash scripts/run_svo_euroc_batch.sh --mode mono
bash scripts/run_svo_euroc_batch.sh --mode mono-imu
```

If only one new sequence was added:

```bash
python3 scripts/prepare_svo_euroc.py --write-configs --sequence NEW_SEQUENCE_NAME
bash scripts/run_svo_euroc_batch.sh --mode both --sequence NEW_SEQUENCE_NAME
```

## 15. Permission Caveat

Some older files in:

- `outputs/logs/svo_benchmarks/...`
- `methods/svo/upstream/svo_benchmarking/data/...`

were created from inside Docker and may be owned by `root` or `nobody`.

Practical rule:

- do benchmark prep and SVO runs inside the container
- do host-side reporting into writable folders such as `reports/evaluation`

If you need to clean ownership later, do it consciously from the host with `sudo`, not by deleting files blindly.

## 16. ETH3D Note

ETH3D is already extracted for the reconstruction track, but it is not used by the SVO odometry pipeline.

Current ETH3D scenes:

- `courtyard`
- `delivery_area`
- `electro`

## 17. Minimal Command Cheatsheet

### Enter SVO container

```bash
bash scripts/enter_svo_container.sh
```

### Inside container: source workspace

```bash
cd /workspace
source /opt/ros/noetic/setup.bash
source /workspace/environment/ros_ws/svo/devel/setup.bash
```

### Prepare all EuRoC sequences

```bash
python3 scripts/prepare_svo_euroc.py --write-configs
```

### Run all mono

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono
```

### Run all mono-imu

```bash
bash scripts/run_svo_euroc_batch.sh --mode mono-imu
```

### Plot a chosen comparison

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig python3 scripts/plot_svo_sanity_compare.py \
  --mono-dir outputs/logs/svo_benchmarks/mono3d_euroc_mono/20260601_080900_mono3d_euroc_mono \
  --mono-imu-dir reports/evaluation/mono_imu_merged_20260601 \
  --output-dir reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601
```

## 18. Current Bottom Line

For this repo, the recommended odometry workflow is:

1. `mono` as the baseline
2. `mono-imu` as the main result
3. use `sanity_report.json` and the comparison plots as the first evaluation layer
4. use the generated CSV and figures in `reports/evaluation` for reporting and thesis drafting
