# Comparative Study for Monocular Odometry and Online 3D Reconstruction

**Project context:** Master's internship / monocular 3D reconstruction benchmark  
**Date:** 2026-05-29  
**Prepared for:** Diyari / future Codex or ChatGPT continuation / supervisors

---

## 1. Core idea

The planned work is a comparative study of several monocular visual odometry, SLAM, visual-inertial, and feed-forward 3D reconstruction methods. The purpose is not simply to find "the best method" in a vague sense, but to answer two separate research questions:

1. **Odometry / trajectory estimation:**
   - For monocular camera motion estimation, is IMU worth including?
   - How much does mono-inertial improve robustness, scale, and drift compared with pure monocular?

2. **Dense / online 3D reconstruction:**
   - Which methods provide the most accurate and useful 3D reconstruction from monocular video?
   - Which methods can operate online or streaming, rather than only offline/batch?
   - Can modern feed-forward 3D models solve the chunk consistency problem observed with Depth Anything 3-style local reconstructions?

The study should be framed as:

> A comparative study of classical, direct, semi-direct, visual-inertial, and feed-forward foundation-model methods for monocular odometry and online 3D reconstruction.

Important: **do not collapse everything into one score.** Classical SLAM, VIO, and modern feed-forward reconstruction models solve overlapping but not identical problems. The report should separate trajectory accuracy from reconstruction quality.

---

## 2. Method families to compare

### 2.1 Odometry / SLAM / VIO group

These methods should be evaluated primarily on trajectory estimation, tracking robustness, scale drift, and runtime.

| Method | Category | IMU support? | Main output | Role in study |
|---|---|---:|---|---|
| ORB-SLAM3 | Feature-based SLAM | Yes, mono-inertial supported | trajectory + sparse map | Main classical baseline |
| DSO | Direct visual odometry | Usually monocular only in classic form | trajectory / sparse direct map | Direct-method baseline |
| SVO | Semi-direct visual odometry | Depends on version | trajectory / sparse or semi-sparse map | Internship-relevant baseline |
| SVO + IMU | Semi-direct VIO, if available | Yes if available in the installed version | trajectory | Test whether IMU helps SVO |

### 2.2 Dense / feed-forward / streaming reconstruction group

These methods should be evaluated primarily on geometry quality, temporal consistency, online capability, GPU memory, and runtime.

| Method | Category | Main output | Role in study |
|---|---|---|---|
| Depth Anything 3 | Feed-forward visual geometry | depth / point clouds / geometry | Current main geometry direction |
| DA3-Streaming | Streaming DA3 pipeline | long-video geometry with carried state | Directly relevant to chunk consistency problem |
| LiteVGGT | Feed-forward 3D foundation model | cameras, depth maps, point maps, tracks | Strong modern geometry baseline |
| MapAnything | Feed-forward metric 3D reconstruction | metric geometry, cameras, depth/rays | Strong flexible metric reconstruction baseline |
| STAC-3R | Streaming/cache-compressed reconstruction framework | streaming 3D reconstruction with bounded memory | Very relevant for long-video online reconstruction |

### 2.3 Important clarification about STAC

The intended STAC is **STAC-3R: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction**, not a generic tracking method. It should be placed in the **streaming reconstruction group**, not the odometry group.

STAC is relevant because the current DA3 chunk issue is also a long-sequence consistency problem. Previously, independent chunks such as `0-15`, `12-27`, `24-39` produced local reconstructions whose geometry changed with context. Sim(3) overlap-frame alignment was not always sufficient because the geometry itself could stretch/warp differently across chunks.

STAC-like methods address the broader problem by preserving or compressing long-term context during streaming reconstruction, rather than reconstructing isolated chunks and merging them afterward.

---

## 3. Dataset strategy

The dataset choice should match the evaluation question.

### 3.1 Odometry / IMU vs no-IMU: use EuRoC MAV

**Main dataset:** EuRoC MAV

Why:

- It contains stereo images, synchronized IMU measurements, and accurate motion/structure ground truth.
- It is one of the standard datasets for visual-inertial odometry and SLAM.
- It allows running the same camera sequence with and without IMU.

Use it to answer:

> Does IMU improve monocular odometry enough to justify including it in the final pipeline?

Recommended first sequences:

| Sequence | Difficulty | Purpose |
|---|---|---|
| `MH_01_easy` | Easy | first sanity test |
| `MH_02_easy` | Easy | repeatability |
| `MH_03_medium` | Medium | realistic comparison |
| `V1_01_easy` | Easy | Vicon-room baseline |
| `V1_02_medium` | Medium | more challenging Vicon-room test |
| `V2_02_medium` | Medium/harder | stress test |

Minimum first-pass odometry experiment:

```text
Same EuRoC sequence:
1. ORB-SLAM3 monocular
2. ORB-SLAM3 monocular-inertial
3. SVO monocular
4. SVO visual-inertial, if the available version supports it cleanly
5. DSO monocular
```

Metrics:

```text
ATE RMSE
RPE translation
RPE rotation
scale drift / scale error
tracking failure count
initialization success rate
runtime / FPS
CPU/GPU usage where relevant
```

### 3.2 3D reconstruction: use datasets with actual geometry ground truth

For reconstruction, visual inspection is not enough. The report needs quantitative geometry metrics.

Recommended reconstruction datasets, in priority order:

| Priority | Dataset | Ground-truth geometry type | Why use it |
|---:|---|---|---|
| 1 | ETH3D | high-precision laser-scanned ground truth / benchmark geometry | Best real-world benchmark-style reconstruction evaluation |

### 3.3 Best first choice for reconstruction: ETH3D

Use **ETH3D** as the main real-world reconstruction benchmark because it provides laser-scanned ground-truth geometry and evaluates reconstruction using accuracy, completeness, and F1/F-score.

Recommended metrics:

```text
Accuracy @ 1 cm / 2 cm / 5 cm
Completeness @ 1 cm / 2 cm / 5 cm
F-score @ 1 cm / 2 cm / 5 cm
Chamfer distance
point-to-plane RMSE, optional
runtime
GPU memory
maximum sequence length before failure
```

ETH3D is especially useful because it gives a scientific, report-friendly way to compare reconstructed point clouds or meshes against reference geometry.

---

## 4. Main benchmark design

### 4.1 Phase 1: Reproducibility table

Before doing heavy evaluation, check whether each method can be installed and run.

| Method | Installs? | Demo runs? | Dataset input supported? | Output format | Runtime notes | Problems |
|---|---:|---:|---|---|---|---|
| ORB-SLAM3 | TBD | TBD | EuRoC/TUM/KITTI | trajectory + map | TBD | TBD |
| DSO | TBD | TBD | EuRoC/TUM/KITTI conversion needed | trajectory | TBD | TBD |
| SVO | TBD | TBD | EuRoC/TUM conversion needed | trajectory | TBD | TBD |
| DA3 | TBD | TBD | image folders/video | depth/point cloud | GPU-heavy | TBD |
| DA3-Streaming | TBD | TBD | long image sequences | streamed geometry | GPU/memory-sensitive | TBD |
| VGGT/LiteVGGT | TBD | TBD | image sets | cameras/depth/point maps | batch/feed-forward | TBD |
| MapAnything | TBD | TBD | flexible | metric geometry | GPU-heavy | TBD |
| STAC-3R | TBD | TBD | streaming sequence | streaming geometry | memory focus | TBD |

This table is important because a comparative study must report reproducibility and installation difficulty, not only final accuracy.

### 4.2 Phase 2: Minimal viable benchmark

Do not benchmark every dataset immediately. Start with a small subset.

Odometry:

```text
EuRoC:
- MH_01_easy
- MH_03_medium
- V1_02_medium
```

Reconstruction:

```text
ETH3D:
- one small indoor scene
- one larger indoor scene


### 4.3 Phase 3: Expand only the promising methods

After the first benchmark, keep only the strongest methods:

```text
Top 2 odometry methods
Top 2 dense reconstruction methods
Top 1 or 2 streaming methods
```

Then test more sequences and datasets.

---

## 5. Evaluation metrics

### 5.1 Odometry metrics

For EuRoC and any trajectory benchmark:

```text
ATE RMSE
RPE translation
RPE rotation
scale drift / scale error
tracking lost count
initialization success/failure
average FPS
CPU/GPU memory
failure notes
```

Suggested tools:

```text
evo_ape
evo_rpe
evo_traj
custom CSV summarizer
```

For monocular-only methods, evaluate:

```text
Sim(3) alignment where scale ambiguity exists
```

Report clearly which alignment is used.

### 5.2 Reconstruction metrics

For point cloud or mesh reconstruction:

```text
Accuracy
Completeness
F-score
Chamfer distance
point-to-plane distance, optional
normal consistency, optional
scale error
runtime
GPU memory
online/streaming latency
maximum processed sequence length
```

Recommended thresholds:

```text
1 cm, 2 cm, 5 cm for small/indoor scenes
5 cm, 10 cm for larger/noisier scenes if needed
```

Definitions:

- **Accuracy / precision:** fraction of reconstructed points close to ground-truth geometry.
- **Completeness / recall:** fraction of ground-truth geometry covered by the reconstruction.
- **F-score:** harmonic mean of accuracy and completeness.

### 5.3 Online reconstruction metrics

For DA3-Streaming, STAC-3R, StreamVGGT, STream3R, etc.:

```text
GPU memory vs sequence length
runtime / FPS
latency per chunk/frame
geometry consistency over time
old-region consistency when revisited
maximum sequence length before failure
point cloud deformation/drift
```

This is important because streaming methods may not win the absolute geometry score, but may be much more practical for long videos.

---

## 6. Output formats to standardize

All methods should be wrapped to produce comparable outputs.

### 6.1 Trajectory output

Use a common text format:

```text
timestamp tx ty tz qx qy qz qw
```

Store under:

```text
outputs/trajectories/{method}/{dataset}/{sequence}.txt
```

### 6.2 Reconstruction output

Use:

```text
PLY point cloud
PLY mesh if available
optional NPZ for depth/camera outputs
```

Store under:

```text
outputs/reconstructions/{method}/{dataset}/{sequence}/
```

### 6.3 Run log output

Each run should generate a metadata JSON or CSV row:

```text
method
dataset
sequence
input_mode
imu_used
online_mode
chunk_size
overlap
alignment_type
runtime_sec
avg_fps
peak_gpu_mem_gb
success
failure_reason
notes
```

---

## 7. Suggested master repo structure

```text
monocular-3d-reconstruction-benchmark/
│
├── README.md
├── environment/
│   ├── docker/
│   ├── conda/
│   └── install_notes/
│
├── datasets/
│   ├── euroc/        # IMU vs no-IMU odometry
│   ├── eth3d/        # reconstruction GT geometry
│   ├── redwood/      # indoor laser-scan GT validation
│   ├── tum_rgbd/     # RGB-D fused reference sanity benchmark
│   └── tartanair/    # synthetic full-GT stress tests
│
├── methods/
│   ├── orb_slam3/
│   ├── dso/
│   ├── svo/
│   ├── depth_anything_3/
│   ├── da3_streaming/
│   ├── litevggt/
│   ├── map_anything/
│   ├── stac_3r/
│
│
├── wrappers/
│   ├── run_orb_slam3.py
│   ├── run_dso.py
│   ├── run_svo.py
│   ├── run_da3.py
│   ├── run_da3_streaming.py
│   ├── run_vggt.py
│   ├── run_map_anything.py
│   └── run_stac.py
│
├── evaluation/
│   ├── odometry/
│   │   ├── run_evo_ape.py
│   │   ├── run_evo_rpe.py
│   │   └── summarize_trajectory_metrics.py
│   │
│   ├── reconstruction/
│   │   ├── align_reconstruction_to_gt.py
│   │   ├── compute_accuracy_completeness_fscore.py
│   │   ├── compute_chamfer.py
│   │   └── summarize_reconstruction_metrics.py
│   │
│   └── runtime/
│       ├── collect_gpu_memory.py
│       └── summarize_runtime.py
│
├── outputs/
│   ├── trajectories/
│   ├── reconstructions/
│   ├── logs/
│   ├── metrics/
│   └── screenshots_for_qualitative_appendix/
│
├── reports/
│   ├── weekly_notes/
│   ├── supervisor_summary/
│   ├── benchmark_tables/
│   └── final_comparative_report/
│
└── scripts/
    ├── download_datasets.sh
    ├── prepare_euroc.py
    ├── prepare_eth3d.py
    ├── convert_trajectory_format.py
    └── make_summary_tables.py
```

---

## 8. CSV schema for benchmark results

### 8.1 Odometry CSV

```csv
method,dataset,sequence,input_mode,imu_used,alignment,ate_rmse,rpe_trans,rpe_rot,scale_error,tracking_lost_count,init_success,fps,peak_gpu_mem_gb,runtime_sec,notes
ORB-SLAM3,EuRoC,MH_01_easy,mono,false,Sim3,,,,,,,,,,
ORB-SLAM3,EuRoC,MH_01_easy,mono-inertial,true,SE3,,,,,,,,,,
SVO,EuRoC,MH_01_easy,mono,false,Sim3,,,,,,,,,,
DSO,EuRoC,MH_01_easy,mono,false,Sim3,,,,,,,,,,
```

### 8.2 Reconstruction CSV

```csv
method,dataset,sequence,input_mode,online_mode,chunk_size,overlap,alignment,acc_1cm,comp_1cm,fscore_1cm,acc_2cm,comp_2cm,fscore_2cm,acc_5cm,comp_5cm,fscore_5cm,chamfer,scale_error,fps,peak_gpu_mem_gb,runtime_sec,notes
DA3,ETH3D,scene_01,rgb,false,,,,,,,,,,,,,,,,,
DA3-Streaming,ETH3D,scene_01,rgb,true,,,,,,,,,,,,,,,,,
VGGT,ETH3D,scene_01,rgb,false,,,,,,,,,,,,,,,,,
MapAnything,ETH3D,scene_01,rgb,false,,,,,,,,,,,,,,,,,
STAC-3R,ETH3D,scene_01,rgb,true,,,,,,,,,,,,,,,,,
```

---

## 9. Specific experiments to run first

### Experiment A: IMU usefulness on EuRoC

Goal:

> Determine whether IMU improves monocular odometry enough to include it in the final reconstruction pipeline.

Procedure:

```text
For each EuRoC sequence:
1. Run ORB-SLAM3 mono.
2. Run ORB-SLAM3 mono-inertial.
3. Run SVO mono.
4. Run SVO + IMU if supported.
5. Run DSO mono.
6. Evaluate ATE/RPE using evo.
7. Record initialization failures and runtime.
```

Expected conclusion style:

```text
IMU is useful if it significantly reduces scale drift, improves tracking robustness, and decreases failures on medium/hard sequences without unacceptable setup/runtime cost.
```

### Experiment B: Reconstruction benchmark on ETH3D

Goal:

> Quantitatively compare dense reconstruction methods against ground-truth geometry.

Procedure:

```text
For each ETH3D scene:
1. Run DA3.
2. Run DA3-Streaming.
3. Run VGGT/LiteVGGT.
4. Run MapAnything.
5. Run STAC-3R / streaming method if installable.
6. Export point cloud or mesh to PLY.
7. Align reconstruction to GT if needed.
8. Compute accuracy, completeness, F-score, Chamfer distance.
9. Record runtime and memory.
```

Important:

- If a method predicts metric scale, evaluate both raw metric output and optionally aligned output.
- If a method is scale-ambiguous, use Sim(3) alignment but clearly report that this was done.
- Do not compare visually only.

### Experiment D: Streaming/chunk consistency test

Goal:

> Test whether streaming methods solve the DA3 chunk consistency problem.

Compare:

```text
1. DA3 independent chunks
2. DA3 independent chunks + overlap Sim(3) merging
3. DA3-Streaming
4. VGGT/LiteVGGT batch or chunked
5. MapAnything
6. STAC-3R / StreamVGGT / STream3R if available
```

Metrics:

```text
F-score vs GT geometry
memory vs sequence length
runtime / FPS
old-region consistency when revisited
maximum sequence length before failure
```

---

## 10. Expected thesis/report narrative

The final report should not simply say which method is best. A stronger narrative is:

> Classical SLAM and VIO methods provide stable camera tracking and online performance, but usually produce sparse or semi-dense maps. Modern feed-forward 3D foundation models provide richer dense geometry, but long-sequence consistency, scale, memory, and streaming behavior are still challenging. Streaming methods such as DA3-Streaming and STAC-3R attempt to solve this by carrying state or compressing long-term context. The comparative study evaluates which family is most suitable for monocular online 3D reconstruction, and whether a hybrid pipeline is better than any single method.

Likely final direction:

```text
ORB-SLAM3 / SVO / VIO for stable pose tracking
+
DA3 / VGGT / MapAnything / STAC-style model for dense geometry
+
metric/global fusion layer for consistent reconstruction
```

The likely answer may not be a single method. It may be a hybrid pipeline.

---

## 11. Immediate next steps

1. Create the master repo with the structure above.
2. Add dataset download/preparation scripts.
3. Install and test ORB-SLAM3 on EuRoC mono and mono-inertial.
4. Install and test DSO/SVO on at least one EuRoC sequence.
5. Build a common trajectory output converter.
6. Run `evo_ape` and `evo_rpe` on the first EuRoC sequence.
7. repeat and complete the tests for all odometry methods with all data sets
8. move on to installing reconstruction methods, then testing them one by one with data sets
9. Export all reconstructions as PLY.
10. Implement point-cloud comparison metrics: accuracy, completeness, F-score, Chamfer.

---

## 12. Source notes

These sources were used to justify dataset/method choices:

1. EuRoC MAV official page: https://projects.asl.ethz.ch/datasets/euroc-mav/  
   Notes: stereo images, synchronized IMU measurements, accurate motion/structure ground truth.

2. ETH3D overview: https://www.eth3d.net/overview  
   Notes: multi-view stereo / 3D reconstruction benchmark; ground-truth geometry from high-precision laser scanner.

3. ETH3D low-res many-view metrics page: https://www.eth3d.net/low_res_many_view?metric=completeness  
   Notes: accuracy, completeness, F1 score, runtime.

8. Depth Anything 3 official GitHub: https://github.com/bytedance-seed/depth-anything-3  
   Notes: predicts spatially consistent geometry from arbitrary visual inputs, with or without known camera poses.

9. DA3-Streaming README: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md  
   Notes: streaming pipeline for long video sequences and large-scale scenes under memory budgets.

10. VGGT official GitHub: https://github.com/facebookresearch/vggt  
    Notes: feed-forward network predicting camera parameters, point maps, depth maps, and point tracks.

11. MapAnything official GitHub: https://github.com/facebookresearch/map-anything  
    Notes: universal feed-forward metric 3D reconstruction framework supporting many input/task configurations.

12. STAC official GitHub: https://github.com/Rainzor/STAC  
    Notes: plug-and-play KV-cache compression framework for memory-efficient streaming 3D reconstruction over long videos.

13. STAC paper/arXiv: https://arxiv.org/abs/2603.20284  
    Notes: spatio-temporal cache compression for streaming 3D reconstruction; memory and runtime improvements reported by authors.

---

## 13. Main caution points

- Do not compare all methods as if they are the same type of system.
- Do not use visual inspection as the main reconstruction evaluation.
- Do not rely only on Sim(3) chunk merging for DA3-style reconstruction; the geometry itself may change with context.
- Always report whether evaluation used raw metric scale, SE(3) alignment, or Sim(3) alignment.
- Always record runtime and memory, especially for streaming/foundation models.
- Start small: EuRoC + TUM sanity + one ETH3D scene. Expand only after wrappers and metrics work.
