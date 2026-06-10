# An Empirical Comparison of SVO and DA3 for Monocular Trajectory Estimation

## Abstract
This report summarizes a practical finding from the `mono3d-benchmark` reconstruction workflow: when the input is truly monocular RGB, `Depth Anything 3` (DA3) produces substantially better trajectory estimates than the current `SVO Pro Open` monocular configuration on the selected TUM RGB-D sequences, while `SVO` remains practical and competitive when inertial measurements are available, as shown by earlier EuRoC `mono` versus `mono+imu` experiments. Across the 11 EuRoC sequences already benchmarked in this repository, the mean `Sim3` trajectory error decreases from `0.898 m` in monocular SVO to `0.125 m` in `mono+imu` SVO. In contrast, on the latest TUM comparison, the mean `Sim3` trajectory error over the 5 directly comparable sequences is `0.481 m` for SVO and `0.034 m` for the best DA3 variant per sequence, a roughly `14.1x` reduction in favor of DA3. These results support a hybrid interpretation: DA3 is the stronger standalone choice for monocular RGB trajectory estimation in this setup, whereas SVO remains valuable as a geometric anchor, an interpretable motion estimator, a cross-validation signal, and a tool for domain-shift analysis.

## Keywords
Monocular visual odometry, visual-inertial odometry, learned depth priors, trajectory evaluation, SVO, Depth Anything 3, TUM RGB-D, EuRoC

## I. Introduction
The working question behind this study was whether `SVO Pro Open` should remain a core trajectory source in the current monocular 3D reconstruction pipeline once `Depth Anything 3` streaming is available and functioning well. The answer matters because trajectory quality directly influences downstream dense fusion, chunk alignment, and reconstruction stability.

Two observations motivated the study:

1. `SVO` had already been restored and validated on EuRoC, including a working `mono+imu` path.
2. `DA3` was producing both useful dense reconstructions and strong trajectory estimates on TUM-prepared runs.

The core discovery is not that SVO is broadly ineffective, but that the value of SVO depends strongly on the sensing regime. In a purely monocular RGB setting, DA3 benefits from strong learned depth priors. In a visual-inertial setting, SVO becomes much more competitive and practically useful.

## II. Experimental Setup

### A. Classical Geometry Baseline: SVO
The restored SVO benchmark path in this repository uses `SVO Pro Open` in three relevant modes:

- `mono` on EuRoC
- `mono+imu` on EuRoC
- `mono` on TUM RGB-D, treating TUM as an RGB-only dataset for the benchmark runner

For the TUM runs, the benchmark configuration remained monocular:

- `use_imu: false`
- `use_ceres_backend: false`
- `runlc: false`

The latest TUM rerun used the official TUM RGB calibration profile written into `calib.yaml` for the RGB camera. The benchmark path does not consume TUM depth maps as input, so these runs remain monocular in the SVO sense.

### B. Learned Dense Baseline: DA3
The DA3 side used the repaired `DA3-LARGE` configuration with matching Hugging Face model files and TUM evaluation scripts already integrated into this repository. For each TUM sequence, DA3 had multiple trajectory candidates produced by:

- `fps ∈ {2, 3, 5}`
- `chunk:overlap ∈ {10:5, 20:10, 30:15}`

For the direct SVO-vs-DA3 report, the best DA3 variant per sequence was selected according to the lowest `Sim3 RMSE` against TUM ground truth.

### C. Datasets
Two benchmark families were used:

- `EuRoC MAV` for assessing the value of IMU integration with SVO
- `TUM RGB-D` for comparing pure monocular SVO against DA3 on indoor RGB sequences

The TUM sequences included:

- `freiburg1_desk`
- `freiburg1_room`
- `freiburg1_xyz`
- `freiburg3_cabinet`
- `freiburg3_large_cabinet`
- `freiburg3_teddy`

### D. Metrics
The report relies primarily on:

- `tracking_ratio`
- `trajectory_ratio`
- `SE3 RMSE`
- `Sim3 RMSE`
- `path_ratio_est_over_gt`

`Sim3 RMSE` is emphasized because it isolates trajectory-shape fidelity after similarity alignment and therefore captures whether a monocular trajectory is structurally correct even when raw scale is imperfect.

## III. EuRoC Evidence: SVO Becomes Practical with IMU
The first important result is that SVO should not be judged only by its monocular performance. On EuRoC, the `mono+imu` configuration is markedly better than monocular SVO.

From [reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/summary.csv](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/summary.csv):

- `11` EuRoC sequences were compared in both modes
- mean `Sim3 RMSE`:
  - `mono`: `0.898 m`
  - `mono+imu`: `0.125 m`
- median `Sim3 RMSE`:
  - `mono`: `0.74 m`
  - `mono+imu`: `0.12 m`
- mean `tracking_ratio`:
  - `mono`: `0.92`
  - `mono+imu`: `0.95`
- mean `trajectory_ratio`:
  - `mono`: `0.854`
  - `mono+imu`: `0.915`

This is consistent with the intended use of SVO as a visual-inertial system: the IMU reduces scale ambiguity, improves robustness during weak-parallax motion, and stabilizes the trajectory under conditions that are difficult for monocular feature-and-photometric tracking alone.

Relevant EuRoC plots are available in:

- [reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601)
- [sim3_rmse.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/sim3_rmse.png)
- [tracking_ratio.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/tracking_ratio.png)
- [trajectory_ratio.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/trajectory_ratio.png)

## IV. TUM Evidence: DA3 Strongly Outperforms Monocular SVO
The second major result comes from the TUM comparison between the latest monocular SVO run and the best DA3 trajectory variant per sequence.

The comparison summary is available at:

- [reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/summary.csv](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/summary.csv)

And the aggregate plots are here:

- [sim3_rmse.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/sim3_rmse.png)
- [se3_rmse.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/se3_rmse.png)
- [path_ratio.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/path_ratio.png)

### A. Sequence-Level Results
For the latest comparison:

- `freiburg1_desk`
  - SVO `Sim3 RMSE`: `0.4873 m`
  - best DA3 `Sim3 RMSE`: `0.0203 m`
- `freiburg1_room`
  - SVO `Sim3 RMSE`: `0.9374 m`
  - best DA3 `Sim3 RMSE`: `0.0832 m`
- `freiburg1_xyz`
  - SVO `Sim3 RMSE`: `0.1164 m`
  - best DA3 `Sim3 RMSE`: `0.0107 m`
- `freiburg3_cabinet`
  - SVO status: `empty_estimate`
  - best DA3 `Sim3 RMSE`: `0.0147 m`
- `freiburg3_large_cabinet`
  - SVO `Sim3 RMSE`: `0.4094 m`
  - best DA3 `Sim3 RMSE`: `0.0303 m`
- `freiburg3_teddy`
  - SVO `Sim3 RMSE`: `0.4566 m`
  - best DA3 `Sim3 RMSE`: `0.0264 m`

### B. Aggregate Result
On the `5` sequences with directly comparable non-empty SVO trajectories:

- mean SVO `Sim3 RMSE`: `0.481 m`
- mean DA3 `Sim3 RMSE`: `0.034 m`

This corresponds to an approximately `14.1x` lower mean `Sim3` error for DA3 in this comparison.

### C. Best DA3 Variants Chosen
The best DA3 runs selected by the comparison script were:

- `freiburg1_desk__fps2__chunk30__overlap15__loop`
- `freiburg1_room__fps3__chunk30__overlap15__loop`
- `freiburg1_xyz__fps2__chunk30__overlap15__loop`
- `freiburg3_cabinet__fps2__chunk30__overlap15__loop`
- `freiburg3_large_cabinet__fps2__chunk30__overlap15__loop`
- `freiburg3_teddy__fps3__chunk30__overlap15__loop`

### D. Non-IMU SVO Tuning Attempts
A follow-up non-IMU tuning sweep was also attempted to test whether monocular SVO could be materially improved without adding IMU information.

The practical outcome was negative:

- the refreshed baseline rerun improved modestly, reducing mean SVO `Sim3 RMSE` from `0.4814 m` to `0.4285 m`
- enabling the Ceres backend was not a valid non-IMU path in this SVO build, because the backend aborts when `use_imu=false`
- enabling loop closure in the TUM monocular path caused hard crashes and empty estimates
- a stricter `conservative` monocular configuration also collapsed to empty estimates on all six tested TUM sequences

So the current evidence does not just say that DA3 beats one weak monocular SVO setting. It says that, within the practically reachable non-IMU SVO variants tested here, no stronger replacement baseline was recovered.

## V. Why DA3 Wins in the Pure Monocular RGB Setting
The practical reason for DA3’s superiority here is that DA3 is not solving the same inference problem as classical monocular SVO.

### A. Learned Depth Priors
DA3 benefits from strong learned depth priors. In this context, a depth prior is a model’s built-in expectation about likely scene depth and structure learned from large-scale training data. This gives DA3 a dense geometric hypothesis even when parallax is weak, features are sparse, or texture is repetitive.

By contrast, monocular SVO must infer depth from image motion and probabilistic triangulation over time. In low-parallax or ambiguous indoor segments, this is fundamentally harder.

### B. TUM Is Unfavorable to Plain Monocular SVO in This Configuration
The TUM benchmark path used here is RGB-only from SVO’s perspective. The current benchmark integration does not consume TUM depth maps directly, and the configuration disables IMU, loop closure, and Ceres backend refinement. As a result, SVO is operating under:

- monocular scale ambiguity
- higher dependence on stable feature correspondence
- greater sensitivity to repeated textures and weak motion baselines

DA3, on the other hand, retains dense learned structure even under those conditions.

### C. Caveat: The DA3 Side Is Best-of-N
One methodological caveat is that the DA3 comparison used the best trajectory variant per sequence from nine evaluated settings, whereas SVO used one shared TUM monocular configuration. Therefore, the result should be interpreted as:

`best practical DA3 trajectory found in this workflow` versus `single current SVO monocular baseline`

This does not weaken the central finding, but it does mean the comparison emphasizes deployable performance rather than equal hyperparameter budgets.

## VI. Implications for Pipeline Design
The results suggest that the role of SVO in the current project should be reframed.

### A. When DA3 Should Be Preferred
If the immediate objective is:

- best monocular RGB trajectory on indoor benchmark data
- strong dense geometry without external IMU
- robust trajectory recovery under weak parallax

then DA3 is the stronger standalone choice in this repository’s current setup.

### B. When SVO Still Adds Value
Despite this, SVO remains useful in at least four roles.

#### 1) Geometric Anchor
SVO provides an explicit geometry-based trajectory estimate that can serve as an anchor or consistency signal for dense learned reconstructions, especially in environments where classical VO remains stable.

#### 2) Interpretability
SVO is easier to diagnose. Failures can be linked to missing parallax, lost features, initialization instability, relocalization behavior, or scale drift. This makes it valuable for engineering and debugging even when it is not the strongest standalone trajectory source.

#### 3) Cross-Validation
Agreement between SVO and DA3 increases confidence. Disagreement highlights failure cases and can trigger further inspection or fallback logic.

#### 4) Generalization and Domain Shift Analysis
Learned systems often excel in-distribution but may degrade in less predictable ways under domain shift. Classical geometric VO, although weaker on average in some settings, provides an alternative behavior profile that can be useful for robustness studies and for understanding where learned priors help or hurt.

## VII. Conclusion
The experiments support the following conclusion:

1. `SVO is practical and effective when IMU is available.`
   The EuRoC `mono+imu` results are substantially better than monocular SVO and show that SVO remains a relevant visual-inertial baseline.

2. `DA3 is clearly stronger than plain monocular SVO for trajectory estimation when the input is truly monocular RGB in the tested TUM setup.`
   On the latest comparison, DA3 achieved approximately `14.1x` lower mean `Sim3 RMSE` than SVO over the directly comparable sequences, and it succeeded on a sequence where SVO returned an empty estimate.

3. `SVO should not be discarded entirely, but its role should shift.`
   In the current pipeline, SVO is best justified as:
   - a geometric anchor
   - an interpretable motion estimator
   - a cross-validation signal
   - a tool for generalization and domain-shift analysis

Accordingly, the current evidence does not support using plain monocular SVO as the primary trajectory estimator when DA3 is available and the sensing regime is monocular RGB only. It does, however, support retaining SVO as a complementary geometric component and as a stronger contender in visual-inertial scenarios.

## References
[1] C. Forster, M. Pizzoli, and D. Scaramuzza, “SVO: Semidirect Visual Odometry for Monocular and Multicamera Systems,” *IEEE Transactions on Robotics*, 2017.

[2] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, “Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data,” *arXiv preprint arXiv:2401.10891*, 2024. Available: https://arxiv.org/abs/2401.10891

[3] Technical University of Munich, “TUM RGB-D Dataset File Formats and Intrinsic Camera Calibration of the Kinect.” Available: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect

## Appendix: Key Local Artifacts
EuRoC SVO mono vs mono+imu:

- [reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/summary.csv](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/summary.csv)
- [reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/sim3_rmse.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_full11_euroc_mono_vs_mono_imu_20260601/sim3_rmse.png)

Latest TUM SVO-only report:

- [reports/evaluation/svo_tum_mono_20260610_080434_mono3d_tum_mono/summary.csv](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/svo_tum_mono_20260610_080434_mono3d_tum_mono/summary.csv)

Latest TUM SVO vs DA3 report:

- [reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/summary.csv](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/summary.csv)
- [reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/sim3_rmse.png](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/tum_svo_vs_da3_20260610_080434_mono3d_tum_mono/sim3_rmse.png)
