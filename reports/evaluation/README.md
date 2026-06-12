# Evaluation Index

This directory is for evaluated summaries and plots, not raw runtime outputs.

Canonical current folders kept at top level:

- `svo_full11_euroc_mono_vs_mono_imu_20260601`
- `svo_full11_euroc_mono_vs_mono_imu_gpu_20260601`
- `euroc_svo_vs_da3_20260610T132107Z_fps5_da3_streaming`
- `svo_tum_baseline_mono3d_tum_mono_20260610_091041_mono3d_tum_mono`
- `tum_svo_vs_da3_baseline_20260610_091041_mono3d_tum_mono`
- `tum_svo_variants_summary_20260610_091759`
- `qcar_svo_vs_gt_20260612_114149_mono3d_qcar_evry_mono`
- `qcar_da3_vs_gt_20260612_114149_mono3d_qcar_evry_mono`
- `qcar_svo_vs_da3_20260612_114149_mono3d_qcar_evry_mono`

Archived exploratory or superseded folders live under `archive/`.

TUM note:

- only the baseline TUM folders are intended to remain at top level
- older reruns and non-canonical variant folders belong under `archive/tum/`
- if they are stuck at top level with `nobody:nogroup` ownership, run:

```bash
sudo chown -R qcar:qcar reports/evaluation
bash scripts/fix_reports_evaluation_layout.sh
```

Supporting docs:

- `SVO_EuRoC_Runbook.md`
- `DA3_Streaming_Runbook.md`
- `euroc_index.md`
- `tum_index.md`
- `qcar_index.md`
