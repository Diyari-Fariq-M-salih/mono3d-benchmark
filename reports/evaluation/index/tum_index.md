# TUM Index

Canonical folders:

- `svo_tum_baseline_mono3d_tum_mono_20260610_091041_mono3d_tum_mono`
  - Main usable monocular SVO baseline.
- `tum_svo_vs_da3_baseline_20260610_091041_mono3d_tum_mono`
  - Main direct TUM SVO vs DA3 comparison.
- `tum_svo_variants_summary_20260610_091759`
  - Summary across the additional non-IMU SVO variants.

Archived exploratory or superseded TUM folders:

- early reruns
- host-fix reruns
- loop / conservative / ceres variant folders

Those now live under `archive/tum/` when the filesystem ownership allowed moving them.

Note:

- several older TUM evaluation folders were created under `nobody:nogroup`
- those remain temporarily at the top level until they can be moved with a higher-privilege filesystem pass
- they should still be treated as historical rather than canonical

Cleanup:

```bash
sudo chown -R qcar:qcar reports/evaluation
bash scripts/archive/diagnostics/fix_reports_evaluation_layout.sh
```
