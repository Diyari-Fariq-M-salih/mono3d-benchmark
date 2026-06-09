# DA3-Streaming Integration Notes

This folder is reserved for local integration code around `Depth Anything 3` streaming experiments.

Current project direction:

- use `SVO Pro Open` for lightweight visual odometry
- use `Depth Anything 3` and `DA3-Streaming` for dense monocular mapping
- compare DA3's internal pose estimates against `SVO`
- later, if needed, prototype a custom pose-conditioned path that injects `SVO` poses into the DA3 pipeline

Important implementation note:

- the official `DA3-Streaming` script does not expose an input flag for external `SVO` poses
- if we want `SVO`-guided streaming, we will likely need a local wrapper or a small upstream patch

Repo-local entry points:

- `bash scripts/setup_da3.sh`
- `bash scripts/install_da3_into_conda.sh da3stream`
- `bash scripts/download_da3_weights.sh`
- `bash scripts/run_da3_streaming.sh --image-dir /path/to/images`
- [reports/evaluation/DA3_Streaming_Runbook.md](/home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark/reports/evaluation/DA3_Streaming_Runbook.md)
