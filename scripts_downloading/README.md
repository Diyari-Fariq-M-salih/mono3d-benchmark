# Dataset Download Scripts

These scripts follow a simple, explicit pattern similar to other SLAM repos:

- one shell script per dataset or preset
- visible `wget` commands
- minimal logic
- reproducible target folders

Current scripts:

- `download_eth3d_first_pass.sh`
- `euroc_manual_download.md`
- `scripts/download_datasets.py euroc_full --skip-existing`

Notes:

- `ETH3D` downloads are known reachable from this machine.
- `EuRoC` can be restored either from the manual workflow or through the grouped archive entries in `datasets/download_manifest.json`.
- Current rebuild state:
- `EuRoC` should be restored first for the active `SVO` workflow.
- `ETH3D` can wait until the `Depth Anything 3` path is ready for reconstruction evaluation.
- The prep validators now live in `scripts/prepare_euroc.py` and `scripts/prepare_eth3d.py`.
