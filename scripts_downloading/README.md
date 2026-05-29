# Dataset Download Scripts

These scripts follow a simple, explicit pattern similar to other SLAM repos:

- one shell script per dataset or preset
- visible `wget` commands
- minimal logic
- reproducible target folders

Current scripts:

- `download_eth3d_first_pass.sh`
- `euroc_manual_download.md`

Notes:

- `ETH3D` downloads are known reachable from this machine.
- `EuRoC` is documented as a manual-download workflow because the official host has recently been unreliable from this machine.
- Current local state:
- `ETH3D` first-pass archives for `courtyard`, `delivery_area`, and `electro` are already present in `datasets/eth3d/`, and those three scenes are extracted.
- `EuRoC` grouped archives are already present in `datasets/euroc/`, and the first-pass extracted sequences are `MH_01_easy`, `MH_03_medium`, and `V1_02_medium`.
- The prep validators now live in `scripts/prepare_euroc.py` and `scripts/prepare_eth3d.py`.
