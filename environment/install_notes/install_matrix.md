# Install Matrix

| Item | Type | Source | Environment Strategy | Status | Notes |
|---|---|---|---|---|---|
| EuRoC MAV | dataset | official ETH ASL page | manual download into `datasets/euroc` | first-pass-ready | grouped `machine_hall.zip`, `vicon_room1.zip`, and `vicon_room2.zip` are present locally; first-pass extracted sequences are `MH_01_easy`, `MH_03_medium`, and `V1_02_medium` |
| ETH3D | dataset | official ETH3D page | direct download into `datasets/eth3d` | first-pass-ready | `courtyard`, `delivery_area`, and `electro` archives are present locally and extracted into scene folders |
| SVO Pro Open | method | `uzh-rpg/rpg_svo_pro_open` | Docker or ROS workspace | active-next | now the only active odometry method priority; workspace helper scripts added for the Noetic container flow |
| Depth Anything 3 | method | `ByteDance-Seed/Depth-Anything-3` | dedicated Python `venv` | pending | likely GPU-heavy |
| LiteVGGT | method | `facebookresearch/vggt` or lighter official checkpoint in same codebase | dedicated Python `venv` | installed-minimal | minimal base environment created in `environment/venv/.venvs/litevggt`; demo and COLMAP extras intentionally skipped for now to keep storage and dependency load down |
| MapAnything | method | `facebookresearch/map-anything` | dedicated Python `venv` | pending | flexible metric geometry baseline |
| STAC-3R | method | `Rainzor/STAC` | dedicated Python `venv` | deferred | official project page is `stac-3r.github.io`; upstream notes say it was tested on RTX 3090 24 GB and A100 40 GB, so treat it as a later-stage or reduced-setting experiment on this 12 GB GPU |
