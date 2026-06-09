# Install Matrix

| Item | Type | Source | Environment Strategy | Status | Notes |
|---|---|---|---|---|---|
| EuRoC MAV | dataset | official ETH ASL page | manual download into `datasets/euroc` | restore-needed | backup repo does not contain the local archives or extracted sequences, so the dataset must be restored before rerunning SVO |
| ETH3D | dataset | official ETH3D page | direct download into `datasets/eth3d` | deferred | kept out of the active rebuild until the `SVO + DA3` path is stable |
| SVO Pro Open | method | `uzh-rpg/rpg_svo_pro_open` | Docker or ROS workspace | restore-needed | only active odometry method path; repo automation is present, but the local upstream clone and ROS workspace are missing |
| Depth Anything 3 | method | `ByteDance-Seed/Depth-Anything-3` | dedicated Python `venv` | restore-needed | primary reconstruction method path; upstream repo and local environment need to be rebuilt |
| DA3-Streaming | method path inside `Depth-Anything-3` | same `Depth Anything 3` `venv` | restore-needed | main dense mapping direction for the project; official script does not directly accept external `SVO` poses, so local integration work is expected |
