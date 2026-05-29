# EuRoC Manual Download

`EuRoC` is currently best handled as a manual download on this machine.

## Data Preparation

[EuRoC Dataset](https://projects.asl.ethz.ch/datasets/)
contains stereo images, synchronized IMU measurements, and accurate motion and
structure ground truth. In XRSLAM, only monocular camera and IMU data are used.

Recommended folder structure:

```text
datasets
└── euroc
    ├── MH_01_easy
    │   └── mav0
    │       └── ...
    ├── MH_02_easy
    │   └── mav0
    └── ...
```
