# Docker Environments

These Dockerfiles are meant to isolate fragile method dependencies from the host system.

## Intended Usage

- `orb_slam3.Dockerfile`: baseline container for ORB-SLAM3 builds
- `dso.Dockerfile`: legacy build environment for DSO
- `svo.Dockerfile`: ROS-oriented base for SVO

## Notes

- Building and running these images will require approval outside the sandbox.
- The Dockerfiles are intentionally conservative and can be refined once the upstream repositories are cloned locally.
- Python-heavy feed-forward methods are better started in `venv` before deciding whether a container is necessary.
