# Docker Environments

These Dockerfiles are meant to isolate fragile method dependencies from the host system.

## Intended Usage

- `svo.Dockerfile`: ROS-oriented base for SVO

## Notes

- Building and running these images will require approval outside the sandbox.
- The Dockerfiles are intentionally conservative and can be refined once the upstream repositories are cloned locally.
- Python-heavy feed-forward methods are better started in `venv` before deciding whether a container is necessary.
- The current images are meant to build from the already-cloned local repositories under `methods/.../upstream`.

## Suggested Build Commands

Build the image from the repo root:

```bash
docker build -f environment/docker/svo.Dockerfile -t mono3d-benchmark/svo:noetic .
```

Open an interactive shell:

```bash
docker run --rm -it -v "$PWD":/workspace mono3d-benchmark/svo:noetic bash
```

## Method-Specific Build Notes

- `SVO Pro Open`: build in a catkin workspace on ROS Noetic. Use the HTTPS dependency file at `methods/svo/dependencies_https.yaml` instead of the upstream SSH-only manifest.

## SVO Flow

Enter the ready-made SVO container:

```bash
bash scripts/enter_svo_container.sh
```

Then inside the container:

```bash
bash scripts/setup_svo_workspace.sh
bash scripts/build_svo.sh
```

This uses the local workspace at `environment/ros_ws/svo/`, symlinks the already-cloned
`methods/svo/upstream` repo into `src/`, imports the remaining ROS dependencies with
the HTTPS manifest, disables `minkindr_python`, downloads the loop-closure vocabulary,
and runs a normal `catkin build` without the optional global-map path.
