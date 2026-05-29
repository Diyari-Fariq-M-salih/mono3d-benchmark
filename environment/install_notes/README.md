# Installation Notes

This folder tracks how each dataset and method should be installed in a reproducible way.

## Principles

- Use one clean setup path per method.
- Prefer `venv` for Python-heavy methods.
- Prefer Docker for legacy C++ or dependency-fragile methods when host installation becomes painful.
- Keep original repositories under `methods/`.
- Keep datasets under `datasets/`.
- Record every install decision before benchmarking.
- Install the minimal working dependency set first, then add optional extras only if benchmark features require them.
- Avoid duplicate clones, duplicate datasets, and avoid installing full optional bundles unless they are needed.

## Current Environment Constraints

- `python3` is available
- `venv` is available
- `git` is available
- `conda` is not installed on this machine
- `docker` exists, but needs execution outside the sandbox to run normally
- hardware from `reports/pc_specs.txt`: about `31 GiB` RAM and an `RTX 4070` with about `12 GiB` VRAM

## Hardware-Aware Decisions

- Prefer `LiteVGGT` over full `VGGT` for the initial benchmark setup.
- Keep batch size, frame count, and checkpoint size conservative for feed-forward 3D models.
- Treat `MapAnything` and `Depth Anything 3` as installable, but expect reduced inference settings on this GPU.
- Prefer a shared GPU Python base stack for heavy methods when dependency compatibility allows it, to avoid downloading and storing separate `torch` and CUDA wheels for every method venv.

## Planned Installation Order

1. Benchmark utility venv
2. EuRoC dataset
3. ETH3D dataset
4. ORB-SLAM3
5. DSO
6. SVO Pro Open
7. Depth Anything 3
8. LiteVGGT
9. MapAnything
10. STAC-3R or nearest available streaming baseline

## Dataset Download Policy

- Use `scripts/download_datasets.py` with named presets from `datasets/download_manifest.json`.
- For manual and repo-style download flows, use the explicit shell scripts in `scripts_downloading/`.
- For now, prefer the first-pass preset rather than downloading every possible benchmark split immediately.
- EuRoC archives can be extracted directly after download.
- ETH3D uses `.7z` archives and this machine now has `7z` available, so downloaded archives can be normalized into extracted scene folders immediately.
- EuRoC is currently documented as a manual-download step in `scripts_downloading/euroc_manual_download.md`.
- Current local state:
- `datasets/euroc/` contains grouped `machine_hall.zip`, `vicon_room1.zip`, and `vicon_room2.zip` archives plus first-pass extracted sequences `MH_01_easy`, `MH_03_medium`, and `V1_02_medium`.
- `datasets/eth3d/` contains `courtyard`, `delivery_area`, and `electro` archives and extracted scene folders.

## Deferred Items

- `STAC-3R` is cloned and documented, but it should be treated as a deferred install on this machine unless we intentionally budget time for reduced-setting experiments.
- Reason: upstream targets heavier GPUs, requires an additional backbone, and includes optional CUDA extensions plus a comparatively heavy dependency stack.

## Tracking

Use `environment/install_notes/install_matrix.md` to log status and issues for each item.
