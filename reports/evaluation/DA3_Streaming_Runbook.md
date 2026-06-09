# DA3-Streaming Runbook

This document records the local `Depth Anything 3` and `DA3-Streaming` workflow for this repo.

## 1. Active Scope

- use `SVO` for lightweight visual odometry
- use `Depth Anything 3` and `DA3-Streaming` for dense reconstruction
- compare DA3 internal poses with `SVO`
- later, if needed, prototype `SVO` pose injection as a local modification

## 2. Local Setup

From the repo root:

```bash
cd /home/qcar/Documents/Diyari_M_salih_2026/mono3d-benchmark
bash scripts/setup_da3.sh
```

This creates:

```text
environment/venv/.venvs/depth_anything_3/
```

If this machine already has a suitable conda env with the heavy GPU stack, a lighter path is:

```bash
bash scripts/install_da3_into_conda.sh da3stream
```

Then run DA3 with:

```bash
export DA3_CONDA_ENV=da3stream
```

## 3. Download Weights

```bash
bash scripts/download_da3_weights.sh
```

The upstream downloader currently fetches:

- `dino_salad.ckpt`
- `config.json`
- `model.safetensors`

into:

```text
methods/depth_anything_3/upstream/da3_streaming/weights/
```

## 4. Recommended First Run

Use a directory of extracted images:

```bash
bash scripts/run_da3_streaming.sh \
  --image-dir /path/to/images \
  --config methods/depth_anything_3/upstream/da3_streaming/configs/base_config.yaml
```

If you want an explicit output location:

```bash
bash scripts/run_da3_streaming.sh \
  --image-dir /path/to/images \
  --output-dir outputs/reconstructions/depth_anything_3/my_test
```

## 5. Important Output Files

DA3-Streaming writes files such as:

- `camera_poses.txt`
- `intrinsic.txt`
- `pcd/combined_pcd.ply`
- `results_output/` when depth/conf export is enabled

## 6. Integration Note

The official `DA3-Streaming` script does not directly accept external `SVO` poses as an input argument.

Current practical workflow:

1. run `SVO`
2. run `DA3-Streaming` on the same frame set
3. compare trajectories
4. align DA3 outputs to `SVO` afterward
5. only then decide whether a custom pose-conditioned integration is worth the added complexity
