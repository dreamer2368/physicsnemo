# Lid-Driven Cavity — ArGEnT (Cross-Attention)

Reproduction of Section 3.3 of the ArGEnT paper (Chen et al. 2026) inside the
PhysicsNeMo framework. Trains the cross-attention ArGEnT variant on 2D
lid-driven cavity flow across 2400 varying-geometry cases and evaluates on a
600-case held-out set plus a 10-case half-circle extrapolation set.

**Model:** [`physicsnemo.experimental.models.argent.ArGEnTDeepONet`](../../../physicsnemo/experimental/models/argent/argent.py)
— DeepONet-style trunk-branch operator with a linear cross-attention trunk
over a geometry-aware point cloud.

## Data

Two h5 files, each with a sibling `_manifest.json`, from the ArGEnT
authors' LidDriven example:

| File | Cases | Purpose |
| --- | --- | --- |
| `LidDriven_dataset_forTrain.h5` | 3000 (2400 train / 600 test) | Main train + test |
| `LidDriven_dataset_halfCircle_forExtrapolation.h5` | 10 | Out-of-distribution eval |

Each case ships `xyd (N, 3)` (query points with SDF), `field (N, 3)` (u, v, p),
and `boundaryPoints (M, 2)` (polygon vertices for the case's cavity shape).
See `dataset.py` for the loader.

A public download location is still to be determined; until then, obtain
the four files from the ArGEnT authors and place them in `./data/`, which is
where `conf/config.yaml` and `run_lid_driven.sh` look by default:

```text
data/LidDriven_dataset_forTrain.h5
data/LidDriven_dataset_forTrain_manifest.json
data/LidDriven_dataset_halfCircle_forExtrapolation.h5
data/LidDriven_dataset_halfCircle_forExtrapolation_manifest.json
```

Or point the config at another location:

```bash
export LIDDRIVEN_TRAIN_H5=/path/to/LidDriven_dataset_forTrain.h5
export LIDDRIVEN_HALFCIRCLE_H5=/path/to/LidDriven_dataset_halfCircle_forExtrapolation.h5
```

Or override on the CLI: `python train.py data.train_path=...`.

## Install

```bash
pip install -e ../../..           # physicsnemo, editable
pip install -r requirements.txt   # example deps
```

`pyDOE<=0.9.3` is a hard requirement — later versions rename the module to
`pydoe`, which breaks the import. No scipy LHS fallback; see
`dataset.py:load_or_create_geometry_cloud` for why.

## Train

Single-GPU:

```bash
python train.py
```

Multi-GPU (DDP, N ranks):

```bash
torchrun --nproc_per_node=N train.py
```

Common overrides:

```bash
# quick smoke test
python train.py training.max_epochs=2 data.num_workers=0
# tune
python train.py training.lr=5e-4 data.batch_size_case=4
# resume
python train.py training.resume=true training.output_dir=./outputs/<run>
```

Outputs land under `${training.output_dir}/`:

- `GeometryPoints.h5` — the LHS geometry cloud (rank-0 sampled, all ranks read)
- `checkpoints/` — periodic checkpoint (every `training.checkpoint_every_epochs`)
- `checkpoint_best/` — best checkpoint by test-set relative L2
- `tb/` — TensorBoard event files

## Running on a multi-GPU node

[run_lid_driven.sh](run_lid_driven.sh) sets up a virtual environment (once,
in `./venv`, with [uv](https://docs.astral.sh/uv/) and its own Python 3.12, so
the system Python version does not matter), picks the PyTorch CUDA build that
matches the driver, trains with DDP over every visible GPU, then evaluates the best
checkpoint on both the test set and the half-circle extrapolation set. It
expects the data in `./data/` (see [Data](#data)) and stops early if it is
missing.

```bash
# 8x A100: 4 cases per rank, global batch 32
./run_lid_driven.sh
# tag the run; outputs go to ./outputs/lid_v1
MODEL_NAME=lid_v1 ./run_lid_driven.sh
# resume that run
MODEL_NAME=lid_v1 ./run_lid_driven.sh training.resume=true
# inference only
MODEL_NAME=lid_v1 STAGES=pred ./run_lid_driven.sh
```

Extra arguments are passed to `train.py` as Hydra overrides. `GLOBAL_BATCH`
(default 32, the single-GPU batch) is split evenly across ranks so the
effective batch does not depend on the GPU count. See the header of the
script for all environment variables.

On a cloud GPU instance (e.g. NVIDIA Brev), clone this repository, copy the
data into `examples/cfd/lid_driven_argent/data/`, and run the script in the
background:

```bash
nohup ./run_lid_driven.sh > run.log 2>&1 &
```

## Inference

```bash
CKPT=./outputs/<run>/checkpoint_best
python inference.py checkpoint_dir=$CKPT                               # main test set
python inference.py checkpoint_dir=$CKPT inference.dataset=halfcircle  # extrapolation
python inference.py checkpoint_dir=$CKPT inference.n_plot=4            # save 4 plots
python inference.py checkpoint_dir=$CKPT inference.save_predictions=true
```

Writes `per_case_errors.csv` (per-case per-field relative L2) and, if enabled,
per-case scatter plots of target/prediction/error and a compressed `.npz` of
raw predictions.

## Epoch count derivation

Training is budgeted in **epochs**, so the amount of data seen does not depend
on the number of ranks or the per-rank batch size. The default corresponds to
100,000 single-GPU iterations at 32 cases per batch over 2400 training cases:

```text
max_epochs ≈ 100000 * batch_size_case / n_train_cases ≈ 1334  (defaults 32 / 2400)
```

The learning-rate schedule is expressed in **progress form**,
`lr_decay_rate ** (400 * step / total_steps)`, so it depends on the fraction of
training completed rather than the absolute step count.

## Implementation notes

- **RoPE layout.** The continuous-coordinate RoPE uses the interleaved cos/sin
  layout of `physicsnemo.nn.module.rope.apply_rotary_pos_emb`, which rotates
  adjacent channel pairs.
- **Case-level DDP sharding.** Each rank owns whole cases; the per-case
  query-point subsample (`data.batch_size_point`) happens on the owning rank
  inside `__getitem__`.
- **Loss.** MSE over valid query points averaged over channels,
  `R2.sum() / (C * pad.sum())`; see `padded_mse_loss` in [utils.py](utils.py).
- **Best checkpoint.** The test set is evaluated every epoch and
  `checkpoint_best/` keeps the lowest test-set relative L2.
- **No output normalization.** The fields are non-dimensional and O(1).

## Limitations

- Only the cross-attention ArGEnT variant; `attention_type="self"` and
  `"hybrid"` raise `NotImplementedError`.
- The ArGEnTplus / GeoAnchor variant, the SOAP optimizer, and self-adaptive
  per-point loss weighting are not included.
