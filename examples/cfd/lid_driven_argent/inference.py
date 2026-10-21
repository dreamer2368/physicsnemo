# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Inference / postprocessing for the lid-driven cavity ArGEnT example.

Loads a trained checkpoint and evaluates it on either the main test set or
the half-circle extrapolation set. Writes per-case per-field relative L2
errors to CSV, and optionally saves solution-vs-truth field plots for a
few cases (matplotlib).

Usage
-----
Main test set::

    python inference.py checkpoint_dir=./outputs/2026-01-15_10-30-00/checkpoint_best

Extrapolation set::

    python inference.py checkpoint_dir=./outputs/.../checkpoint_best inference.dataset=halfcircle

With plots::

    python inference.py checkpoint_dir=./outputs/.../checkpoint_best inference.n_plot=4
"""

from __future__ import annotations

import csv
import os

import hydra
import numpy as np
import torch
from dataset import ArGEnTDataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils import per_channel_rel_l2, per_field_rel_l2

from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    r"""Evaluate a trained checkpoint and write per-case errors and plots."""
    log = PythonLogger("lid_driven_argent.inference")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    ckpt_dir = cfg.get("checkpoint_dir")
    if ckpt_dir is None:
        raise ValueError(
            "Pass checkpoint_dir=<path> (e.g. ./outputs/<run>/checkpoint_best)"
        )

    # Where the training run wrote the LHS geometry cloud. Inference MUST read
    # the same cloud the model was trained on — resampling would give a
    # different (x, y) cloud and invalidate the trained weights.
    train_output_dir = cfg.inference.get(
        "train_output_dir", os.path.dirname(os.path.abspath(ckpt_dir))
    )
    geo_path = os.path.join(train_output_dir, "GeometryPoints.h5")
    if not os.path.exists(geo_path):
        raise FileNotFoundError(
            f"Expected geometry cloud at {geo_path} (produced by train.py). "
            f"Pass inference.train_output_dir=<path> if it lives elsewhere."
        )

    dataset_mode = cfg.inference.get("dataset", "main")
    if dataset_mode == "main":
        h5_path = cfg.data.train_path
        task = "test"
    elif dataset_mode == "halfcircle":
        h5_path = cfg.data.extrapolation_path
        task = "test"
    else:
        raise ValueError(
            f"Unknown inference.dataset={dataset_mode!r}; use 'main' or 'halfcircle'"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint from {ckpt_dir} on {device}")
    # physicsnemo.utils.load_checkpoint requires an existing model instance to
    # populate; alternatively for the ".mdlus" flat-file API you'd use
    # Module.from_checkpoint(path). We use the training-format checkpoint that
    # save_checkpoint(path=dir) writes, and rebuild the model from the config.
    model = hydra.utils.instantiate(cfg.model).to(device)

    epoch = load_checkpoint(path=ckpt_dir, models=model, device=device)
    log.info(f"Loaded checkpoint from epoch {epoch}")
    model.eval()

    log.info(f"Evaluating on dataset={dataset_mode} (h5={h5_path}, task={task})")
    ds = ArGEnTDataset(
        h5_path=h5_path,
        task=task,
        use_fields=list(cfg.training.target_fields),
        n_point_cloud_geo_rep=cfg.data.n_point_cloud_geo_rep,
        max_num_points=cfg.data.max_num_points,  # padding pad; caller can override
        geometry_domain=[list(d) for d in cfg.data.geometry_domain],
        geometry_points_path=geo_path,
        subsample_points=None,  # evaluate on every point per case
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    target_fields = list(cfg.training.target_fields)
    rows = []
    field_errs, chan_errs = [], []
    predictions = {} if cfg.inference.get("save_predictions", False) else None
    n_plot = int(cfg.inference.get("n_plot", 0))
    plot_dir = os.path.join(
        cfg.inference.get("output_dir", "./inference_out"), dataset_mode
    )
    os.makedirs(plot_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_query = batch["x_query"].to(device)
            x_kv = batch["x_kv"].to(device)
            op_params = batch["op_params"].to(device)
            pad_query = batch["pad_query"].to(device)
            target = batch["target"].to(device)
            pred = model(x_query, x_kv, op_params, padding_kv=None)

            field_err = per_field_rel_l2(pred, target, pad_query).cpu().numpy()  # (1,)
            chan_err = (
                per_channel_rel_l2(pred, target, pad_query).cpu().numpy()
            )  # (1, C)

            case_id = int(batch["case_id"][0])
            row = {"case_id": case_id, "rel_L2_avg": float(field_err[0])}
            for j, name in enumerate(target_fields):
                row[f"rel_L2_{name}"] = float(chan_err[0, j])
            rows.append(row)
            field_errs.append(field_err[0])
            chan_errs.append(chan_err[0])

            if predictions is not None:
                predictions[case_id] = {
                    "x_query": x_query[0].cpu().numpy(),
                    "pred": pred[0].cpu().numpy(),
                    "target": target[0].cpu().numpy(),
                    "pad_query": pad_query[0].cpu().numpy(),
                }
            if i < n_plot:
                _plot_case(
                    case_id=case_id,
                    x_query=x_query[0].cpu().numpy(),
                    pred=pred[0].cpu().numpy(),
                    target=target[0].cpu().numpy(),
                    pad_query=pad_query[0].cpu().numpy(),
                    target_fields=target_fields,
                    out_path=os.path.join(plot_dir, f"case_{case_id:06d}.png"),
                )

    # Summary
    field_errs_arr = np.array(field_errs)
    chan_errs_arr = np.stack(chan_errs, axis=0)  # (n_cases, C)
    log.info(f"Mean per-case rel-L2 (avg over channels): {field_errs_arr.mean():.4e}")
    for j, name in enumerate(target_fields):
        log.info(f"  Mean rel-L2[{name}]: {chan_errs_arr[:, j].mean():.4e}")

    # Write per-case CSV
    csv_path = os.path.join(plot_dir, "per_case_errors.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote per-case errors to {csv_path}")

    if predictions is not None:
        pred_path = os.path.join(plot_dir, "predictions.npz")
        np.savez_compressed(
            pred_path,
            case_ids=np.array(list(predictions.keys())),
            **{
                f"case_{cid}_{k}": v
                for cid, d in predictions.items()
                for k, v in d.items()
            },
        )
        log.info(f"Saved raw predictions to {pred_path}")


def _plot_case(case_id, x_query, pred, target, pad_query, target_fields, out_path):
    r"""Save a target-vs-prediction scatter plot for one case (one panel per field)."""
    import matplotlib.pyplot as plt

    mask = pad_query[:, 0].astype(bool)
    xs = x_query[mask, 0]
    ys = x_query[mask, 1]
    n_fields = len(target_fields)
    fig, axes = plt.subplots(
        3, n_fields, figsize=(4 * n_fields, 10), constrained_layout=True
    )
    if n_fields == 1:
        axes = axes[:, None]
    for j, name in enumerate(target_fields):
        vmin = float(min(target[mask, j].min(), pred[mask, j].min()))
        vmax = float(max(target[mask, j].max(), pred[mask, j].max()))
        for row, (data, label) in enumerate(
            [
                (target[mask, j], f"target {name}"),
                (pred[mask, j], f"pred {name}"),
                (pred[mask, j] - target[mask, j], f"err {name}"),
            ]
        ):
            ax = axes[row, j]
            kwargs = dict(c=data, s=4, cmap="viridis" if row < 2 else "coolwarm")
            if row < 2:
                kwargs.update(vmin=vmin, vmax=vmax)
            sc = ax.scatter(xs, ys, **kwargs)
            ax.set_title(label)
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f"case {case_id}")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
