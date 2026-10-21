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

r"""DDP training driver for the lid-driven cavity ArGEnT example.

Usage
-----
Single-GPU::

    python train.py

Multi-GPU (DDP)::

    torchrun --nproc_per_node=<N> train.py

Overrides are Hydra-style::

    python train.py training.max_epochs=2 data.batch_size_case=4

See ``README.md`` for the epoch-count derivation.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Optional

import hydra
import torch
import torch.distributed as dist
from dataset import ArGEnTDataset, load_or_create_geometry_cloud
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import (
    PhaseTimer,
    padded_mse_loss,
    per_channel_rel_l2,
    per_field_rel_l2,
    set_seed,
    worker_init_fn_factory,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


def build_dataloaders(
    cfg: DictConfig,
    dist_mgr: DistributedManager,
    startup_timer: Optional["PhaseTimer"] = None,
):
    r"""Build train/test datasets and their DataLoaders.

    Rank 0 pre-materializes the LHS geometry cloud to disk and all ranks then
    load it, so every rank sees the same cloud. The test set inherits
    ``max_num_points`` from the train set.

    Two DDP data paths, selected by ``cfg.data.load_shard``:
    - ``load_shard=False`` (default): each rank builds the full cache;
      ``DistributedSampler`` shards the *access*, not the storage.
    - ``load_shard=True``: each rank builds only its 1/world_size shard of
      the cache and uses a plain sampler. Cuts startup FS contention and
      per-rank memory ~1/N; requires that ``evaluate(...)`` reduces metrics
      across ranks (it does).

    ``startup_timer`` (optional) accumulates per-phase seconds for the
    geometry-cloud step and each of the dataset constructions — see
    ``PhaseTimer`` in ``utils.py``.
    """
    _t = (
        (lambda name: startup_timer.time(name))
        if startup_timer is not None
        else (lambda _name: nullcontext())
    )
    with _t("geometry_cloud"):
        geo_path = os.path.join(cfg.training.output_dir, "GeometryPoints.h5")
        if dist_mgr.rank == 0:
            os.makedirs(cfg.training.output_dir, exist_ok=True)
            load_or_create_geometry_cloud(
                path=geo_path,
                n_points=cfg.data.n_point_cloud_geo_rep,
                domain=[list(d) for d in cfg.data.geometry_domain],
                seed=cfg.training.seed,
            )
        if dist_mgr.distributed:
            dist.barrier()

    load_shard = bool(cfg.data.get("load_shard", False))
    with _t("dataset_train"):
        train_set = ArGEnTDataset(
            h5_path=cfg.data.train_path,
            task="train",
            use_fields=list(cfg.training.target_fields),
            n_point_cloud_geo_rep=cfg.data.n_point_cloud_geo_rep,
            max_num_points=cfg.data.max_num_points,
            geometry_domain=[list(d) for d in cfg.data.geometry_domain],
            geometry_points_path=geo_path,
            subsample_points=cfg.data.batch_size_point,
            load_shard=load_shard,
            rank=dist_mgr.rank,
            world_size=dist_mgr.world_size,
        )
    with _t("dataset_test"):
        test_set = ArGEnTDataset(
            h5_path=cfg.data.train_path,
            task="test",
            use_fields=list(cfg.training.target_fields),
            n_point_cloud_geo_rep=cfg.data.n_point_cloud_geo_rep,
            max_num_points=train_set.max_num_points,  # inherit from train
            geometry_domain=[list(d) for d in cfg.data.geometry_domain],
            geometry_points_path=geo_path,
            subsample_points=None,  # eval on all points per case
            load_shard=load_shard,
            rank=dist_mgr.rank,
            world_size=dist_mgr.world_size,
        )

    # Sampler selection:
    #   - load_shard=True → per-rank local dataset; plain sampler (shuffle
    #     kwarg on DataLoader picks RandomSampler / SequentialSampler).
    #   - not load_shard, distributed → DistributedSampler shards indices
    #     over the global dataset.
    #   - not distributed → single-process, plain sampler.
    if load_shard or not dist_mgr.distributed:
        train_sampler = None
        test_sampler = None
    else:
        train_sampler = DistributedSampler(
            train_set,
            rank=dist_mgr.rank,
            num_replicas=dist_mgr.world_size,
            shuffle=True,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_set,
            rank=dist_mgr.rank,
            num_replicas=dist_mgr.world_size,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size_case,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.num_workers,
        pin_memory=(dist_mgr.device.type == "cuda"),
        worker_init_fn=worker_init_fn_factory(cfg.training.seed, dist_mgr.rank),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        sampler=test_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=(dist_mgr.device.type == "cuda"),
    )
    return train_set, train_loader, test_loader


def evaluate(model, loader, device, target_fields, dm):
    r"""Compute per-field and per-channel rel-L2 over a loader — DDP-aware.

    Every rank forwards over its local shard (whether that shard was
    created by ``DistributedSampler`` on a globally-addressable dataset, or
    by the dataset's own ``load_shard`` slicing). Per-case error tensors
    are gathered across ranks via ``all_gather_object`` (handles unequal
    shard sizes) and the aggregate is computed identically on every rank.
    All ranks return the same dict.
    """
    model.eval()
    field_errs = []
    chan_errs = []
    with torch.no_grad():
        for batch in loader:
            x_query = batch["x_query"].to(device)
            x_kv = batch["x_kv"].to(device)
            op_params = batch["op_params"].to(device)
            pad_query = batch["pad_query"].to(device)
            target = batch["target"].to(device)
            pred = model(x_query, x_kv, op_params, padding_kv=None)
            field_errs.append(per_field_rel_l2(pred, target, pad_query).cpu())
            chan_errs.append(per_channel_rel_l2(pred, target, pad_query).cpu())
    field_local = torch.cat(field_errs) if field_errs else torch.zeros(0)
    chan_local = (
        torch.cat(chan_errs) if chan_errs else torch.zeros(0, len(target_fields))
    )

    if dm.distributed:
        gathered: list = [None] * dm.world_size
        dist.all_gather_object(gathered, (field_local, chan_local))
        field = torch.cat([g[0] for g in gathered])
        chan = torch.cat([g[1] for g in gathered])
    else:
        field = field_local
        chan = chan_local

    per_field = {name: chan[:, i].mean().item() for i, name in enumerate(target_fields)}
    per_field["avg"] = field.mean().item()
    return per_field


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    r"""Train ArGEnT on the lid-driven cavity dataset (single GPU or DDP)."""
    DistributedManager.initialize()
    dm = DistributedManager()

    plog = PythonLogger("lid_driven_argent")
    log = RankZeroLoggingWrapper(plog, dm)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"World size: {dm.world_size}, rank: {dm.rank}, device: {dm.device}")

    set_seed(cfg.training.seed, dm.rank)

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    # One PhaseTimer, reused: startup phases first, then reset + reused for
    # per-epoch phases. Startup phase names ("geometry_cloud", "dataset_*",
    # "model_init", "ddp_wrap") do not collide with epoch names ("train",
    # "data_wait", "eval_*", "ckpt", "barrier"), and reset() between them
    # keeps each log line focused on one bucket.
    pt = PhaseTimer(dm.device)
    train_set, train_loader, test_loader = build_dataloaders(cfg, dm, startup_timer=pt)
    log.info(
        f"Datasets built: train n_cases={len(train_set)}, "
        f"max_num_points={train_set.max_num_points}, "
        f"steps/epoch={len(train_loader)}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    with pt.time("model_init"):
        model = hydra.utils.instantiate(cfg.model).to(dm.device)
    with pt.time("ddp_wrap"):
        if dm.distributed:
            device_ids = [dm.local_rank] if dm.device.type == "cuda" else None
            model = DDP(model, device_ids=device_ids)
    pt.log(dm, log, "startup")  # collective; rank 0 prints
    pt.reset()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model instantiated with {n_params:,} trainable parameters")

    # -----------------------------------------------------------------
    # Optimizer + progress-based LR schedule (world_size-invariant)
    # -----------------------------------------------------------------
    optimizer = Adam(model.parameters(), lr=cfg.training.lr, weight_decay=0.0)
    total_steps = int(cfg.training.max_epochs) * max(1, len(train_loader))
    # LR schedule: lr_decay_rate ** (400 * step / total_steps), i.e. 400 decay
    # steps over the run regardless of world size.
    decay_rate = float(cfg.training.lr_decay_rate)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: decay_rate ** (400.0 * step / max(1, total_steps)),
    )

    # -----------------------------------------------------------------
    # Resume from checkpoint (if requested)
    # -----------------------------------------------------------------
    ckpt_dir = os.path.join(cfg.training.output_dir, "checkpoints")
    if dm.rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    if cfg.training.resume:
        start_epoch = load_checkpoint(
            path=ckpt_dir,
            models=model.module if dm.distributed else model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=dm.device,
        )
        log.info(f"Resumed from checkpoint at epoch {start_epoch}")

    writer = (
        SummaryWriter(log_dir=os.path.join(cfg.training.output_dir, "tb"))
        if dm.rank == 0
        else None
    )
    # `checkpoint_best` is selected by test-set rel-L2.
    best_test_l2 = float("inf")

    # -----------------------------------------------------------------
    # Training loop — epoch-based
    # -----------------------------------------------------------------
    step = start_epoch * max(1, len(train_loader))
    for epoch in range(start_epoch, int(cfg.training.max_epochs)):
        if dm.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # `t_ready` marks "GPU is idle, waiting for next batch". After each
        # training step's sync inside `pt.time("train")`, we reset it — the
        # delta to the next batch's arrival is `data_wait` (no extra sync —
        # a sync here would kill DataLoader/compute overlap).
        t_ready = time.perf_counter()
        for batch in train_loader:
            t_batch = time.perf_counter()
            pt.add("data_wait", t_batch - t_ready)
            with pt.time("train"):
                x_query = batch["x_query"].to(dm.device, non_blocking=True)
                x_kv = batch["x_kv"].to(dm.device, non_blocking=True)
                op_params = batch["op_params"].to(dm.device, non_blocking=True)
                pad_query = batch["pad_query"].to(dm.device, non_blocking=True)
                target = batch["target"].to(dm.device, non_blocking=True)

                pred = model(x_query, x_kv, op_params, padding_kv=None)
                loss = padded_mse_loss(pred, target, pad_query)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1
            t_ready = time.perf_counter()

        avg_loss = epoch_loss / max(1, n_batches)
        current_lr = scheduler.get_last_lr()[0]
        log.info(
            f"epoch {epoch + 1}/{int(cfg.training.max_epochs)}  "
            f"train_loss={avg_loss:.4e}  lr={current_lr:.3e}  step={step}"
        )
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)

        # -----------------------------------------------------------------
        # Periodic evaluation (all ranks; metrics reduced inside evaluate)
        # + checkpoint (rank 0 only)
        # -----------------------------------------------------------------
        should_eval = (
            (epoch + 1) % int(cfg.training.checkpoint_every_epochs) == 0
        ) or (epoch + 1 == int(cfg.training.max_epochs))
        if should_eval:
            eval_model = model.module if dm.distributed else model
            with pt.time("eval_test"):
                test_metrics = evaluate(
                    eval_model,
                    test_loader,
                    dm.device,
                    list(cfg.training.target_fields),
                    dm,
                )
            if dm.rank == 0:
                log.info(f"  eval@epoch {epoch + 1}: test_L2={test_metrics}")
                if writer is not None:
                    for name, v in test_metrics.items():
                        writer.add_scalar(f"test/rel_L2_{name}", v, epoch)

                with pt.time("ckpt"):
                    save_checkpoint(
                        path=ckpt_dir,
                        models=eval_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        metadata={"test_rel_L2": test_metrics},
                    )
                    if test_metrics["avg"] < best_test_l2:
                        best_test_l2 = test_metrics["avg"]
                        best_dir = os.path.join(
                            cfg.training.output_dir, "checkpoint_best"
                        )
                        os.makedirs(best_dir, exist_ok=True)
                        save_checkpoint(
                            path=best_dir,
                            models=eval_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch + 1,
                            metadata={
                                "test_rel_L2": test_metrics,
                                "reason": "best_test_L2",
                            },
                        )
                        log.info(f"  -> new best (test_L2={best_test_l2:.4e})")

        if dm.distributed:
            with pt.time("barrier"):
                dist.barrier()

        pt.log(dm, log, f"epoch {epoch + 1}")  # collective; rank 0 prints
        pt.reset()

    if writer is not None:
        writer.close()
    log.info("Training complete")
    # NB: do not call dist.destroy_process_group() here — DistributedManager
    # registers cleanup() as an atexit handler that already
    # calls it. A manual call plus the atexit call makes the second one hit
    # `assert pg is not None` in torch.distributed and prints a bogus
    # AssertionError traceback per rank at shutdown.


if __name__ == "__main__":
    train()
