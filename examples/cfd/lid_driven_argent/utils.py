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

r"""Training utilities for the lid_driven_argent example.

Contains the padded MSE loss, relative-L2 error metrics, and RNG helpers
for reproducible DDP + multi-worker DataLoader runs.

No custom ``collate_fn`` is needed: :class:`ArGEnTDataset.__getitem__`
returns a flat ``dict[str, torch.Tensor]`` (plus scalar ``case_id``), which
PyTorch's default_collate stacks correctly.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


def sync_and_time(device: torch.device) -> float:
    r"""Return ``time.perf_counter()`` after a CUDA sync when applicable.

    On CUDA, kernel launches are async — a bare ``perf_counter()`` around a
    forward/backward measures launch time, not compute time. Synchronize
    first. On CPU (or non-CUDA accelerators without cuda semantics), just
    read the timer.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class PhaseTimer:
    r"""Per-phase second accumulator with cross-rank max/min reduce.

    Wraps regions of the training loop with named phases; accumulates the
    seconds spent in each; at epoch boundaries, reduces the per-phase totals
    across ranks so rank 0 can print ``max / min`` per phase. That surfaces
    imbalance directly — e.g. rank 0 doing eval while other ranks sit in
    ``barrier`` shows up as `eval_train=[42s max, 0s min]` alongside
    `barrier=[52s max, 0s min]`.

    Use :meth:`time` (a ``sync_and_time``-bracketed contextmanager) for GPU
    work — the sync ensures async kernel launches are actually finished
    before the timer stops. Use :meth:`add` with a raw ``perf_counter``
    delta for CPU-side waits (e.g. DataLoader fetch) — a sync there would
    kill DataLoader/compute overlap.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.times: dict[str, float] = {}

    def add(self, name: str, dt: float) -> None:
        r"""Accumulate ``dt`` seconds into phase ``name`` (no sync)."""
        self.times[name] = self.times.get(name, 0.0) + float(dt)

    @contextmanager
    def time(self, name: str):
        r"""Context manager: accumulate GPU-synchronized elapsed time into ``name``."""
        t0 = sync_and_time(self.device)
        try:
            yield
        finally:
            self.add(name, sync_and_time(self.device) - t0)

    def reset(self) -> None:
        r"""Discard all accumulated phase timings."""
        self.times.clear()

    def reduce(self, dm) -> dict:
        r"""Return ``{name: (max_s, min_s)}`` across ranks.

        Uses ``all_gather_object`` — collective; all ranks must call. In
        non-distributed mode, returns ``(v, v)`` for each local phase. Key
        order follows first-appearance across ranks (rank 0 first), which
        keeps the printed log stable epoch-to-epoch.
        """
        times = dict(self.times)
        if not dm.distributed:
            return {k: (v, v) for k, v in times.items()}
        gathered: list[dict] = [None] * dm.world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, times)
        seen: list[str] = []
        for d in gathered:
            for k in d:
                if k not in seen:
                    seen.append(k)
        return {
            k: (
                max(d.get(k, 0.0) for d in gathered),
                min(d.get(k, 0.0) for d in gathered),
            )
            for k in seen
        }

    def log(self, dm, log, tag: str) -> None:
        r"""Collective: reduce, then print one line on rank 0. Does NOT reset."""
        r = self.reduce(dm)
        if dm.rank == 0 and r:
            parts = [f"{k}=[{mx:.3f}s max, {mn:.3f}s min]" for k, (mx, mn) in r.items()]
            log.info(f"timings@{tag}: " + ", ".join(parts))


def set_seed(seed: int, rank: int = 0) -> None:
    r"""Deterministically seed Python, NumPy, and PyTorch RNGs (per-rank offset).

    Adds ``rank`` to the base seed so different ranks under DDP see different
    shuffles / sampling while still being reproducible.
    """
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn_factory(base_seed: int, rank: int) -> callable:
    r"""Return a ``worker_init_fn`` that seeds each DataLoader worker deterministically.

    Combines ``(base_seed, rank, worker_id)`` so different (rank, worker)
    combinations get different but reproducible RNG streams, so intra-case
    point subsampling is reproducible and uncorrelated across ranks.

    Parameters
    ----------
    base_seed : int
        The base training seed (typically ``cfg.training.seed``).
    rank : int
        The DDP rank (0 in single-process training).

    Returns
    -------
    callable
        A ``worker_init_fn(worker_id)`` closure suitable for
        :class:`torch.utils.data.DataLoader`.
    """

    def _init(worker_id: int) -> None:
        s = int(base_seed) + int(rank) * 1000 + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _init


def padded_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_query: torch.Tensor,
) -> torch.Tensor:
    r"""MSE over valid (unpadded) query points, averaged over channels.

    Computes ``mean(R2) * pad.numel() / pad.sum()`` with
    ``R2 = (pred - target)**2 * pad``, i.e. ``R2.sum() / (C * pad.sum())``.
    Note the ``1/C`` factor: ``R2.sum() / pad.sum()`` would scale the loss by
    the number of output channels.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape :math:`(B, N, C)`.
    pad_query : torch.Tensor
        Boolean or float mask of shape :math:`(B, N, 1)`; ``True`` (or nonzero)
        counts toward the loss.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if not torch.compiler.is_compiling():
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pad_query.shape[:2] != pred.shape[:2] or pad_query.shape[-1] != 1:
            raise ValueError(
                f"pad_query must have shape (B, N, 1); got {tuple(pad_query.shape)}"
            )

    pad_f = pad_query.to(pred.dtype)
    R2 = (pred - target).pow(2) * pad_f  # (B, N, C)
    return R2.mean() * pad_f.numel() / pad_f.sum().clamp(min=1.0)


def per_field_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_query: torch.Tensor,
) -> torch.Tensor:
    r"""Relative L2 error per case, pooled over all channels.

    .. math::

        \text{error} = \sqrt{
            \frac{\sum_{i,c} (pred_{ic} - target_{ic})^2 \cdot pad_{i}}
                 {\sum_{i,c} target_{ic}^2 \cdot pad_{i}}
        }

    where the sums run over valid query points :math:`i` and channels
    :math:`c`. See :func:`per_channel_rel_l2` for one error per channel.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape :math:`(B, N, C)`.
    pad_query : torch.Tensor
        Shape :math:`(B, N, 1)`.

    Returns
    -------
    torch.Tensor
        Per-case relative L2 error, shape :math:`(B,)`.
    """
    if not torch.compiler.is_compiling():
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pad_query.shape[:2] != pred.shape[:2] or pad_query.shape[-1] != 1:
            raise ValueError(
                f"pad_query must have shape (B, N, 1); got {tuple(pad_query.shape)}"
            )

    pad_f = pad_query.to(pred.dtype)
    n_valid = pad_f.sum(dim=(1, 2)) * pred.shape[-1]  # (B,) valid entries
    mse = ((pred - target).pow(2) * pad_f).sum(dim=(1, 2)) / n_valid.clamp(min=1.0)
    ms = (target.pow(2) * pad_f).sum(dim=(1, 2)) / n_valid.clamp(min=1.0)
    return torch.sqrt(mse / ms.clamp(min=1e-30))


def per_channel_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_query: torch.Tensor,
) -> torch.Tensor:
    r"""Per-case, per-channel relative L2 error.

    Complements :func:`per_field_rel_l2` (which averages MSE across channels
    before taking the square root). This variant keeps channels separate so
    a caller can report ``[u, v, p]`` errors individually.

    Parameters
    ----------
    pred, target : torch.Tensor
        Shape :math:`(B, N, C)`.
    pad_query : torch.Tensor
        Shape :math:`(B, N, 1)`.

    Returns
    -------
    torch.Tensor
        Per-case per-channel relative L2 error, shape :math:`(B, C)`.
    """
    if not torch.compiler.is_compiling():
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape; "
                f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if pad_query.shape[:2] != pred.shape[:2] or pad_query.shape[-1] != 1:
            raise ValueError(
                f"pad_query must have shape (B, N, 1); got {tuple(pad_query.shape)}"
            )

    pad_f = pad_query.to(pred.dtype)
    n_valid = pad_f.sum(dim=1)  # (B, 1) — per-batch valid-point count
    mse = ((pred - target).pow(2) * pad_f).sum(dim=1) / n_valid.clamp(min=1.0)  # (B, C)
    ms = (target.pow(2) * pad_f).sum(dim=1) / n_valid.clamp(min=1.0)  # (B, C)
    return torch.sqrt(mse / ms.clamp(min=1e-30))
