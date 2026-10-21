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

r"""H5-backed dataset for the lid-driven cavity example.

:class:`ArGEnTDataset` handles both the main training/test set and the
half-circle extrapolation set. Both h5 files share the
``cases/case######/{xyd, field, boundaryPoints}`` hierarchy, with sibling
manifest JSONs keyed by ``train`` / ``test``.

The SDF is computed at runtime from the 2D boundary polygon; 3D datasets
should subclass and override :meth:`ArGEnTDataset._compute_sdf`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from physicsnemo.mesh.generate import sdf_polygon_2d

logger = logging.getLogger(__name__)


def _read_manifest_field_names(manifest: dict) -> List[str]:
    r"""Return the manifest's ``field_names`` list (or ``[]`` if missing)."""
    val = manifest.get("field_names", [])
    return list(val) if val is not None else []


def sample_lhs_geometry_points(
    n_points: int,
    domain: Sequence[Sequence[float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""Sample ``n_points`` points via Latin Hypercube over ``domain``.

    Uses ``pyDOE.lhs`` directly. There is no scipy fallback: a different LHS
    RNG would produce a different geometry cloud, and a model trained on
    one cloud must be evaluated on the exact same cloud.

    Parameters
    ----------
    n_points : int
        Number of samples to draw.
    domain : sequence of ``(low, high)`` pairs
        Per-dimension bounds. For LidDriven: ``((-0.6, 0.6), (-2.1, 0.1))``.
    seed : int, optional
        Seed passed to numpy's RNG for the ``pyDOE.lhs`` call. If ``None``,
        the global numpy RNG is used.

    Returns
    -------
    np.ndarray
        Sampled points of shape ``(n_points, len(domain))``.
    """
    try:
        from pyDOE import lhs  # requires pyDOE<=0.9.3; newer versions rename
        # the import to `pydoe`. See requirements.txt.
    except ImportError as e:
        raise ImportError(
            "pyDOE is required to sample the geometry point cloud. "
            "Install a version that exposes `import pyDOE` (i.e. "
            "`pip install 'pyDOE<=0.9.3'`). No fallback to scipy.stats.qmc — a "
            "different LHS RNG would produce a different cloud, breaking "
            "trained-model reproducibility."
        ) from e

    if seed is not None:
        # pyDOE.lhs uses the global numpy RNG; seed it locally.
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            unit = lhs(len(domain), n_points)
        finally:
            np.random.set_state(rng_state)
    else:
        unit = lhs(len(domain), n_points)  # (n_points, n_dims), uniform in [0,1]

    low = np.array([d[0] for d in domain], dtype=np.float64)
    high = np.array([d[1] for d in domain], dtype=np.float64)
    return unit * (high - low) + low  # (n_points, n_dims)


def load_or_create_geometry_cloud(
    path: str,
    n_points: int,
    domain: Sequence[Sequence[float]],
    seed: Optional[int] = None,
    force_resample: bool = False,
) -> np.ndarray:
    r"""Load an LHS geometry cloud from ``path`` or create+save a fresh one.

    Central to DDP correctness: rank 0 calls this before dataset construction
    to write the cloud to disk, then :class:`ArGEnTDataset` on all ranks reads
    it (via its ``geometry_points_path`` argument). Never let ranks sample
    independently — different RNG per rank corrupts training silently.

    Parameters
    ----------
    path : str
        Path to an h5 file. Read from if it exists; otherwise sampled and written.
    n_points : int
        Number of points to sample when creating.
    domain : sequence of ``(low, high)`` pairs
        Per-dimension bounds; see :func:`sample_lhs_geometry_points`.
    seed : int, optional
        Passed to :func:`sample_lhs_geometry_points`.
    force_resample : bool, optional, default=False
        If ``True``, resample and overwrite even when ``path`` exists.

    Returns
    -------
    np.ndarray
        Cloud of shape ``(n_points, len(domain))``.
    """
    if os.path.exists(path) and not force_resample:
        with h5py.File(path, "r") as f:
            cloud = f["points"][:]
        logger.info(f"Loaded geometry cloud from {path} (shape={cloud.shape})")
        return cloud

    cloud = sample_lhs_geometry_points(n_points, domain, seed=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("points", data=cloud)
    logger.info(f"Sampled and saved geometry cloud to {path} (shape={cloud.shape})")
    return cloud


class ArGEnTDataset(Dataset):
    r"""H5-backed lid-driven cavity dataset for ArGEnT training and evaluation.

    Serves one case per ``__getitem__``, with intra-case query-point
    subsampling for training. The same class handles the main dataset and
    the half-circle extrapolation set.

    Parameters
    ----------
    h5_path : str
        Path to the h5 file (e.g. ``LidDriven_dataset_forTrain.h5``).
    task : str
        Manifest key selecting which case IDs to load — ``"train"`` or
        ``"test"``. The extrapolation set has only ``"test"``.
    use_fields : list of str
        Ordered list of field names to load as target channels; each must
        appear in ``hf["cases"].attrs["field_names"]``. Example:
        ``["u", "v", "p"]``.
    n_point_cloud_geo_rep : int, optional, default=200
        Number of geometry-cloud points. Ignored if ``geometry_points_path``
        points at an existing file (in which case the count comes from disk).
    max_num_points : int, optional
        Padding size for query points per case. If ``None``, uses the 95th
        percentile of the task's case sizes. The test set must inherit
        ``max_num_points`` from the train set (pass ``train.max_num_points``)
        rather than recompute it.
    dim_space : int, optional, default=2
        Spatial dimension. Only ``2`` uses the default polygon-SDF path;
        subclass and override :meth:`_compute_sdf` for 3D.
    geometry_domain : sequence of ``(low, high)`` pairs, optional
        Per-axis bounds for the LHS point cloud when it must be sampled.
        Default matches LidDriven ``((-0.6, 0.6), (-2.1, 0.1))``.
    geometry_points_path : str, optional
        Path to an h5 with a ``points`` dataset holding the pre-sampled LHS
        cloud (typically produced by :func:`load_or_create_geometry_cloud`).
        If given and exists, the cloud is loaded — cases share these (x, y)
        columns; the ``sdf`` column is refilled per case (see
        :meth:`_compute_sdf`).
    subsample_points : int, optional
        If given, ``__getitem__`` returns ``subsample_points`` query points
        per case, topping up with pad slots when a case has fewer valid
        points than requested (see :meth:`__getitem__`).
        If ``None`` / ``<=0``, returns all ``max_num_points`` slots (eval mode).
    dtype : numpy dtype, optional, default=``np.float32``
        Numeric dtype for cached tensors.
    load_shard : bool, optional, default=``False``
        If ``True``, this rank only loads its 1/``world_size`` slice of
        ``case_ids`` (interleaved: ``case_ids[rank::world_size]``). The
        dataset becomes rank-local: ``len(self)`` returns the local count.
        Requires the caller to use a plain sampler (not DistributedSampler)
        and to reduce eval metrics across ranks. Only meaningful when
        ``world_size > 1``.
    rank, world_size : int, optional, defaults ``0, 1``
        DDP rank / total ranks. Consulted only when ``load_shard=True``.
    """

    def __init__(
        self,
        h5_path: str,
        task: str,
        use_fields: Sequence[str],
        n_point_cloud_geo_rep: int = 200,
        max_num_points: Optional[int] = None,
        dim_space: int = 2,
        geometry_domain: Sequence[Sequence[float]] = ((-0.6, 0.6), (-2.1, 0.1)),
        geometry_points_path: Optional[str] = None,
        subsample_points: Optional[int] = None,
        dtype: np.dtype = np.float32,
        load_shard: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.task = task
        self.use_fields = list(use_fields)
        self.n_point_cloud_geo_rep = int(n_point_cloud_geo_rep)
        self.dim_space = int(dim_space)
        self.geometry_domain = tuple(tuple(d) for d in geometry_domain)
        self.geometry_points_path = geometry_points_path
        self.subsample_points = (
            subsample_points if subsample_points and subsample_points > 0 else None
        )
        self.dtype = dtype

        manifest_path = h5_path.replace(".h5", "_manifest.json")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        if task not in self.manifest:
            raise KeyError(
                f"Manifest at {manifest_path} has no task {task!r}; "
                f"available: {list(self.manifest.keys())}"
            )
        # NB: schema + case-size scan reads the FULL task manifest so the
        # 95th-percentile-derived max_num_points is consistent across ranks
        # regardless of load_shard. The shard slice happens after.
        self.case_ids: List[int] = list(self.manifest[task])
        self._read_schema_and_case_sizes()
        if max_num_points is not None and max_num_points > 0:
            self.max_num_points = int(max_num_points)
        else:
            self.max_num_points = int(np.percentile(self._case_sizes, 95))
        logger.info(
            f"[{task}] global_cases={len(self.case_ids)}, "
            f"max_num_points={self.max_num_points} "
            f"(sizes min/med/95%/max = {int(np.min(self._case_sizes))}/"
            f"{int(np.median(self._case_sizes))}/"
            f"{int(np.percentile(self._case_sizes, 95))}/"
            f"{int(np.max(self._case_sizes))})"
        )

        # Optional per-rank sharding: interleaved so all ranks see similar
        # case-size distributions when cases are grouped by geometry in the h5.
        self.load_shard = bool(load_shard) and int(world_size) > 1
        if self.load_shard:
            self.case_ids = self.case_ids[int(rank) :: int(world_size)]
            logger.info(
                f"[{task}] load_shard=True: rank {rank}/{world_size} "
                f"→ local_cases={len(self.case_ids)}"
            )

        # Field indexing.
        for f in self.use_fields:
            if f not in self.field_names:
                raise ValueError(
                    f"Field {f!r} not in h5 field_names={self.field_names}"
                )
        self.field_indices = [self.field_names.index(f) for f in self.use_fields]
        self.dim_out = len(self.field_indices)

        # Geometry cloud: shared (x, y) columns across all cases; the sdf
        # column is refilled per case in _build_case_arrays.
        self.geometry_cloud_xy = self._load_geometry_cloud()

        # Load and pre-pad every case into an in-memory cache. Modest size for
        # LidDriven (2400 train cases * ~3000 pts * 3 fields * 4 bytes ~= 80 MB).
        self.cache: List[dict] = []
        self._build_cache()

    # ------------------------------------------------------------------
    # Schema and sizes
    # ------------------------------------------------------------------
    def _read_schema_and_case_sizes(self) -> None:
        r"""Read ``cases.attrs`` and per-case sizes for the selected task IDs."""
        self._case_sizes: List[int] = []
        with h5py.File(self.h5_path, "r") as hf:
            cases_grp = hf["cases"]
            self.field_names = [c.decode() for c in cases_grp.attrs["field_names"]]
            self.coord_names = [c.decode() for c in cases_grp.attrs["coord_names"]]
            self.op_param_names = [c.decode() for c in cases_grp.attrs["OPParas_names"]]
            self.geo_param_names = [
                c.decode() for c in cases_grp.attrs["GeoParas_names"]
            ]
            selected = set(self.case_ids)
            for case_name in cases_grp.keys():
                dst = cases_grp[case_name]
                if not isinstance(dst, h5py.Group):
                    continue
                if int(dst.attrs["caseID"]) not in selected:
                    continue
                self._case_sizes.append(int(dst["field"].shape[0]))
        if not self._case_sizes:
            raise RuntimeError(f"No cases matched task={self.task!r} in {self.h5_path}")

    def _load_geometry_cloud(self) -> np.ndarray:
        r"""Return the shared LHS geometry cloud ``(x, y)`` — from disk or sampled.

        DDP callers must pre-materialize this on rank 0 via
        :func:`load_or_create_geometry_cloud` and pass the path in
        ``geometry_points_path`` so all ranks read the same cloud.
        """
        if self.geometry_points_path and os.path.exists(self.geometry_points_path):
            with h5py.File(self.geometry_points_path, "r") as f:
                cloud = f["points"][:]
            logger.info(
                f"Loaded geometry cloud from {self.geometry_points_path} "
                f"(shape={cloud.shape})"
            )
        else:
            # Rank-safe fallback for single-process use; DDP callers should not hit this.
            cloud = load_or_create_geometry_cloud(
                path=self.geometry_points_path or "GeometryPoints.h5",
                n_points=self.n_point_cloud_geo_rep,
                domain=self.geometry_domain,
            )
        if cloud.shape[-1] < self.dim_space:
            raise ValueError(
                f"Geometry cloud has {cloud.shape[-1]} spatial dims, need at least "
                f"dim_space={self.dim_space}"
            )
        return np.ascontiguousarray(cloud[:, : self.dim_space], dtype=np.float64)

    # ------------------------------------------------------------------
    # Per-case build
    # ------------------------------------------------------------------
    def _compute_sdf(self, case_group: h5py.Group, query_xy: np.ndarray) -> np.ndarray:
        r"""Signed distance from ``query_xy`` to the case's boundary polygon.

        Default is the 2D polygon path — uses
        :func:`~physicsnemo.mesh.generate.implicit_functions.sdf_polygon_2d`
        over the case's ``boundaryPoints`` dataset.

        For 3D data, subclass :class:`ArGEnTDataset` and override this
        method, e.g. to read a precomputed SDF from the h5 or to call
        :func:`~physicsnemo.mesh.spatial.signed_distance_field` for triangle
        meshes.

        Parameters
        ----------
        case_group : h5py.Group
            The case's group inside ``hf["cases"]``.
        query_xy : np.ndarray
            Query points of shape ``(N, dim_space)``.

        Returns
        -------
        np.ndarray
            Signed distances of shape ``(N,)``. Negative inside the domain,
            positive outside.
        """
        if self.dim_space != 2:
            raise NotImplementedError(
                f"Default _compute_sdf only supports dim_space=2, got "
                f"{self.dim_space}. Subclass ArGEnTDataset and override "
                f"_compute_sdf for 3D — see the dataset module docstring."
            )
        vertices = case_group["boundaryPoints"][:]
        phi = sdf_polygon_2d(vertices)
        d = phi(torch.as_tensor(query_xy, dtype=torch.float64)).numpy()
        return d

    def _build_case_arrays(self, case_group: h5py.Group):
        r"""Extract x_query, x_kv (with per-case SDF), op_params, geo_params, target for one case."""
        x_query = case_group["xyd"][:]  # (N, dim_space+1) — third col is sdf per case
        target = case_group["field"][:][:, self.field_indices]  # (N, dim_out)
        op_params = case_group.attrs["OPParas"].reshape(1, -1).astype(self.dtype)
        geo_params = case_group.attrs["GeoParas"].reshape(1, -1).astype(self.dtype)

        # x_kv: shared (x, y) cloud + per-case SDF column
        x_kv = np.zeros(
            (self.geometry_cloud_xy.shape[0], self.dim_space + 1), dtype=np.float64
        )
        x_kv[:, : self.dim_space] = self.geometry_cloud_xy
        x_kv[:, self.dim_space] = self._compute_sdf(case_group, self.geometry_cloud_xy)
        return x_query, x_kv, op_params, geo_params, target

    def _build_cache(self) -> None:
        r"""Load every selected case into an in-memory record with padded x_query/target."""
        selected = set(self.case_ids)
        max_pts = int(self.max_num_points)
        rng = np.random.default_rng(0)
        with h5py.File(self.h5_path, "r") as hf:
            cases_grp = hf["cases"]
            for case_name in cases_grp.keys():
                dst = cases_grp[case_name]
                if not isinstance(dst, h5py.Group):
                    continue
                cid = int(dst.attrs["caseID"])
                if cid not in selected:
                    continue
                x_query, x_kv, op_params, geo_params, target = self._build_case_arrays(
                    dst
                )

                n_pts = target.shape[0]
                if n_pts > max_pts:
                    logger.warning(
                        f"Case {cid}: {n_pts} points > max_num_points={max_pts}; "
                        f"randomly downsampling."
                    )
                    idx = rng.permutation(n_pts)[:max_pts]
                    x_query = x_query[idx]
                    target = target[idx]
                    n_pts = max_pts

                pad_query = np.zeros((max_pts, 1), dtype=bool)
                pad_query[:n_pts] = True
                if n_pts < max_pts:
                    pad_x = np.zeros(
                        (max_pts - n_pts, x_query.shape[1]), dtype=x_query.dtype
                    )
                    pad_t = np.zeros(
                        (max_pts - n_pts, target.shape[1]), dtype=target.dtype
                    )
                    x_query = np.concatenate([x_query, pad_x], axis=0)
                    target = np.concatenate([target, pad_t], axis=0)

                self.cache.append(
                    dict(
                        case_id=cid,
                        x_query=x_query.astype(self.dtype, copy=False),
                        x_kv=x_kv.astype(self.dtype, copy=False),
                        op_params=op_params.astype(self.dtype, copy=False),
                        geo_params=geo_params.astype(self.dtype, copy=False),
                        target=target.astype(self.dtype, copy=False),
                        pad_query=pad_query,
                    )
                )
        if len(self.cache) != len(self.case_ids):
            raise RuntimeError(
                f"Loaded {len(self.cache)} cases but manifest has {len(self.case_ids)} "
                f"for task={self.task}; some IDs may be missing from the h5"
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> dict:
        r"""Return one case (as a dict) with optional intra-case point subsampling.

        Subsampling: if ``nValid >= subsample_points``, sample ``subsample_points`` from
        valid indices without replacement; if ``nValid < subsample_points``,
        take all valid indices and top up with random pad-slot indices.
        The "top up" branch matters when a case has fewer valid points than
        the requested subsample size; do not simplify away.
        """
        c = self.cache[idx]
        x_query = c["x_query"]
        target = c["target"]
        pad_query = c["pad_query"]

        n_max = int(self.max_num_points)
        n_valid = int(pad_query.sum())

        if self.subsample_points is not None and n_max > self.subsample_points:
            k = int(self.subsample_points)
            # NB: uses NumPy's default RNG; DataLoader workers get a per-worker
            # seed via `worker_init_fn` in the training script (see utils.py).
            if n_valid >= k:
                sel = np.random.choice(n_valid, k, replace=False)
                idx_points = np.arange(n_max)[pad_query[:, 0]][sel]
            else:
                # Take all valid + top up from pad slots.
                valid_idx = np.arange(n_max)[pad_query[:, 0]]
                pad_slots = np.arange(n_max)[~pad_query[:, 0]]
                topup = np.random.choice(pad_slots, k - n_valid, replace=False)
                idx_points = np.concatenate([valid_idx, topup], axis=0)
            x_query = x_query[idx_points]
            target = target[idx_points]
            pad_query = pad_query[idx_points]

        # Return as torch tensors, ready for the default collate.
        return {
            "x_query": torch.from_numpy(np.ascontiguousarray(x_query)),
            "x_kv": torch.from_numpy(np.ascontiguousarray(c["x_kv"])),
            "op_params": torch.from_numpy(np.ascontiguousarray(c["op_params"])),
            "geo_params": torch.from_numpy(np.ascontiguousarray(c["geo_params"])),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "pad_query": torch.from_numpy(np.ascontiguousarray(pad_query)),
            "case_id": int(c["case_id"]),
        }
