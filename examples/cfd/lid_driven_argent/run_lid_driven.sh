#!/usr/bin/env bash
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

# Train ArGEnT on the lid-driven cavity data with DDP over every GPU on one
# node (e.g. 8x A100), then evaluate the best checkpoint on the test and
# half-circle extrapolation sets.
#
# Usage:
#   ./run_lid_driven.sh [hydra overrides for train.py ...]
#
# Environment variables (all optional):
#   NGPUS          number of ranks (default: all GPUs reported by nvidia-smi)
#   MODEL_NAME     run tag; outputs go to ./outputs/$MODEL_NAME
#   OUT_DIR        output directory shared by train and pred (overrides MODEL_NAME)
#   GLOBAL_BATCH   cases per optimizer step summed over ranks (default 32,
#                  the single-GPU batch); split evenly across ranks
#   VENV           virtual environment to create/use (default ./venv); built
#                  with uv, which is installed to ~/.local/bin if missing
#   PYTHON_VERSION Python for the venv (default 3.12; uv downloads it)
#   STAGES         subset of "setup train pred" to run (default: all three)
#   LIDDRIVEN_TRAIN_H5, LIDDRIVEN_HALFCIRCLE_H5
#                  dataset paths (default ./data/...; see README > Data)
#
# Examples:
#   ./run_lid_driven.sh
#   MODEL_NAME=lid_v1 ./run_lid_driven.sh training.max_epochs=100
#   MODEL_NAME=lid_v1 ./run_lid_driven.sh training.resume=true
#   MODEL_NAME=lid_v1 STAGES=pred ./run_lid_driven.sh

set -euo pipefail
cd "$(dirname "$0")"

MODEL_NAME=${MODEL_NAME:-lid_driven_argent_cross}
OUT_DIR=${OUT_DIR:-./outputs/${MODEL_NAME}}
GLOBAL_BATCH=${GLOBAL_BATCH:-32}
VENV=${VENV:-./venv}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
STAGES=${STAGES:-"setup train pred"}
export LIDDRIVEN_TRAIN_H5=${LIDDRIVEN_TRAIN_H5:-./data/LidDriven_dataset_forTrain.h5}
export LIDDRIVEN_HALFCIRCLE_H5=${LIDDRIVEN_HALFCIRCLE_H5:-./data/LidDriven_dataset_halfCircle_forExtrapolation.h5}

die() {
    echo "run_lid_driven.sh: $*" >&2
    exit 1
}

has_stage() {
    [[ " ${STAGES} " == *" $1 "* ]]
}

# The dataset reads a sibling <name>_manifest.json next to each h5 file.
for h5 in "${LIDDRIVEN_TRAIN_H5}" "${LIDDRIVEN_HALFCIRCLE_H5}"; do
    for f in "${h5}" "${h5%.h5}_manifest.json"; do
        [[ -f "${f}" ]] || die "missing ${f} -- fetch the LidDriven data into ./data/ (see README > Data)"
    done
done

if has_stage setup; then
    if ! "${VENV}/bin/python" -c "import physicsnemo, pyDOE, h5py" 2>/dev/null; then
        echo "==> Creating ${VENV} and installing physicsnemo + example requirements"
        # uv fetches its own Python, so this works on images whose system
        # python3 is older than physicsnemo's minimum (3.11) or lacks
        # python3-venv/ensurepip (e.g. stock Ubuntu 22.04 cloud VMs).
        if ! command -v uv >/dev/null; then
            echo "==> Installing uv"
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="${HOME}/.local/bin:${PATH}"
        fi
        uv venv --python "${PYTHON_VERSION}" "${VENV}"
        # Match the torch wheel's CUDA major to the driver; a bare PyPI torch
        # may target a newer CUDA than the driver supports.
        CUDA_MAJOR=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -n1)
        if [[ -z "${CUDA_MAJOR}" ]]; then
            echo "==> nvidia-smi not found; installing torch from PyPI"
        elif (( CUDA_MAJOR >= 13 )); then
            uv pip install --python "${VENV}/bin/python" torch torchvision \
                --index-url https://download.pytorch.org/whl/cu130
        elif (( CUDA_MAJOR == 12 )); then
            uv pip install --python "${VENV}/bin/python" torch torchvision \
                --index-url https://download.pytorch.org/whl/cu129
        else
            die "unsupported CUDA driver version ${CUDA_MAJOR} (need >= 12)"
        fi
        uv pip install --python "${VENV}/bin/python" -e ../../.. -r requirements.txt
    else
        echo "==> Reusing ${VENV}"
    fi
fi
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

if has_stage train; then
    NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}
    (( NGPUS > 0 )) || die "no GPUs found; set NGPUS explicitly"
    (( GLOBAL_BATCH % NGPUS == 0 )) || die "GLOBAL_BATCH=${GLOBAL_BATCH} is not divisible by NGPUS=${NGPUS}"
    echo "==> Training on ${NGPUS} rank(s), $(( GLOBAL_BATCH / NGPUS )) case(s) per rank -> ${OUT_DIR}"
    # Single node: rendezvous over loopback so it works even when the
    # hostname does not resolve (common on cloud VMs and laptops).
    torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node="${NGPUS}" train.py \
        training.output_dir="${OUT_DIR}" \
        data.batch_size_case=$(( GLOBAL_BATCH / NGPUS )) \
        "$@"
fi

if has_stage pred; then
    for dataset in main halfcircle; do
        echo "==> Inference on ${dataset}"
        python inference.py \
            checkpoint_dir="${OUT_DIR}/checkpoint_best" \
            inference.dataset="${dataset}" \
            inference.output_dir="${OUT_DIR}/inference"
    done
fi
