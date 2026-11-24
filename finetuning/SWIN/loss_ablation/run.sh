#!/bin/bash -l
# Run one loss-ablation analysis script on SCC with the shared environment.
# Usage (from finetuning/SWIN):  bash loss_ablation/run.sh <script.py> [args...]
#   e.g. bash loss_ablation/run.sh geometry.py base --splits val
# Training and eval-only scoring go through train_advanced.sh instead:
#   HERBDL_NO_RELOCATE=1 [NPROC_PER_NODE=2] CONFIG_FILE=configs_advanced/<config>.yml bash train_advanced.sh
SCRIPT=${1:?usage: bash loss_ablation/run.sh <script.py> [args...]}
shift
# No `set -e` around the module loads: miniconda's mamba.sh returns non-zero in batch shells
# ("export: mamba: not a function"), which would end the job before Python starts.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt || exit 1
exec python "loss_ablation/analysis/${SCRIPT}" "$@"
