#!/bin/bash -l
# Run one loss-ablation analysis script on SCC with the shared environment.
# Usage (from finetuning/SWIN):  bash loss_ablation/run.sh <script.py> [args...]
#   e.g. bash loss_ablation/run.sh paper_measures.py
# Training and eval-only scoring go through train_advanced.sh instead:
#   HERBDL_NO_RELOCATE=1 [NPROC_PER_NODE=2] CONFIG_FILE=configs_advanced/<config>.yml bash train_advanced.sh
set -e
SCRIPT=${1:?usage: bash loss_ablation/run.sh <script.py> [args...]}
shift
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python "loss_ablation/${SCRIPT}" "$@"
