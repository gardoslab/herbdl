#!/bin/bash -l
# Same structure as run_hace_tree_smoke.sh: env setup lives inside this script's
# own body (no `set -e` propagation issue), not chained at the top level of an
# orx run command -- see that script's comment for the exact failure this avoids.

module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

python loss_ablation/loss_registry_smoke.py
