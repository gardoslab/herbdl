#!/bin/bash -l
# Near-duplicate check (1 GPU). Same env as train_advanced.sh.
module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip
python loss_ablation/near_duplicates.py
