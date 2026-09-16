#!/bin/bash -l
#
# Fixed run contract for the label-cardinality line. Child nodes vary
# configs/scaling.yml (and the code); this command stays byte-identical.
set -euo pipefail

module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt

# fall-2026-pyt is a read-only shared env; extras live in a project-dir
# `pip install --user`, not $HOME (10 GB quota). Same convention as
# finetuning/SWIN/train_advanced.sh.
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd "$(dirname "$0")"
echo "host=$(hostname)  cwd=$(pwd)  commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

python sweep.py --config configs/scaling.yml
