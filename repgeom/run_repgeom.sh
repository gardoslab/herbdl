#!/bin/bash -l
#
# Fixed run contract for the representation-geometry line. Child nodes vary
# configs/repgeom.yml and the code; this command stays byte-identical.

# Deliberately NO `set -e` around the module loads. SCC's miniconda profile
# script (mamba.sh) runs `export mamba` on something that is not a function and
# returns non-zero, so `set -e` kills the job before anything of ours runs.
# finetuning/SWIN/env_fall2026.sh and scaling/run_scaling.sh do the same.
module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt

# fall-2026-pyt is a read-only shared env; extras go to a project-dir
# `pip install --user`, never $HOME (10 GB quota).
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd "$(dirname "$0")" || exit 1

echo "host=$(hostname)  cwd=$(pwd)"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# Fail fast and legibly if the shared env is missing something, rather than
# dying after 168k images have already been through the GPU.
python - <<'PY' || exit 1
import importlib, sys
missing = [m for m in ("torch", "transformers", "numpy", "pandas", "yaml", "PIL")
           if not importlib.util.find_spec(m)]
if missing:
    print("ERROR: missing modules in fall-2026-pyt: %s" % missing, file=sys.stderr)
    sys.exit(1)
import torch
print("torch=%s cuda_available=%s" % (torch.__version__, torch.cuda.is_available()))
PY

# From here on, a failure is a real failure.
set -euo pipefail
python analyze.py --config configs/repgeom.yml
