#!/bin/bash -l
#
# Fixed run contract for the label-cardinality line. Child nodes vary
# configs/scaling.yml (and the code); this command stays byte-identical.

# Deliberately NO `set -e` around the module loads. SCC's miniconda profile
# script (mamba.sh) runs `export mamba` on something that is not a function and
# returns non-zero, so `set -e` kills the job before anything of ours runs.
# finetuning/SWIN/train_advanced.sh avoids this the same way.
module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt

# fall-2026-pyt is a read-only shared env; extras live in a project-dir
# `pip install --user`, not $HOME (10 GB quota). Same convention as
# finetuning/SWIN/train_advanced.sh.
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd "$(dirname "$0")" || exit 1

echo "host=$(hostname)  cwd=$(pwd)"
echo "python=$(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# Fail fast and legibly if the shared env is missing something, rather than
# dying halfway through a multi-hour feature extraction.
python - <<'PY' || exit 1
import importlib, sys
missing = [m for m in ("torch", "torchvision", "transformers", "numpy", "pandas", "yaml", "PIL")
           if not importlib.util.find_spec(m)]
if missing:
    print("ERROR: missing modules in fall-2026-pyt: %s" % missing, file=sys.stderr)
    sys.exit(1)
import torch
print("torch=%s cuda_available=%s" % (torch.__version__, torch.cuda.is_available()))
PY

# From here on, a failure is a real failure.
set -euo pipefail
python sweep.py --config configs/scaling.yml
