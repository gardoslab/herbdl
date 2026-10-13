#!/bin/bash -l
#
# Fixed run contract for the loss-registry E2E harness. Runs every
# configuration in e2e_harness.py's registry for a few hundred real steps on
# a small real subset of Kaggle-Herbarium-2022 and exits nonzero if any fails.

# Deliberately NO `set -e` around the module loads -- see train_advanced.sh
# and repgeom/run_repgeom.sh for why (SCC's mamba.sh returns non-zero on a
# harmless `export mamba` line and would otherwise kill the job before
# anything of ours runs).
module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd "$(dirname "$0")/.." || exit 1  # finetuning/SWIN, so `import loss_ablation` resolves

echo "host=$(hostname)  cwd=$(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

python - <<'PY' || exit 1
import importlib, sys
missing = [m for m in ("torch", "transformers", "PIL") if not importlib.util.find_spec(m)]
if missing:
    print("ERROR: missing modules in fall-2026-pyt: %s" % missing, file=sys.stderr)
    sys.exit(1)
import torch
print("torch=%s cuda_available=%s" % (torch.__version__, torch.cuda.is_available()))
PY

# From here on, a real failure should fail the job -- but the harness script
# itself decides pass/fail per-config and exits 1 only if something failed, so
# we do not set -e (a single config's controlled failure should still let the
# rest run and be reported, not abort the whole harness).
python loss_ablation/e2e_harness.py "$@"
