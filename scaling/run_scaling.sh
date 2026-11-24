#!/bin/bash -l
#
# Fixed run contract for the label-cardinality line: `cd scaling && bash
# run_scaling.sh`. This node is a CPU-only data census, so the script body runs
# census_2021.py instead of sweep.py. The command itself is unchanged.

# Deliberately NO `set -e` around the module loads. SCC's miniconda profile
# script (mamba.sh) runs `export mamba` on something that is not a function and
# returns non-zero, so `set -e` kills the job before anything of ours runs.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt

# fall-2026-pyt is a read-only shared env; extras live in a project-dir
# `pip install --user`, not $HOME (10 GB quota).
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd "$(dirname "$0")" || exit 1

echo "host=$(hostname)  cwd=$(pwd)"
echo "python=$(command -v python)"

# No cuda module and no nvidia-smi check: this node reads CSVs and stats files.
# It needs pandas and numpy only, so fail fast and legibly if they are missing.
python - <<'PY' || exit 1
import importlib, sys
missing = [m for m in ("numpy", "pandas") if not importlib.util.find_spec(m)]
if missing:
    print("ERROR: missing modules in fall-2026-pyt: %s" % missing, file=sys.stderr)
    sys.exit(1)
PY

# From here on, a failure is a real failure.
set -euo pipefail
python census_2021.py
