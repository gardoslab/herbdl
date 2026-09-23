#!/bin/bash -l
# Wrapper mirroring train_advanced.sh's own env-setup structure, deliberately NOT a
# top-level `&&`-chained orx run command: `module load miniconda` sources mamba.sh,
# which emits a handful of harmless "export: mamba: not a function" warnings (real
# nonzero exit statuses from sub-commands, just not fatal ones) -- fine inside a plain
# script with no `set -e`, but if it's one link in a top-level `a && b && c` run
# command, orx's job wrapper aborts the whole chain right there before ever reaching
# python. Caught by running this exact chain first as a top-level command and seeing
# it die silently after only the two warning lines.

module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

python loss_ablation/hace_tree_smoke.py
