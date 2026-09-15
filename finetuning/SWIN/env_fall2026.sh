# source this before using the fall-2026-pyt shared conda env, so `pip install --user`
# and pip's download cache land in the project dir instead of $HOME (10GB quota).
module load academic-ml/fall-2026
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip
