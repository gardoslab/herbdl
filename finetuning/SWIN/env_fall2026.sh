# source this before using the fall-2026-pyt shared conda env, so `pip install --user`
# and pip's download cache land in the project dir instead of $HOME (10GB quota).
module load academic-ml/fall-2026
# fall-2026-pyt's torch is built against CUDA 13.2 but doesn't ship CUDA_HOME itself;
# without this, anything that needs to build/JIT CUDA extensions (deepspeed, flash-attn,
# etc.) fails with a missing-CUDA_HOME error even though torch.cuda.is_available() is True.
module load cuda/13.2
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip
