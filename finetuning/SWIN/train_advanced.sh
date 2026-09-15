#!/bin/bash -l

module load miniconda
module load academic-ml/fall-2026
# fall-2026-pyt's torch is built against CUDA 13.2 but doesn't ship CUDA_HOME itself;
# without this, anything that needs to build/JIT CUDA extensions (deepspeed, flash-attn,
# etc.) fails with a missing-CUDA_HOME error even though torch.cuda.is_available() is True.
module load cuda/13.2
conda activate fall-2026-pyt

# fall-2026-pyt is a read-only shared env, so extra packages (evaluate, wandb,
# pytorch_metric_learning) live in a project-dir `pip install --user`, not $HOME
# (10GB quota). See finetuning/CLAUDE.md for details, and for herb_env as a fallback
# if fall-2026-pyt ever breaks or gets retired at semester turnover.
export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

# CONFIG_FILE must be provided (e.g. via `qsub -v CONFIG_FILE=...`, as submit_concrete.sh
# does). Fail fast rather than silently running an arbitrary default config.
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE is not set. Pass it explicitly, e.g.:" >&2
    echo "  qsub -v CONFIG_FILE=configs_advanced/swin_large_384_concrete.yml ... train_advanced.sh" >&2
    echo "  (or use submit_concrete.sh, which sets it for you)" >&2
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE '$CONFIG_FILE' not found (cwd: $(pwd))." >&2
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
[ -n "$SET_ARGS" ] && echo "Overrides: $SET_ARGS"

# Multi-GPU: set NPROC_PER_NODE=<n> in the qsub -v args to launch with torchrun (DDP).
# Single GPU (default): plain python.
NPROC=${NPROC_PER_NODE:-1}
if [ "$NPROC" -gt 1 ]; then
    echo "Launching with torchrun --nproc_per_node=$NPROC"
    torchrun --nproc_per_node=$NPROC --standalone \
        SWIN_finetuning_advanced.py --config $CONFIG_FILE ${SET_ARGS}
else
    python SWIN_finetuning_advanced.py --config $CONFIG_FILE ${SET_ARGS}
fi

# Example qsub commands:
#
# Single-GPU:
#   qsub -l h_rt=24:00:00 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=48G \
#        -v CONFIG_FILE=configs_advanced/swin_baseline_augmented.yml \
#        -m beas -M faridkar@bu.edu -N SWIN_BASELINE train_advanced.sh
#
# Multi-GPU (DDP via torchrun):
#   qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=80G \
#        -v NPROC_PER_NODE=2,CONFIG_FILE=configs_advanced/swin_large_384_concrete.yml \
#        -m beas -M faridkar@bu.edu -N SWIN_MULTIGPU train_advanced.sh
