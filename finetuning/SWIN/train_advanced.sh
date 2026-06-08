#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2026

conda activate spring-2026-pyt

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

# Example qsub command for multi-GPU training:
# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=80G \
#      -v NPROC_PER_NODE=2 -m beas -M faridkar@bu.edu -N SWIN_MULTIGPU train_advanced.sh
