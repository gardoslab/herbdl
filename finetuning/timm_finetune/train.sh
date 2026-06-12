#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2026

conda activate spring-2026-pyt

# CONFIG_FILE must be provided (e.g. via `qsub -v CONFIG_FILE=...`, as submit.sh does).
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE is not set. Pass it explicitly, e.g.:" >&2
    echo "  qsub -v CONFIG_FILE=configs/convnext_large_inat_384_2gpu.yml ... train.sh" >&2
    echo "  (or use submit.sh, which sets it for you)" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE '$CONFIG_FILE' not found (cwd: $(pwd))." >&2
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
[ -n "$SET_ARGS" ] && echo "Overrides: $SET_ARGS"

# Multi-GPU: set NPROC_PER_NODE=<n> in the qsub -v args to launch with torchrun (DDP).
NPROC=${NPROC_PER_NODE:-1}
if [ "$NPROC" -gt 1 ]; then
    echo "Launching with torchrun --nproc_per_node=$NPROC"
    torchrun --nproc_per_node=$NPROC --standalone \
        train.py --config $CONFIG_FILE ${SET_ARGS}
else
    python train.py --config $CONFIG_FILE ${SET_ARGS}
fi

# Example qsub (2-GPU DDP):
# qsub -l h_rt=24:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=48G \
#      -v CONFIG_FILE=configs/convnext_large_inat_384_2gpu.yml,NPROC_PER_NODE=2 \
#      -m beas -M you@bu.edu -N CNXL_INAT train.sh
