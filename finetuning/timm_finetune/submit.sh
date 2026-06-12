#!/bin/bash
# Seed-ensemble launcher for the timm_finetune module. Mirrors SWIN/submit_concrete.sh.
# Defaults to the ConvNeXt-L iNat21 @384 run.
#
# Key env vars (all optional):
#   CONFIG      — config file path   (default: configs/convnext_large_inat_384_2gpu.yml)
#   RUN_PREFIX  — base name for output dirs + W&B run id/name (default: CONVNEXT_L_INAT_384)
#   OUT_BASE    — output root (default: <this checkout>/finetuning/output/timm)
#   SEEDS       — space-separated    (default: "0")
#   NGPUS       — GPUs per job       (default: 2; triggers torchrun DDP when >1)
#   GPU_MEM     — GPU memory request (default: 48G; these backbones fit 48G at batch 16)
#   H_RT        — wall-time limit    (default: 24:00:00; herbdl GPU queues cap at 24h)
#   EMAIL       — notification email (default: tgardos@bu.edu)
#
# Examples:
#   bash submit.sh                                            # ConvNeXt-L, seed 0
#   CONFIG=configs/eva02_large_inat_336_2gpu.yml RUN_PREFIX=EVA02_L_INAT_336 bash submit.sh
#   SEEDS="0 1 2" bash submit.sh
#
# Nothing is auto-submitted by Claude — run this yourself when ready.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEEDS=${SEEDS:-"0"}
CONFIG=${CONFIG:-"configs/convnext_large_inat_384_2gpu.yml"}
RUN_PREFIX=${RUN_PREFIX:-"CONVNEXT_L_INAT_384"}
OUT_BASE=${OUT_BASE:-"${REPO_ROOT}/finetuning/output/timm"}
NGPUS=${NGPUS:-2}
GPU_MEM=${GPU_MEM:-"48G"}
EMAIL=${EMAIL:-"tgardos@bu.edu"}
H_RT=${H_RT:-"24:00:00"}

OMP_THREADS=$(( NGPUS * 8 ))
QSUB_ARGS="-l h_rt=${H_RT} -pe omp ${OMP_THREADS} -P herbdl -l gpus=${NGPUS} -l gpu_c=8.0 -l gpu_memory=${GPU_MEM} -m beas -M ${EMAIL}"

NPROC_VAR=""
[ "$NGPUS" -gt 1 ] && NPROC_VAR=",NPROC_PER_NODE=${NGPUS}"

JOB_PREFIX=$(echo "$RUN_PREFIX" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9' | cut -c1-8)

for seed in $SEEDS; do
    RUN_ID=$(echo "${RUN_PREFIX}_seed${seed}" | tr '[:upper:]' '[:lower:]')
    RUN_NAME="${RUN_PREFIX}_Seed${seed}"
    OUT="${OUT_BASE}/${RUN_PREFIX}_SEED${seed}"

    SET_ARGS="--set training.seed=${seed} --set training.output_dir=${OUT} --set training.logging_dir=${OUT} --set custom.run_id=${RUN_ID} --set custom.run_name=${RUN_NAME}"

    JOB=$(qsub $QSUB_ARGS \
        -N "${JOB_PREFIX}_S${seed}" \
        -v CONFIG_FILE="${CONFIG}",SET_ARGS="${SET_ARGS}"${NPROC_VAR} \
        train.sh | grep -oP '(?<=job )\d+')
    echo "Submitted seed ${seed}: job ${JOB}  ->  ${OUT}"
done

echo
echo "Monitor with: qstat -u \$USER"
