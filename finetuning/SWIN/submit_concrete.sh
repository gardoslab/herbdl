#!/bin/bash
# General-purpose seed-ensemble launcher for any advanced config.
# Defaults to the concrete SWIN-L 384 run for backward compatibility.
#
# Key env vars (all optional):
#   CONFIG      — config file path   (default: swin_large_384_concrete.yml)
#   RUN_PREFIX  — base name used for output dirs and W&B run id/name
#                 (default: SWIN_L_384_CONCRETE)
#   OUT_BASE    — output root        (default: .../workspaces/faridkar/.../SWIN)
#   SEEDS       — space-separated    (default: "0 1 2")
#   NGPUS       — GPUs per job       (default: 1; set to 2 for multi-GPU / DDP)
#   GPU_MEM     — GPU memory request (default: 80G)
#   H_RT        — wall-time limit     (default: 24:00:00)
#                 NOTE: project herbdl's GPU access caps at 24h (cds-gpu-long) / 12h
#                 (public buy-in queues). Requesting >24h locks the job out of those
#                 queues, so it waits behind the scarce shared long-GPU pool. Keep <=24h
#                 and rely on per-epoch checkpoint + auto-resume across jobs.
#   CKPT        — warm-start checkpoint dir (overrides model.model_name_or_path)
#   EMAIL       — notification email (default: faridkar@bu.edu)
#
# Usage examples:
#   bash submit_concrete.sh                            # concrete run, seeds 0 1 2 (defaults)
#   SEEDS="0 1 2 3 4" bash submit_concrete.sh          # concrete run, 5 seeds
#
#   CONFIG=configs_advanced/swinv2_large_192_heavy_multitask.yml \
#   RUN_PREFIX=SWINV2_L_192_HEAVY_MT \
#   SEEDS="0 1 2 3 4" \
#   NGPUS=2 \
#       bash submit_concrete.sh
#
# Nothing is auto-submitted by Claude — run this yourself when ready.

# Default OUT_BASE to *this checkout's* output dir, derived from the script location
# (.../<workspace>/herbdl/finetuning/SWIN/submit_concrete.sh -> .../herbdl/finetuning/output/SWIN),
# so runs land in the submitter's own workspace regardless of who authored the config.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SEEDS=${SEEDS:-"0 1 2"}
CONFIG=${CONFIG:-"configs_advanced/swin_large_384_concrete.yml"}
RUN_PREFIX=${RUN_PREFIX:-"SWIN_L_384_CONCRETE"}
OUT_BASE=${OUT_BASE:-"${REPO_ROOT}/finetuning/output/SWIN"}
NGPUS=${NGPUS:-1}
GPU_MEM=${GPU_MEM:-"80G"}
EMAIL=${EMAIL:-"faridkar@bu.edu"}
# 24h (not 48h): herbdl's GPU queues cap at 24h (cds-gpu-long) / 12h (public); a >24h
# request can't use them and queues behind the scarce shared long-GPU pool. Checkpoints
# are saved per epoch and the trainer auto-resumes, so multi-job runs are seamless.
H_RT=${H_RT:-"24:00:00"}

OMP_THREADS=$(( NGPUS * 8 ))
QSUB_ARGS="-l h_rt=${H_RT} -pe omp ${OMP_THREADS} -P herbdl -l gpus=${NGPUS} -l gpu_c=8.0 -l gpu_memory=${GPU_MEM} -m beas -M ${EMAIL}"

# Pass NPROC_PER_NODE only when using multiple GPUs (triggers torchrun in train_advanced.sh)
NPROC_VAR=""
[ "$NGPUS" -gt 1 ] && NPROC_VAR=",NPROC_PER_NODE=${NGPUS}"

# Derive a short job-name prefix (first 8 chars, uppercase, no special chars)
JOB_PREFIX=$(echo "$RUN_PREFIX" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9' | cut -c1-8)

for seed in $SEEDS; do
    RUN_ID=$(echo "${RUN_PREFIX}_seed${seed}" | tr '[:upper:]' '[:lower:]')
    RUN_NAME="${RUN_PREFIX}_Seed${seed}"
    OUT="${OUT_BASE}/${RUN_PREFIX}_SEED${seed}"

    SET_ARGS="--set training.seed=${seed} --set training.output_dir=${OUT} --set training.logging_dir=${OUT} --set custom.run_id=${RUN_ID} --set custom.run_name=${RUN_NAME}"
    if [ -n "$CKPT" ]; then
        SET_ARGS="${SET_ARGS} --set model.model_name_or_path=${CKPT}"
    fi

    JOB=$(qsub $QSUB_ARGS \
        -N "${JOB_PREFIX}_S${seed}" \
        -v CONFIG_FILE="${CONFIG}",SET_ARGS="${SET_ARGS}"${NPROC_VAR} \
        train_advanced.sh | grep -oP '(?<=job )\d+')
    echo "Submitted seed ${seed}: job ${JOB}  ->  ${OUT}"
done

echo
echo "Monitor with: qstat -u \$USER"
