#!/bin/bash
# Submit pretrained curriculum pipeline (S1→S2→S3).
# Starts from ImageNet-22k pretrained weights instead of the no-aug baseline checkpoint.
# Each stage waits for the previous to complete.
# Set START_STAGE to skip already-completed stages (1-3).
#
# Usage:
#   bash submit_curriculum_pretrained.sh          # run all 3 stages
#   START_STAGE=2 bash submit_curriculum_pretrained.sh  # start from S2

START_STAGE=${START_STAGE:-1}

QSUB_COMMON="-l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -m beas -M faridkar@bu.edu"

PREV_JOB=""

submit_stage() {
    local STAGE_NUM=$1
    local JOB_NAME=$2
    local CONFIG=$3

    if [ "$STAGE_NUM" -lt "$START_STAGE" ]; then
        echo "Skipping Stage $STAGE_NUM ($JOB_NAME) — START_STAGE=$START_STAGE"
        return
    fi

    if [ -z "$PREV_JOB" ]; then
        JOB=$(qsub $QSUB_COMMON \
            -N "$JOB_NAME" \
            -v CONFIG_FILE="$CONFIG" \
            train_advanced.sh | grep -oP '(?<=job )\d+')
        echo "Submitted Stage $STAGE_NUM ($JOB_NAME): job $JOB"
    else
        JOB=$(qsub $QSUB_COMMON \
            -N "$JOB_NAME" \
            -hold_jid "$PREV_JOB" \
            -v CONFIG_FILE="$CONFIG" \
            train_advanced.sh | grep -oP '(?<=job )\d+')
        echo "Submitted Stage $STAGE_NUM ($JOB_NAME): job $JOB (holds on $PREV_JOB)"
    fi
    PREV_JOB=$JOB
}

# Stage 1 — Mild augmentation from ImageNet-22k pretrained (LR=1e-4, 50 epochs)
submit_stage 1 PRETRAINED_S1  configs_advanced/swin_curriculum_pretrained_s1.yml

# Stage 2 — Medium augmentation
submit_stage 2 PRETRAINED_S2  configs_advanced/swin_curriculum_pretrained_s2.yml

# Stage 3 — Heavy augmentation
submit_stage 3 PRETRAINED_S3  configs_advanced/swin_curriculum_pretrained_s3.yml

echo ""
echo "Pretrained curriculum pipeline submitted."
echo "Monitor with: qstat -u faridkar"
