#!/bin/bash
# Submit full curriculum pipeline (S1→S2→S3→S3_cont→MultiTask→ArcFace→Hybrid→384→V2).
# Each stage waits for the previous to complete successfully.
# Set START_STAGE to skip already-completed stages (1-9).
#
# Usage:
#   bash submit_curriculum.sh          # run all 9 stages
#   START_STAGE=5 bash submit_curriculum.sh  # start from MultiTask

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

# Stage 1 — Mild augmentation from baseline
submit_stage 1 CURRIC_S1      configs_advanced/swin_curriculum_s1.yml

# Stage 2 — Medium augmentation
submit_stage 2 CURRIC_S2      configs_advanced/swin_curriculum_s2.yml

# Stage 3 — Heavy augmentation
submit_stage 3 CURRIC_S3      configs_advanced/swin_curriculum_s3.yml

# Stage 4 — Heavy augmentation continued
submit_stage 4 CURRIC_S3_CONT configs_advanced/swin_curriculum_s3_cont.yml

# Stage 5 — Multi-task (family/genus/species CE heads)
submit_stage 5 CURRIC_MT      configs_advanced/swin_curriculum_multitask.yml

# Stage 6 — SubCenter ArcFace + MultiTask
submit_stage 6 CURRIC_AF      configs_advanced/swin_curriculum_arcface.yml

# Stage 7 — Hybrid loss (ArcFace 0.7 + CE 0.3)
submit_stage 7 CURRIC_HYB     configs_advanced/swin_curriculum_hybrid.yml

# Stage 8 — 384 resolution (backbone from Hybrid, 384 arch)
submit_stage 8 CURRIC_384     configs_advanced/swin_curriculum_384.yml

# Stage 9 — SWIN V2 with all techniques (fresh from ImageNet22k V2 pretrain)
submit_stage 9 CURRIC_V2      configs_advanced/swin_curriculum_v2.yml

echo ""
echo "Full curriculum pipeline submitted."
echo "Monitor with: qstat -u faridkar"
