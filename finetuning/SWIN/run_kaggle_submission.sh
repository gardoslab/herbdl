#!/bin/bash -l
#$ -l h_rt=12:00:00
#$ -pe omp 8
#$ -P herbdl
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -m beas
#$ -M faridkar@bu.edu
#$ -N KAGGLE_SUBMISSION

module load miniconda
module load academic-ml/fall-2025
conda activate herb_env

CHECKPOINT=${CHECKPOINT:-"/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/SWIN_CURRICULUM_S3_CONT"}
OUTPUT=${OUTPUT:-"submission.csv"}
MULTITASK_FLAG=${MULTITASK_FLAG:-""}

echo "Checkpoint : $CHECKPOINT"
echo "Output     : $OUTPUT"

python kaggle_submission.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --batch_size 128 \
    --num_workers 8 \
    $MULTITASK_FLAG
