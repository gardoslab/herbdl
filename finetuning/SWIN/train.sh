#!/bin/bash -l
#$ -N SWIN_TRAINING

module load miniconda
module load academic-ml/fall-2025

#set env variable
export LR_TYPE="cosine"
export FROZEN="true"

# v1 - freeze all except last linear layer, # v2 - all layers except last transformer block and linear layer, v3 - all layers except last 2 transformer blocks and linear layer, v4 - all layers except last 3 transformer blocks and linear layer
export RUN_GROUP="SWIN_frozen_v3"
export FROZEN_TYPE="v3" 
export RUN_NAME="SWIN_v3_full_kaggle"
export RUN_ID="swin_fr_v3_101125"

conda activate herb_env

save_dir="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/KAGGLE"

python SWIN_finetuning.py \
    --output_dir  $save_dir \
    --logging_dir $save_dir \
    --model_name_or_path "microsoft/swin-base-patch4-window7-224-in22k"\
    --train_file "/projectnb/herbdl/data/kaggle-herbaria/train.json" \
    --validation_file "/projectnb/herbdl/data/kaggle-herbaria/val.json" \
    --image_column_name filepath \
    --label_column_name scientificNameEncoded \
    --max_seq_length=15 \
    --num_train_epochs=10 \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=128 \
    --warmup_steps=500 \
    --learning_rate=0.005 --save_strategy="epoch" --save_total_limit=5  --eval_strategy="steps" --eval_steps=8964 \
    --lr_scheduler_type=$LR_TYPE \
    --ignore_mismatched_sizes --report_to="wandb" \
    --bf16

# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -m beas -M faridkar@bu.edu train.sh
