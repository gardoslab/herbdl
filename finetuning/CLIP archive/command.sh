#!/bin/bash -l

#$ -l gpus=1
#$ -l gpu_c=7.0

# Directory to search
dir="/projectnb/herbdl/workspaces/smritis/finetuning/output/retraining"

# Find the most recent file
rc=$(ls -t "$dir"/* | head -n 1)

recent_file=${rc::-1}

var=$(jq '.log_history[-1].learning_rate' "$recent_file/trainer_state.json");

if [[ -n $var ]] 
then 
    lr=$(jq '.log_history[-2].learning_rate' "$recent_file/trainer_state.json");
else
    lr=$(jq '.log_history[-1].learning_rate' "$recent_file/trainer_state.json");
fi

python CLIP_finetuning.py \
    --output_dir $dir \
    --resume_from_checkpoint $recent_file \
    --model_name_or_path "openai/clip-vit-large-patch14-336" \
    --train_file "/projectnb/herbdl/workspaces/smritis/finetuning/training/pairs.json" \
    --image_column image \
    --overwrite_output_dir=True \
    --max_seq_length=14 \
    --num_train_epochs=3 \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --learning_rate=$lr --warmup_steps="0" --weight_decay 0.1 --save_strategy="epoch" --save_total_limit=5 --evaluation_strategy="epoch" --save_total_limit=5


