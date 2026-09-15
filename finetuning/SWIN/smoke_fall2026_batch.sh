#!/bin/bash -l
#$ -l h_rt=00:20:00
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -P herbdl
#$ -N SMOKE_FALL2026_BATCH
#$ -j y
#$ -o /projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/SWIN/SMOKE_FALL2026_batch.log

module load miniconda
module load academic-ml/fall-2026
module load cuda/13.2
conda activate fall-2026-pyt

export PYTHONUSERBASE=/projectnb/herbdl/workspaces/faridkar/.local-fall2026-pyt
export PIP_CACHE_DIR=/projectnb/herbdl/workspaces/faridkar/.cache/pip

cd /projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/SWIN

python SWIN_finetuning_advanced.py --config configs_advanced/swin_large_384_allin.yml \
    --set data.max_train_samples=200 \
    --set data.max_eval_samples=200 \
    --set training.num_train_epochs=1 \
    --set training.output_dir=/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/SMOKE_FALL2026_BATCH \
    --set training.overwrite_output_dir=true \
    --set wandb.enabled=false
