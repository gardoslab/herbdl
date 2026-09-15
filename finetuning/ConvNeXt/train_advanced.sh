#!/bin/bash -l

module load miniconda
module load academic-ml/fall-2025
conda activate /projectnb/herbdl/workspaces/llindon/.conda/herb_env

# Path to config file - can be set via environment variable or use default
# If CONFIG_FILE is not set (e.g., via qsub -v), use default
CONFIG_FILE="${CONFIG_FILE:-configs/convnext_base_224_unfrozen.yml}"

echo "Using config file: $CONFIG_FILE"

python ConvNeXt_finetuning_advanced.py --config $CONFIG_FILE

# Example qsub command for multi-GPU training:
# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=80G -m beas -M faridkar@bu.edu -N SWINB_MT train_advanced.sh
