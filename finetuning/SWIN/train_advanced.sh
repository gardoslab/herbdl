#!/bin/bash -l

module load miniconda
module load academic-ml/fall-2025

conda activate herb_env

# Path to config file - can be set via environment variable or use default
# Options:
#   - configs_advanced/swin_base_224_enhanced.yml
#   - configs_advanced/swin_base_384_enhanced.yml
#   - configs_advanced/swinv2_base_192_enhanced.yml
# If CONFIG_FILE is not set (e.g., via qsub -v), use default
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/swin_large_unfrozen_15k.yml"
fi

echo "Using config file: $CONFIG_FILE"

python SWIN_finetuning_advanced.py --config $CONFIG_FILE

# Example qsub command for multi-GPU training:
# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=80G -m beas -M faridkar@bu.edu -N SWIN_Large_50K train_advanced.sh
