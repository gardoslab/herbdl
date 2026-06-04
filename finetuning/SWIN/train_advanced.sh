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
    CONFIG_FILE="hyperparameter_configs/swin_base_cosine_lr1e4_warmup.yml"
fi

echo "Using config file: $CONFIG_FILE"
[ -n "$SET_ARGS" ] && echo "Overrides: $SET_ARGS"

python SWIN_finetuning_advanced.py --config $CONFIG_FILE ${SET_ARGS}

# Example qsub command for multi-GPU training:
# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=4 -l gpu_c=8.0 -m beas -M faridkar@bu.edu -N SWIN_BASELINE train_advanced.sh
