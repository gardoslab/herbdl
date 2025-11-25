#!/bin/bash -l

module load miniconda
module load academic-ml/fall-2025

conda activate herb_env

# Path to config file - change this to use different configurations
CONFIG_FILE="configs/swin_base_unfrozen.yml"

python SWIN_finetuning.py --config $CONFIG_FILE

# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=48G -m beas -M faridkar@bu.edu train.sh
