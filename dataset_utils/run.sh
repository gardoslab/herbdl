#!/bin/bash -l
#$ -N analyze_datasets

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python merge_datasets.py

# qsub -l h_rt=5:00:00 -pe omp 32 -P herbdl -l gpus=1 -l gpu_c=8.0 -m beas -M faridkar@bu.edu run.sh