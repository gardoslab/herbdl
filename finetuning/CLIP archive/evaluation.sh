#!/bin/bash

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

python evaluation.py

### The command below is used to submit the job to the cluster
### qsub -l h_rt=24:00:00 -pe omp 32 -P herbdl -l gpus=3 -l gpu_c=8.0 -m beas -M faridkar@bu.edu evaluation.sh