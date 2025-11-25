#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python analyze_image_progress.py

### The command below is used to submit the job to the cluster
### qsub -N analyze_progress -l h_rt=1:00:00 -pe omp 1 -P herbdl -m beas -M tsehou26@bu.edu analyze_progress.sh
