#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

python resize_images.py

### The command below is used to submit the job to the cluster
### qsub -l h_rt=24:00:00 -pe omp 10 -P herbdl -m beas -M faridkar@bu.edu image_resize.sh
