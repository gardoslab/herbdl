#!/bin/bash -l
#$ -P herbdl
#$ -l h_rt=01:00:00
#$ -pe omp 4
#$ -N species_sphere
#$ -j y
#$ -cwd
# Species-sphere figure (species_sphere.py). CPU only: reads the cached PHASE3_GEOM
# features, so no GPU is needed. Submit from finetuning/SWIN; extra arguments are passed
# to species_sphere.py:
#   qsub loss_ablation/run_species_sphere.sh
#   qsub loss_ablation/run_species_sphere.sh --models arcface dbm kappaface --min-val 8

module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/species_sphere.py "$@"
