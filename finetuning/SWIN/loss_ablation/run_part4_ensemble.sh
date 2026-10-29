#!/bin/bash -l
# Part 4 ensemble analysis (CPU only): reads the saved predictions under PHASE3_EVAL / PHASE4_EVAL.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/part4_ensemble.py
