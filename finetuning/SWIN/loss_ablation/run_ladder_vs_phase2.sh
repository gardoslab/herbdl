#!/bin/bash -l
# CPU only. Env setup inside the script, same as the other loss_ablation wrappers.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/ladder_vs_phase2.py
