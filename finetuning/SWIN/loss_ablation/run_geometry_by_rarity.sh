#!/bin/bash -l
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/geometry_by_rarity.py
