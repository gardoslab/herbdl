#!/bin/bash -l
# Phase 3 per-species analysis (CPU only). Env setup inside the script, same as the
# other loss_ablation wrappers -- see run_hace_tree_smoke.sh for why.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/phase3_per_species.py
