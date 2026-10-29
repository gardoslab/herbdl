#!/bin/bash -l
# Ladder data characterization (CPU only). Reads the n15k/n32k/n64k manifests built
# by build_ladder.py and prints species-by-training-image distributions.
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/ladder_distribution.py
