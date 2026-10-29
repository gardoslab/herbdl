#!/bin/bash -l
# Same-specimen copy check on the full val_2022 set + re-scoring of every saved prediction set (1 GPU).
module load miniconda
module load academic-ml/fall-2026
conda activate fall-2026-pyt
python loss_ablation/part4_leakage.py
