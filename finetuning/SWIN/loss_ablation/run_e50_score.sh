#!/bin/bash -l
# Score the epoch-matched (50-epoch) 32k NormFace and ArcFace models on the 32k rung's full validation set (1 GPU).
for m in cosine_m0 arcface; do
  echo "__E50__ score n32k_${m}_e50 $(date +%H:%M)"
  HERBDL_NO_RELOCATE=1 CONFIG_FILE="configs_advanced/ladder_eval/n32k_${m}_e50.yml" bash train_advanced.sh || echo "__E50__ FAILED $m"
done
