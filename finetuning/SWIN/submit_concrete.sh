#!/bin/bash
# Submit the "concrete next run" (SWIN-L 384) as a seed ensemble, or a single seed.
# Each job: 1 A100-80G GPU, up to 48h, herbdl project. Uses train_advanced.sh
# (module load miniconda + academic-ml/fall-2025, conda activate herb_env).
#
# Usage:
#   bash submit_concrete.sh                  # submit seeds 0 1 2
#   SEEDS="0"           bash submit_concrete.sh   # single run, seed 0
#   SEEDS="0 1 2 3 4"   bash submit_concrete.sh   # full 5-seed ensemble
#
# Optional — warm-start from an existing SWIN-L 384 checkpoint (Tier 2.5, recommended
# once you have one; keep it on the 384 arch). Overrides model.model_name_or_path:
#   CKPT=/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/SWIN_L_384_... \
#       bash submit_concrete.sh
#
# Nothing is auto-submitted by Claude — run this yourself when ready.

SEEDS=${SEEDS:-"0 1 2"}
CONFIG="configs_advanced/swin_large_384_concrete.yml"
OUT_BASE="/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN"
QSUB_ARGS="-l h_rt=48:00:00 -pe omp 8 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=80G -m beas -M tgardos@bu.edu"

for seed in $SEEDS; do
    OUT="${OUT_BASE}/SWIN_L_384_CONCRETE_SEED${seed}"
    SET_ARGS="--set training.seed=${seed} --set training.output_dir=${OUT} --set training.logging_dir=${OUT} --set custom.run_id=swin_l_384_concrete_seed${seed} --set custom.run_name=SWIN_L_384_Concrete_Seed${seed}"
    if [ -n "$CKPT" ]; then
        SET_ARGS="${SET_ARGS} --set model.model_name_or_path=${CKPT}"
    fi

    JOB=$(qsub $QSUB_ARGS \
        -N "SWINL384_S${seed}" \
        -v CONFIG_FILE="${CONFIG}",SET_ARGS="${SET_ARGS}" \
        train_advanced.sh | grep -oP '(?<=job )\d+')
    echo "Submitted seed ${seed}: job ${JOB}  ->  ${OUT}"
done

echo
echo "Monitor with: qstat -u \$USER"
