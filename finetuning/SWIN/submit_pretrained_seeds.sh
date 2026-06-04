#!/bin/bash
# Submit 5 independent seed runs of the pretrained-384 full pipeline.
# Each job runs on 1 GPU (A100 80G) for up to 48h.
#
# Usage:
#   bash submit_pretrained_seeds.sh           # submit all 5 seeds
#   SEEDS="0 2 4" bash submit_pretrained_seeds.sh  # submit a subset

SEEDS=${SEEDS:-"0 1 2 3 4"}

QSUB_ARGS="-l h_rt=48:00:00 -pe omp 8 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=80G -m beas -M faridkar@bu.edu"

for seed in $SEEDS; do
    CONFIG="configs_advanced/swin_pretrained_384_seed${seed}.yml"
    JOB_NAME="PRETRAINED_384_S${seed}"

    JOB=$(qsub $QSUB_ARGS \
        -N "$JOB_NAME" \
        -v CONFIG_FILE="$CONFIG" \
        train_advanced.sh | grep -oP '(?<=job )\d+')

    echo "Submitted seed ${seed}: job ${JOB}  (${CONFIG})"
done

echo ""
echo "Monitor with: qstat -u faridkar"
