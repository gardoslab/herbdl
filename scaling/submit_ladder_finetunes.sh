#!/bin/bash
# Submit one SWIN fine-tuning job per ladder rung.
#
# Each job resumes from the SAME checkpoint the 15k arm used
# (SWIN_BASE_BASELINE) and differs only in the size of its label space, so any
# difference between them is attributable to label cardinality.
#
# Must be run from finetuning/SWIN/ on SCC, because train_advanced.sh expects
# SWIN_finetuning_advanced.py beside it.
#
# Usage:  bash submit_ladder_finetunes.sh <config_dir> [h_rt] [ngpus]

set -uo pipefail
CFG_DIR="${1:?usage: submit_ladder_finetunes.sh <config_dir> [h_rt] [ngpus]}"
H_RT="${2:-24:00:00}"
NGPUS="${3:-1}"
EMAIL="faridkar@bu.edu"

if [ ! -d "$CFG_DIR" ]; then echo "ERROR: no such config dir: $CFG_DIR" >&2; exit 1; fi
if [ ! -f train_advanced.sh ]; then
    echo "ERROR: run this from finetuning/SWIN/ (train_advanced.sh not found here)" >&2
    exit 1
fi

shopt -s nullglob
CONFIGS=("$CFG_DIR"/ladder_n*.yml)
if [ ${#CONFIGS[@]} -eq 0 ]; then echo "ERROR: no ladder_n*.yml in $CFG_DIR" >&2; exit 1; fi

echo "Submitting ${#CONFIGS[@]} job(s), h_rt=$H_RT gpus=$NGPUS"
for CFG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CFG" .yml | tr '[:lower:]' '[:upper:]')
    JOB=$(qsub -terse \
        -l h_rt="$H_RT" \
        -pe omp 16 \
        -P herbdl \
        -l gpus="$NGPUS" \
        -l gpu_c=8.0 \
        -m beas -M "$EMAIL" \
        -N "$NAME" \
        -cwd \
        -v CONFIG_FILE="$CFG" \
        train_advanced.sh)
    echo "  $NAME  job=$JOB  config=$CFG"
done
echo "Done. Track with: qstat -u $USER"
