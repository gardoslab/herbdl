#!/bin/bash

# Script to launch multiple training jobs from a list of config files
# Usage: ./launch_multiple_jobs.sh <config_list.txt>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_list_file>"
    echo "Example: $0 configs_to_run.txt"
    exit 1
fi

CONFIG_LIST_FILE="$1"

if [ ! -f "$CONFIG_LIST_FILE" ]; then
    echo "Error: Config list file '$CONFIG_LIST_FILE' not found!"
    exit 1
fi

# Read each config file from the list and submit a job
while IFS= read -r CONFIG_FILE || [ -n "$CONFIG_FILE" ]; do
    # Skip empty lines and comments
    [[ -z "$CONFIG_FILE" ]] && continue
    [[ "$CONFIG_FILE" =~ ^#.* ]] && continue

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Warning: Config file '$CONFIG_FILE' not found, skipping..."
        continue
    fi

    # Extract a job name from the config file path
    # e.g., configs/swinv2_base_unfrozen_15k.yml -> SWINV2_BASE_UNFROZEN_15K
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yml)
    JOB_NAME=$(echo "$CONFIG_BASENAME" | tr '[:lower:]' '[:upper:]' | tr '-' '_')

    echo "Submitting job for config: $CONFIG_FILE"
    echo "  Job name: $JOB_NAME"

    # Submit the job with qsub
    # Using the same parameters as in train_advanced.sh example
    qsub -l h_rt=48:00:00 \
         -pe omp 16 \
         -P herbdl \
         -l gpus=2 \
         -l gpu_c=8.0 \
         -m beas \
         -M faridkar@bu.edu \
         -N "$JOB_NAME" \
         -v CONFIG_FILE="$CONFIG_FILE" \
         train_advanced.sh

    echo "  Submitted!"
    echo ""

done < "$CONFIG_LIST_FILE"

echo "All jobs submitted successfully!"
