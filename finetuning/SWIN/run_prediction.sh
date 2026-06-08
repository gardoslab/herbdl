#!/bin/bash

# This script submits multiple prediction jobs to qsub
# Each checkpoint gets its own job so they run in parallel

# Configuration
BATCH_SIZE=64
OUTPUT_DIR="prediction_results_training"

# Create output and log directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Define list of (config, checkpoint) pairs
# Format: "config_file|checkpoint_directory"
CHECKPOINT_CONFIGS=(
    "configs/swin_base_unfrozen_15k.yml|/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/SWIN_BASE_UNFROZEN_15K"
)

# Function to ensure label_mapping.json exists in checkpoint directory
ensure_label_mapping() {
    local checkpoint_dir=$1
    local label_mapping="${checkpoint_dir}/label_mapping.json"

    # If label_mapping.json doesn't exist in checkpoint dir, look in parent
    if [ ! -f "$label_mapping" ]; then
        local parent_dir=$(dirname "$checkpoint_dir")
        local parent_mapping="${parent_dir}/label_mapping.json"

        if [ -f "$parent_mapping" ]; then
            echo "Copying label_mapping.json from $parent_dir to $checkpoint_dir"
            cp "$parent_mapping" "$label_mapping"
        else
            echo "Warning: label_mapping.json not found in checkpoint or parent directory"
            echo "Prediction script will use fallback mapping from eval dataset"
        fi
    fi
}

# Submit a job for each checkpoint
for ENTRY in "${CHECKPOINT_CONFIGS[@]}"; do
    # Split the entry into config and checkpoint
    IFS='|' read -r CONFIG_FILE CHECKPOINT_DIR <<< "$ENTRY"

    # Ensure label mapping exists
    ensure_label_mapping "$CHECKPOINT_DIR"

    # Generate output filename and job name based on model and checkpoint name
    PARENT_DIR=$(basename "$(dirname "$CHECKPOINT_DIR")")
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")

    # If checkpoint dir is same as parent (final checkpoint), just use parent name
    if [ "$PARENT_DIR" == "SWIN" ]; then
        # This is a full path to output dir, use the last dir name
        MODEL_NAME=$(basename "$CHECKPOINT_DIR")
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}.json"
        JOB_NAME="PRED_${MODEL_NAME}"
    else
        # This is a specific checkpoint subdirectory
        OUTPUT_FILE="${OUTPUT_DIR}/${PARENT_DIR}_${CHECKPOINT_NAME}.json"
        JOB_NAME="PRED_${PARENT_DIR}_${CHECKPOINT_NAME}"
    fi

    echo "Submitting job for: $CHECKPOINT_DIR"
    echo "  Config: $CONFIG_FILE"
    echo "  Output: $OUTPUT_FILE"
    echo "  Job name: $JOB_NAME"

    # Submit the job
    qsub -l h_rt=24:00:00 \
         -pe omp 16 \
         -P herbdl \
         -l gpus=1 \
         -l gpu_c=8.0 \
         -m beas \
         -M faridkar@bu.edu \
         -N "$JOB_NAME" \
         -o "logs/${JOB_NAME}.out" \
         -e "logs/${JOB_NAME}.err" \
         <<EOF
#!/bin/bash -l

# Load environment
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load miniconda
module load academic-ml/fall-2025
conda activate herb_env

# Run prediction
python prediction.py \
    --config $CONFIG_FILE \
    --checkpoint $CHECKPOINT_DIR \
    --output $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
    --use_training

echo "Predictions saved to: $OUTPUT_FILE"
EOF

    echo "  ✓ Job submitted"
    echo ""
done

echo "All jobs submitted! Check 'qstat' to monitor progress."
