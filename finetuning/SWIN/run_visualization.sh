#!/bin/bash
# Launch the interactive plot visualization dashboard

# Activate the herb_env conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate herb_env

# Run the streamlit app
streamlit run visualize_plots.py --server.port 8501 --server.address localhost
