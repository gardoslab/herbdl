#!/bin/bash

conda activate herb_env

python run_duplicate_analysis.py \
    --dataset1_train_csv "/projectnb/herbdl/data/kaggle-herbaria/index.csv" \
    --dataset1_filename_col "filepath" \
    --dataset1_label_col "scientificName" \
    --dataset2_train_csv "/projectnb/herbdl/data/harvard-herbaria/index.csv" \
    --dataset2_filename_col "filepath" \
    --dataset2_label_col "scientificName" \
    --save_path ./kaggle_harvard_results \
    --issue_types "exact_duplicates"