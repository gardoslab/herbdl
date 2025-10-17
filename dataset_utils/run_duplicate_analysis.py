#!/usr/bin/env python
# coding: utf-8

"""
This script runs CleanVision to find duplicates, similar images, odd sizes and more
on any two datasets with user-specified configurations.
"""

import os
import argparse
from cleanvision import Imagelab
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


def load_and_process_dataset(csv, filename_col, label_col=None, drop_cols=None):
    """
    Load and combine train/val CSVs for a dataset.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        image_dir: Base directory containing images
        filename_col: Column name containing image filenames
        label_col: Column name containing labels (optional, for overlap analysis)
        drop_cols: List of columns to drop (e.g., ['Unnamed: 0'])
    
    Returns:
        combined_df: Combined dataframe
        file_paths: List of full file paths
    """
    # Load datasets
    df = pd.read_csv(csv)
    
    # Drop unwanted columns if specified
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    print(f"Number of samples: {len(df)}")
   
    # Create full file paths
    file_paths = [file for file in df[filename_col].tolist()]
    print(f"Total number of images: {len(file_paths)}")
    
    return df, file_paths


def analyze_label_overlap(df1, df2, label_col):
    """
    Analyze overlap between labels in two datasets.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        label_col: Column name containing labels
    """
    if label_col not in df1.columns or label_col not in df2.columns:
        print(f"Warning: Label column '{label_col}' not found in one or both datasets. Skipping overlap analysis.")
        return
    
    # Normalize labels (lowercase)
    labels1 = df1[label_col].str.lower()
    labels2 = df2[label_col].str.lower()
    
    # Find overlap
    overlap = labels1.isin(labels2)
    unique_labels1 = labels1.nunique()
    unique_labels2 = labels2.nunique()
    overlapping_labels = labels1[overlap].nunique()
    
    print(f"\n=== Label Overlap Analysis ===")
    print(f"Unique labels in Dataset 1: {unique_labels1}")
    print(f"Unique labels in Dataset 2: {unique_labels2}")
    print(f"Number of overlapping labels: {overlapping_labels}")
    print(f"Percentage overlap: {overlapping_labels / unique_labels1 * 100:.2f}%")


def run_cleanvision(file_paths, issue_types=None, save_path="./results"):
    """
    Run CleanVision on the provided file paths.
    
    Args:
        file_paths: List of image file paths
        issue_types: Dict of issue types to check (default: duplicates only)
        save_path: Path to save results
    """
    if issue_types is None:
        issue_types = {"exact_duplicates": {}, "near_duplicates": {}}
    
    print(f"\n=== Running CleanVision ===")
    print(f"Total images to analyze: {len(file_paths)}")
    
    # Initialize Imagelab
    imagelab = Imagelab(filepaths=file_paths)
    
    # Find issues
    imagelab.find_issues(issue_types)
    
    # Print report
    print("\n=== CleanVision Report ===")
    imagelab.report()
    
    # Save results
    imagelab.save(save_path, force=True)
    print(f"\nResults saved to: {save_path}")
    
    return imagelab


def main():
    parser = argparse.ArgumentParser(description='Run CleanVision on two datasets')
    
    # Dataset 1 arguments
    parser.add_argument('--dataset1_train_csv', required=True, help='Path to dataset 1 training CSV')
    parser.add_argument('--dataset1_filename_col', required=True, help='Column name for image filenames in dataset 1')
    parser.add_argument('--dataset1_label_col', default=None, help='Column name for labels in dataset 1 (optional)')
    
    # Dataset 2 arguments
    parser.add_argument('--dataset2_train_csv', required=True, help='Path to dataset 2 training CSV')
    parser.add_argument('--dataset2_filename_col', required=True, help='Column name for image filenames in dataset 2')
    parser.add_argument('--dataset2_label_col', default=None, help='Column name for labels in dataset 2 (optional)')
    
    # Optional arguments
    parser.add_argument('--drop_cols', nargs='+', default=['Unnamed: 0'], help='Columns to drop from CSVs')
    parser.add_argument('--save_path', default='./results', help='Path to save CleanVision results')
    parser.add_argument('--issue_types', nargs='+', default=['exact_duplicates'],
                        help='Issue types to check (exact_duplicates, near_duplicates, odd_aspect_ratio, etc.)')
    
    args = parser.parse_args()
    
    # Process Dataset 1
    print("=" * 60)
    print("DATASET 1")
    print("=" * 60)
    df1, paths1 = load_and_process_dataset(
        args.dataset1_train_csv,
        args.dataset1_filename_col,
        args.dataset1_label_col,
        args.drop_cols
    )
    
    # Process Dataset 2
    print("\n" + "=" * 60)
    print("DATASET 2")
    print("=" * 60)
    df2, paths2 = load_and_process_dataset(
        args.dataset2_train_csv,
        args.dataset2_filename_col,
        args.dataset2_label_col,
        args.drop_cols
    )
    
    # Combine all file paths
    all_paths = paths1 + paths2
    print(f"\n{'=' * 60}")
    print(f"COMBINED DATASETS")
    print(f"{'=' * 60}")
    print(f"Total images from both datasets: {len(all_paths)}")
    print(f"Number of overlapping filepaths: {len(set(paths1).intersection(set(paths2)))}")
    
    # Analyze label overlap if label columns provided
    if args.dataset1_label_col and args.dataset2_label_col:
        analyze_label_overlap(df1, df2, args.dataset1_label_col if args.dataset1_label_col == args.dataset2_label_col 
                             else args.dataset1_label_col)
    
    # Prepare issue types
    issue_types_dict = {issue_type: {} for issue_type in args.issue_types}
    
    # Run CleanVision
    imagelab = run_cleanvision(all_paths, issue_types_dict, args.save_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()