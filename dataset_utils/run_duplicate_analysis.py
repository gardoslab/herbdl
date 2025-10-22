#!/usr/bin/env python
# coding: utf-8

"""
This script runs CleanVision to find duplicates, similar images, odd sizes and more
on any two datasets with user-specified configurations.
It can also load existing results and remove duplicate images from dataframes.
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
    file_paths = df[filename_col].tolist()
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


def load_cleanvision_results(results_path):
    """
    Load existing CleanVision results.

    Args:
        results_path: Path to the saved CleanVision results directory

    Returns:
        imagelab: Loaded Imagelab object
    """
    print(f"\n=== Loading CleanVision Results ===")
    print(f"Loading from: {results_path}")

    imagelab = Imagelab.load(results_path)

    return imagelab


def get_duplicate_sets(imagelab):
    """
    Extract duplicate sets from imagelab.

    Args:
        imagelab: Imagelab object with duplicate analysis results

    Returns:
        duplicate_sets: List of lists, where each inner list contains paths of duplicate images
    """
    duplicate_sets = imagelab.info['exact_duplicates']['sets']

    print(f"\n=== Duplicate Sets Summary ===")
    print(f"Number of duplicate sets: {len(duplicate_sets)}")

    if duplicate_sets:
        # Calculate total duplicates
        total_duplicates = sum(len(s) - 1 for s in duplicate_sets)  # Keep one from each set
        print(f"Total duplicate images to remove: {total_duplicates}")

        # Show some statistics
        set_sizes = [len(s) for s in duplicate_sets]
        print(f"Smallest duplicate set size: {min(set_sizes)}")
        print(f"Largest duplicate set size: {max(set_sizes)}")
        print(f"Average duplicate set size: {sum(set_sizes)/len(set_sizes):.2f}")

    return duplicate_sets


def select_images_to_remove(duplicate_sets, keep_strategy='first'):
    """
    From each duplicate set, select which images to remove.

    Args:
        duplicate_sets: List of lists containing duplicate image paths
        keep_strategy: Strategy for selecting which image to keep
                      'first': Keep the first image in each set (default)
                      'last': Keep the last image in each set
                      'shortest_path': Keep image with shortest path

    Returns:
        images_to_remove: Set of image paths to remove
        images_to_keep: Set of image paths to keep (one per duplicate set)
    """
    images_to_remove = set()
    images_to_keep = set()

    for dup_set in duplicate_sets:
        if keep_strategy == 'first':
            keep_image = dup_set[0]
            remove_images = dup_set[1:]
        elif keep_strategy == 'last':
            keep_image = dup_set[-1]
            remove_images = dup_set[:-1]
        elif keep_strategy == 'shortest_path':
            keep_image = min(dup_set, key=len)
            remove_images = [img for img in dup_set if img != keep_image]
        else:
            raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

        images_to_keep.add(keep_image)
        images_to_remove.update(remove_images)

    print(f"\n=== Image Selection Summary ===")
    print(f"Strategy: {keep_strategy}")
    print(f"Images to keep (one per duplicate set): {len(images_to_keep)}")
    print(f"Images to remove: {len(images_to_remove)}")

    return images_to_remove, images_to_keep


def remove_duplicates_from_dataframe(df, filename_col, images_to_remove):
    """
    Remove rows from dataframe where the filename is in images_to_remove.

    Args:
        df: DataFrame to clean
        filename_col: Column name containing image filenames
        images_to_remove: Set of image paths to remove

    Returns:
        cleaned_df: DataFrame with duplicate images removed
        removed_count: Number of rows removed
    """
    original_size = len(df)

    # Filter out rows where filename is in images_to_remove
    cleaned_df = df[~df[filename_col].isin(images_to_remove)].copy()

    removed_count = original_size - len(cleaned_df)

    print(f"\nRemoved {removed_count} rows from dataframe")
    print(f"Original size: {original_size}, New size: {len(cleaned_df)}")

    return cleaned_df, removed_count


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
    parser = argparse.ArgumentParser(description='Run CleanVision on two datasets and optionally remove duplicates')

    # Mode selection
    parser.add_argument('--load_results', type=str, default=None,
                        help='Path to existing CleanVision results directory (skips analysis)')
    parser.add_argument('--remove_duplicates', action='store_true',
                        help='Remove duplicate images from dataframes and save cleaned CSVs')
    parser.add_argument('--keep_strategy', type=str, default='first',
                        choices=['first', 'last', 'shortest_path'],
                        help='Strategy for selecting which duplicate image to keep (default: first)')

    # Dataset 1 arguments
    parser.add_argument('--dataset1_train_csv', required=True, help='Path to dataset 1 training CSV')
    parser.add_argument('--dataset1_filename_col', required=True, help='Column name for image filenames in dataset 1')
    parser.add_argument('--dataset1_label_col', default=None, help='Column name for labels in dataset 1 (optional)')
    parser.add_argument('--dataset1_output_csv', default=None, help='Output path for cleaned dataset 1 CSV')

    # Dataset 2 arguments
    parser.add_argument('--dataset2_train_csv', required=True, help='Path to dataset 2 training CSV')
    parser.add_argument('--dataset2_filename_col', required=True, help='Column name for image filenames in dataset 2')
    parser.add_argument('--dataset2_label_col', default=None, help='Column name for labels in dataset 2 (optional)')
    parser.add_argument('--dataset2_output_csv', default=None, help='Output path for cleaned dataset 2 CSV')

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

    # Run or load CleanVision analysis
    if args.load_results:
        # Load existing results
        imagelab = load_cleanvision_results(args.load_results)
    else:
        # Prepare issue types
        issue_types_dict = {issue_type: {} for issue_type in args.issue_types}

        # Run CleanVision
        imagelab = run_cleanvision(all_paths, issue_types_dict, args.save_path)

    # Remove duplicates if requested
    if args.remove_duplicates:
        print("\n" + "=" * 60)
        print("REMOVING DUPLICATES")
        print("=" * 60)

        # Get duplicate sets
        duplicate_sets = get_duplicate_sets(imagelab)

        if not duplicate_sets:
            print("\nNo duplicates found. Nothing to remove.")
        else:
            # Select images to remove
            images_to_remove, images_to_keep = select_images_to_remove(duplicate_sets, args.keep_strategy)

            # Remove duplicates from dataframes
            print("\n--- Dataset 1 ---")
            df1_cleaned, removed1 = remove_duplicates_from_dataframe(
                df1, args.dataset1_filename_col, images_to_remove
            )

            print("\n--- Dataset 2 ---")
            df2_cleaned, removed2 = remove_duplicates_from_dataframe(
                df2, args.dataset2_filename_col, images_to_remove
            )

            # Save cleaned dataframes
            if args.dataset1_output_csv:
                df1_cleaned.to_csv(args.dataset1_output_csv, index=False)
                print(f"\nSaved cleaned Dataset 1 to: {args.dataset1_output_csv}")
            else:
                # Default: save with _cleaned suffix
                base_name = args.dataset1_train_csv.rsplit('.', 1)[0]
                output_path = f"{base_name}_cleaned.csv"
                df1_cleaned.to_csv(output_path, index=False)
                print(f"\nSaved cleaned Dataset 1 to: {output_path}")

            if args.dataset2_output_csv:
                df2_cleaned.to_csv(args.dataset2_output_csv, index=False)
                print(f"Saved cleaned Dataset 2 to: {args.dataset2_output_csv}")
            else:
                # Default: save with _cleaned suffix
                base_name = args.dataset2_train_csv.rsplit('.', 1)[0]
                output_path = f"{base_name}_cleaned.csv"
                df2_cleaned.to_csv(output_path, index=False)
                print(f"Saved cleaned Dataset 2 to: {output_path}")

            print(f"\n{'=' * 60}")
            print(f"DUPLICATE REMOVAL COMPLETE")
            print(f"{'=' * 60}")
            print(f"Total images removed from Dataset 1: {removed1}")
            print(f"Total images removed from Dataset 2: {removed2}")
            print(f"Total images removed: {removed1 + removed2}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()