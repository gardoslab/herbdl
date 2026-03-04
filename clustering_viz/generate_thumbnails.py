#!/usr/bin/env python3
"""
Generate thumbnails for herbarium specimen images used in the clustering visualization.

This script reads the Plotly JSON file, extracts image paths, and generates
thumbnails to improve hover preview performance in the web interface.

Usage:
    python generate_thumbnails.py <json_file> [--thumbnail-dir DIR] [--max-size SIZE]
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def extract_image_paths(json_file):
    """Extract unique image paths from Plotly JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_paths = set()

    # Extract customdata from all traces
    for trace in data.get('data', []):
        customdata = trace.get('customdata', [])
        for item in customdata:
            if item and len(item) > 0:
                # customdata contains relative paths like '/00012/1234567.jpg'
                image_paths.add(item[0])

    return sorted(image_paths)


def create_thumbnail(source_path, thumbnail_path, max_size=200):
    """Create a thumbnail from source image."""
    try:
        # Open and create thumbnail
        with Image.open(source_path) as img:
            # Calculate new size maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Create directory if it doesn't exist
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

            # Save thumbnail with optimization
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)

        return True, source_path
    except Exception as e:
        return False, f"Error processing {source_path}: {e}"


def generate_thumbnails(json_file, base_image_dir, thumbnail_dir, max_size=200, max_workers=8):
    """
    Generate thumbnails for all images referenced in the JSON file.

    Args:
        json_file: Path to Plotly JSON file
        base_image_dir: Base directory containing full resolution images
        thumbnail_dir: Directory to save thumbnails
        max_size: Maximum dimension (width or height) for thumbnails
        max_workers: Number of parallel workers
    """
    print(f"Reading image paths from {json_file}...")
    image_paths = extract_image_paths(json_file)
    print(f"Found {len(image_paths)} unique images")

    # Create thumbnail directory
    thumbnail_dir = Path(thumbnail_dir)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    base_image_dir = Path(base_image_dir)

    # Prepare tasks
    tasks = []
    for rel_path in image_paths:
        # Remove leading slash if present
        rel_path_clean = rel_path.lstrip('/')
        source_path = base_image_dir / rel_path_clean
        thumbnail_path = thumbnail_dir / rel_path_clean

        # Skip if thumbnail already exists
        if thumbnail_path.exists():
            continue

        tasks.append((source_path, thumbnail_path, max_size))

    if not tasks:
        print("All thumbnails already exist!")
        return

    print(f"Generating {len(tasks)} thumbnails with max size {max_size}px...")

    # Process thumbnails in parallel
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(create_thumbnail, src, dst, size): (src, dst)
                   for src, dst, size in tasks}

        with tqdm(total=len(futures), desc="Generating thumbnails") as pbar:
            for future in as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\n{message}")
                pbar.update(1)

    print(f"\nCompleted: {success_count} successful, {error_count} errors")
    print(f"Thumbnails saved to: {thumbnail_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate thumbnails for clustering visualization'
    )
    parser.add_argument(
        'json_file',
        help='Path to Plotly JSON file (e.g., asteraceae_tsne_plot_*.json)'
    )
    parser.add_argument(
        '--base-image-dir',
        default='/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_images',
        help='Base directory containing full resolution images'
    )
    parser.add_argument(
        '--thumbnail-dir',
        default='thumbnails',
        help='Directory to save thumbnails (default: ./thumbnails)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=200,
        help='Maximum thumbnail dimension in pixels (default: 200)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)

    if not os.path.exists(args.base_image_dir):
        print(f"Error: Base image directory not found: {args.base_image_dir}")
        sys.exit(1)

    # Generate thumbnails
    generate_thumbnails(
        args.json_file,
        args.base_image_dir,
        args.thumbnail_dir,
        args.max_size,
        args.workers
    )


if __name__ == '__main__':
    main()
