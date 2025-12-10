# Utils Directory

This directory contains utility scripts for managing herbarium specimen images, including downloading, processing, organizing, and labeling datasets.

## Scripts Overview

### Image Download & Installation

#### `image_install_parallel.py`
**Purpose**: Primary script for downloading herbarium specimen images from GBIF (Global Biodiversity Information Facility) multimedia datasets.

**Key Features**:
- Parallel downloading with ThreadPoolExecutor (5 workers)
- Host-based rate limiting and circuit breaker pattern
- Duplicate detection across multiple GBIF datasets
- IIIF (International Image Interoperability Framework) manifest support
- Automatic image resizing to 1024px max dimension
- Checkpoint system for resumable downloads
- Failed download tracking
- Hierarchical directory organization (3-digit prefix structure)

**Usage**:
```bash
python image_install_parallel.py [-c COUNTRY_CODE]
```

**Configuration**:
- Input: `/projectnb/herbdl/data/GBIF-F25/multimedia.txt`
- Output: `/projectnb/herbdl/data/GBIF-F25h/`
- Logs: `/projectnb/herbdl/logs/image_install_*.log`
- Checkpoints: `processed_ids.txt`, `failed_ids.txt`

**Advanced Features**:
- Host cooldown on rate limiting (429 errors): 30 minutes default
- Host cooldown on timeouts: 60 minutes
- Circuit breaker: Permanently blocks hosts after 50+ errors
- Multiple URL fallback per GBIF ID
- Retry strategy with backoff for 500-level errors

#### `image_install.sh`
**Purpose**: SCC job submission wrapper for `image_install_parallel.py`.

**Usage**:
```bash
qsub -N image_install -l h_rt=48:00:00 -pe omp 16 -P herbdl -m beas -M your_email@bu.edu image_install.sh
```

### Image Processing

#### `image_utils.py`
**Purpose**: Core image processing utilities used by other scripts.

**Functions**:
- `get_file_size_in_mb(file_path)`: Returns file size in megabytes
- `resize_with_aspect_ratio(image_path, output_path, max_size=1600, format="JPEG", quality=85)`:
  - Downscales images preserving aspect ratio
  - Handles alpha channels (RGBA/LA/P with transparency)
  - Converts to RGB JPEG format
  - Returns (changed: bool, final_size: tuple)

**Key Features**:
- Safe alpha channel removal with white background
- Progressive JPEG encoding
- Optimized output
- LANCZOS resampling for high-quality downscaling

#### `resize_images.py`
**Purpose**: Batch resize images in a directory using parallel processing.

**Configuration**:
- Input directory: `/projectnb/herbdl/data/harvard-herbaria/images`
- Target size: Images > 2MB are resized
- Workers: 10 parallel threads
- Log: `image_resize.log`

**Usage**:
```bash
python resize_images.py
```

#### `image_resize.sh`
**Purpose**: SCC job submission wrapper for `resize_images.py`.

**Usage**:
```bash
qsub -l h_rt=24:00:00 -pe omp 10 -P herbdl -m beas -M your_email@bu.edu image_resize.sh
```

#### `compression.sh`
**Purpose**: Compress images to 2MB target size and measure quality degradation using PSNR (Peak Signal-to-Noise Ratio).

**How it works**:
1. Reads images from `./images/` directory
2. Compresses each to 2MB using ImageMagick `convert`
3. Saves compressed versions to `./compressed/` directory
4. Calculates PSNR using FFmpeg and logs to `./logs/`

**Dependencies**: ImageMagick, FFmpeg

**Usage**:
```bash
./compression.sh
```

### Image Organization

#### `reorganize_images.py`
**Purpose**: Reorganize images from flat directory structure into hierarchical structure based on GBIF IDs.

**How it works**:
- Reads images from source directory
- Uses GBIF ID from filename (must be numeric)
- Creates hierarchical structure: `prefix1/prefix2/filename.jpg`
  - prefix1: First 3 digits of GBIF ID
  - prefix2: Digits 4-6 of GBIF ID
- Skips non-numeric filenames

**Configuration**:
- Source: `/projectnb/herbdl/data/GBIF-F25/images`
- Destination: `/projectnb/herbdl/data/GBIF-F25h`
- Supported formats: jpg, jpeg, png, tif, tiff (case-insensitive)

**Example**:
```
Image: 1234567.jpg
→ Moved to: 123/456/1234567.jpg
```

**Usage**:
```bash
python reorganize_images.py
```

### Dataset Labeling

#### `labeling.ipynb`
**Purpose**: Process Kaggle Herbarium 2021 and 2022 metadata to create labeled training/validation datasets.

**What it does**:
1. Loads metadata JSON files from Kaggle Herbarium competitions
2. Extracts taxonomic information (family, genus, species)
3. Generates natural language captions for each specimen
4. Encodes scientific names as numeric labels
5. Creates 80/20 train/validation splits
6. Exports to CSV and JSON formats

**Output Files**:
- `train_2022.csv`, `val_2022.csv` (Herbarium 2022)
- `train_2021.csv`, `val_2021.csv` (Herbarium 2021)
- JSON versions for direct use with HuggingFace datasets

**Columns**:
- `image_id`: Unique identifier
- `filename`: Relative path to image
- `caption`: Natural language description
- `scientificName`: Family + Genus + Species
- `family`, `genus`, `species`: Taxonomic labels
- `scientificNameEncoded`: Numeric label for classification

**Caption Format**:
```
"This is an image of species {species}, in the genus {genus} of family {family}. It is part of the collection of institution {institution}."
```

### Validation & Monitoring

#### `link_check.py`
**Purpose**: Validate image URLs from GBIF multimedia.txt files to identify broken links.

**How it works**:
- Reads multimedia.txt with GBIF IDs and image URLs
- Makes HEAD/GET requests to verify accessibility
- Checks Content-Type headers for valid image types
- Logs invalid links and their GBIF IDs
- Uses parallel processing (10 workers)

**Configuration**:
- Input: `/projectnb/herbdl/data/harvard-herbaria/gbif/multimedia.txt`
- Log: `link_check.log`
- Retry strategy: Up to 5 retries with backoff

**Usage**:
```bash
python link_check.py
```

#### `notifications.py`
**Purpose**: Send push notifications via Pushover API for long-running job monitoring.

**Setup**:
1. Create a `.env` file with:
```
PUSHOVER_API_TOKEN=your_token_here
PUSHOVER_USER_KEY=your_user_key_here
```

**Function**:
```python
send_notification(title, message)
```

**Usage Example**:
```python
from notifications import send_notification
send_notification("Image Installation", "Downloaded 50,000 images")
```

**Integration**: Used by `image_install_parallel.py` to send progress updates every 50,000 images.

## Common Workflows

### 1. Download GBIF Images
```bash
# Submit parallel download job
qsub -N image_install -l h_rt=48:00:00 -pe omp 16 -P herbdl image_install.sh

# Monitor progress in logs
tail -f /projectnb/herbdl/logs/image_install_*.log
```

### 2. Organize Downloaded Images
```bash
# Reorganize flat structure to hierarchical
python reorganize_images.py
```

### 3. Batch Resize Images
```bash
# Submit resize job
qsub -l h_rt=24:00:00 -pe omp 10 -P herbdl image_resize.sh
```

### 4. Create Training Datasets
```bash
# Run labeling notebook
jupyter notebook labeling.ipynb
```

### 5. Validate Image Links
```bash
# Check for broken URLs
python link_check.py
```

## Directory Structures

### Hierarchical Image Storage
Images are organized by GBIF ID prefix for efficient filesystem access:
```
/projectnb/herbdl/data/GBIF-F25h/
├── 000/
│   ├── 000/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   ├── 001/
│   │   ├── 000001000.jpg
├── 001/
│   ├── 000/
│   ├── 001/
```

This structure prevents issues with directories containing millions of files.

## Dependencies

**Python Libraries**:
- pandas
- PIL (Pillow)
- requests
- scikit-learn (for labeling)
- python-dotenv (for notifications)

**System Tools**:
- ImageMagick (for compression.sh)
- FFmpeg (for compression.sh)

## Notes

- All scripts are designed for use on Boston University's Shared Computing Cluster (SCC)
- Many scripts use parallel processing for performance
- Checkpoint files enable resumable operations after interruptions
- Always verify paths before running scripts to avoid data loss
