#!/usr/bin/env python
# coding: utf-8

"""
This script runs the basic check of cleanvision to find duplicates, similar images, odd sizes and more. 
Output is printed. 
"""

# In[2]:


import os
from constants import KAGGLE_HERBARIUM_21_TRAIN_CSV, KAGGLE_HERBARIUM_21_VAL_CSV, KAGGLE_HERBARIUM_22_TRAIN_CSV, KAGGLE_HERBARIUM_22_VAL_CSV, KAGGLE_HERBARIUM_21_TRAIN, KAGGLE_HERBARIUM_22_TRAIN
from cleanvision import Imagelab
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# ## Kaggle images

kaggle22_train = pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV).drop(columns=['Unnamed: 0'])
print(f"Number of training samples in Kaggle Herbarium 22: {len(kaggle22_train)}")
kaggle22_val = pd.read_csv(KAGGLE_HERBARIUM_22_VAL_CSV).drop(columns=['Unnamed: 0'])
print(f"Number of validation samples in Kaggle Herbarium 22: {len(kaggle22_val)}")

kaggle22 = pd.concat([kaggle22_train, kaggle22_val]).reset_index(drop=True)

kaggle22_file_paths = [os.path.join(KAGGLE_HERBARIUM_22_TRAIN, file) for file in kaggle22['filename'].tolist()]
print(f"Number of images in Kaggle Herbarium 22: {len(kaggle22_file_paths)}")

kaggle21_train = pd.read_csv(KAGGLE_HERBARIUM_21_TRAIN_CSV).drop(columns=['Unnamed: 0'])
print(f"Number of training samples in Kaggle Herbarium 21: {len(kaggle21_train)}")
kaggle21_val = pd.read_csv(KAGGLE_HERBARIUM_21_VAL_CSV).drop(columns=['Unnamed: 0'])
print(f"Number of validation samples in Kaggle Herbarium 21: {len(kaggle21_val)}")

kaggle21 = pd.concat([kaggle21_train, kaggle21_val]).reset_index(drop=True)

kaggle21_file_paths = [os.path.join(KAGGLE_HERBARIUM_21_TRAIN, file) for file in kaggle21['filename'].tolist()]
print(f"Number of images in Kaggle Herbarium 21 Train: {len(kaggle21_file_paths)}")


KAGGLE_FILEPATHS = kaggle22_file_paths + kaggle21_file_paths
print(f"Total number of images from Kaggle: {len(KAGGLE_FILEPATHS)}")

# find the overlap between the two datasets
kaggle21_train['scientificName'] = kaggle21_train['scientificName'].str.lower()
kaggle22_train['scientificName'] = kaggle22_train['scientificName'].str.lower()

# find the overlap in the scientific names
overlap = kaggle21_train['scientificName'].isin(kaggle22_train['scientificName'])
print(f"Number of overlapping scientific names: {overlap.sum()}")

# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(filepaths=KAGGLE_FILEPATHS)

# Automatically check for duplicates
issue_types = {"exact_duplicates": {}, "near_duplicates": {}}
imagelab.find_issues(issue_types)

# Produce a neat report of the issues found in your dataset
imagelab.report()

save_path = "./results"
imagelab.save(save_path, force=True)

