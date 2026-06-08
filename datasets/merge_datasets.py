#!/usr/bin/env python
# coding: utf-8

"""
This script runs the basic check of cleanvision to find duplicates, similar images, odd sizes and more. 
Output is printed. 
"""

# In[2]:


import os
from constants import *
from cleanvision import Imagelab
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[3]:


KAGGLE_HERBARIUM_22_TRAIN


# ## Kaggle images

# In[4]:


kaggle22_train = pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV).drop(columns=['Unnamed: 0'])

kaggle22_file_paths = [os.path.join(KAGGLE_HERBARIUM_22_TRAIN, file) for file in kaggle22_train['filename']]
print(f"Number of images in Kaggle Herbarium 22 Train: {len(kaggle22_file_paths)}")
kaggle22_file_paths[:5]


# In[5]:


kaggle21_train = pd.read_csv(KAGGLE_HERBARIUM_21_METADATA_CSV).drop(columns=['Unnamed: 0'])

kaggle21_file_paths = [os.path.join(KAGGLE_HERBARIUM_21_TRAIN, file) for file in kaggle21_train['filename']]
print(f"Number of images in Kaggle Herbarium 21 Train: {len(kaggle21_file_paths)}")
kaggle21_file_paths[:5]


# In[6]:


KAGGLE_FILEPATHS = kaggle22_file_paths + kaggle21_file_paths
print(f"Total number of images from Kaggle: {len(KAGGLE_FILEPATHS)}")


# In[7]:


kaggle21_train['scientificName'] = kaggle21_train['family'] + ' ' + kaggle21_train['genus'] + ' ' + kaggle21_train['species']
kaggle22_train['scientificName'] = kaggle22_train['family'] + ' ' + kaggle22_train['genus'] + ' ' + kaggle22_train['species']


# In[8]:


# find the overlap between the two datasets
kaggle21_train['scientificName'] = kaggle21_train['scientificName'].str.lower()
kaggle22_train['scientificName'] = kaggle22_train['scientificName'].str.lower()

# find the overlap in the scientific names
overlap = kaggle21_train['scientificName'].isin(kaggle22_train['scientificName'])
print(f"Number of overlapping scientific names: {overlap.sum()}")


# ## GBIF Images

# In[11]:


harvard_image_data = pd.read_csv(GBIF_MULTIMEDIA_DATA, sep="\t")
image_dir = GBIF_INSTALL_PATH   
# harvard_dataset = harvard_image_data[['gbifID', 'family', 'species']]
harvard_image_data.head()


# In[16]:


with open("/projectnb/herbdl/data/GBIF/occurrence.txt") as f:
    gbif_occurrence = f.readlines()

gbif_occurrence = [line.split("\t") for line in gbif_occurrence]




# In[17]:


print(gbif_occurrence[:5])


# In[13]:


gbif_occurence = pd.read_csv("/projectnb/herbdl/data/GBIF/occurrence.txt", sep="\t")
gbif_occurence.head()


# In[18]:


# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(filepaths=KAGGLE_FILEPATHS)

# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues()

# Produce a neat report of the issues found in your dataset
imagelab.report()

