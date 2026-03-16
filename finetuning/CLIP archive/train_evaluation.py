"""
Script to evaluate the performance of the model on the training data.
"""

import torch

from transformers import (
    AutoTokenizer,
    set_seed,
)
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
import json
import pandas as pd
import os
import logging
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='train_evaluation.log')

# Model definition
MODEL_CPKT = "openai/clip-vit-large-patch14-336"
PRETRAINED_MODEL = "dl4ds/herbaria_foundation_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CPKT, cache_dir=".")
processor = CLIPProcessor.from_pretrained(MODEL_CPKT, cache_dir=".")
model = CLIPModel.from_pretrained(PRETRAINED_MODEL, cache_dir=".")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Dataset
IMAGE_BASE_DIR = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_images"
TEST_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_metadata.json"

metadata = json.load(open(TEST_METADATA))

IMAGE_PATHS = [os.path.join(IMAGE_BASE_DIR, image['file_name']) for image in metadata]

test_annotations = pd.DataFrame(metadata)
test_annotations.head()

root_csv = "/projectnb/herbdl/data/kaggle-herbaria/train_2022_labeled.csv"
out_json = "/projectnb/herbdl/workspaces/smritis/finetuning/training/pairs.json"

dataset = load_dataset("json", data_files=out_json)

TRAIN_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_metadata.json"

train_metadata = json.load(open(TRAIN_METADATA))
categories = train_metadata['categories']

label_df = pd.read_csv(root_csv)
label_df.drop(columns=['Unnamed: 0'], inplace=True)

categories_df = pd.DataFrame(categories)

def map_label_to_category(label):
    label = label.split(" ")
    species, genus, family = label[6][:-1], label[10], label[13][:-1]
    category = categories_df[(categories_df['species'] == species) & (categories_df['genus'] == genus) & (categories_df['family'] == family)].iloc[0]['category_id']
    return category

def label_to_taxons(label):
    label = label.split(" ")
    species, genus, family = label[6][:-1], label[10], label[13][:-1]

    return species, genus, family

def preprocess_image(image_path):
    image = Image.open(image_path)
    
    return processor(images=image, return_tensors="pt",).pixel_values.squeeze()


class ImageDatasetTrain:
    def __init__(self, image_paths, labels):
        self.images = image_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        category = map_label_to_category(label)
        file_name = os.path.basename(path)
        image_id = int(file_name.split(".")[0].split("__")[1])
        return {"pixel_values": preprocess_image(path), "image_id": image_id, "label": label, "category": category, "path": path}

    def sample(self, n):
        indices = np.random.choice(len(self), n, replace=False)
        return [self[i] for i in indices]

train_dataset = ImageDatasetTrain(dataset['train']['image'], dataset['train']['caption'])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

logger.info(f"Number of images: {len(train_dataset)}")

taxons = np.array([label_to_taxons(i) for i in dataset['train']['caption']])

species = taxons[:, 0]
genus = taxons[:, 1]
family = taxons[:, 2]

unique_species = sorted(list(set(species)))
unique_genus = sorted(list(set(genus)))
unique_family = sorted(list(set(family)))

logger.info(f"Unique species: {len(unique_species)}")
logger.info(f"Unique genus: {len(unique_genus)}")
logger.info(f"Unique family: {len(unique_family)}")

unique_captions = sorted(list(set(dataset['train']['caption'])))
logger.info(f"Unique captions: {len(unique_captions)}")

caption_embeddings = []

if not os.path.exists("./caption_embeddings.pt"):
    for caption in unique_captions:
        inputs = processor(text=caption, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            caption_embedding = model.get_text_features(**inputs)
        caption_embeddings.append(caption_embedding.cpu())

    text_features = torch.stack(caption_embeddings).squeeze(1)
    torch.save(caption_embeddings, "caption_embeddings.pt")

else:

    text_features = torch.load('caption_embeddings.pt').to(device)

logger.info(f"Label embeddings size: {text_features.size()}")

model = model.eval()
model.to(device)

submission_lst = []
images = 0

correct_captions = 0
correct_family = 0
correct_genus = 0
correct_species = 0

species_dict = {species: 0 for species in unique_species}
genus_dict = {genus: 0 for genus in unique_genus}
family_dict = {family: 0 for family in unique_family}

if __name__ == '__main__':
    
    for batch in train_loader:
            
            pixel_values = batch['pixel_values'].to(device)
            image_ids = batch['image_id']
            categories = batch['category']
            captions = batch['label']
            #logger.info(f"Image ids: {image_ids}")

            try: 

                with torch.no_grad():

                    image_features = model.get_image_features(pixel_values=batch['pixel_values'].to(device))

                logits = torch.matmul(image_features, text_features.T)

                predictions = logits.argmax(dim=1)

                predicted_captions = [unique_captions[predictions[i].item()] for i in range(len(predictions))]

                predicted_categories = [map_label_to_category(caption) for caption in predicted_captions]
                
                #logger.info(f"Predicted categories: {predicted_categories}")
                #logger.info(f"True categories: {[c.item() for c in categories]}")
                #logger.info(f"Predicted captions: {predicted_captions}")
                #logger.info(f"True captions: {captions}")
                #logger.info("-----")

                correct_captions += sum([categories[i] == predicted_categories[i] for i in range(len(categories))])

                predicted_taxons = [label_to_taxons(caption) for caption in predicted_captions]
                true_taxons = [label_to_taxons(caption) for caption in captions]

                correct_family += sum([predicted_taxons[i][2] == true_taxons[i][2] for i in range(len(true_taxons))])
                correct_genus += sum([predicted_taxons[i][1] == true_taxons[i][1] for i in range(len(true_taxons))])
                correct_species += sum([predicted_taxons[i][0] == true_taxons[i][0] for i in range(len(true_taxons))])

                # count correct species, genus, family
                for i in range(len(categories)):
                    species_dict[true_taxons[i][0]] += 1 if predicted_taxons[i][0] == true_taxons[i][0] else 0
                    genus_dict[true_taxons[i][1]] += 1 if predicted_taxons[i][1] == true_taxons[i][1] else 0
                    family_dict[true_taxons[i][2]] += 1 if predicted_taxons[i][2] == true_taxons[i][2] else 0
                    
                images += len(image_ids)

                if images % 50000 == 0:
                    logger.info(f"Processed {images} images")

            except KeyboardInterrupt:
                logger.info(f"Keyboard interrupt -- processed {images} images")
                break

# Write dicts to file
with open("evaluation_result/species_dict.json", "w") as f:
    json.dump(species_dict, f)

with open("evaluation_result/genus_dict.json", "w") as f:
    json.dump(genus_dict, f)

with open("evaluation_result/family_dict.json", "w") as f:
    json.dump(family_dict, f)

logger.info("===== Evaluation Summary =====")
logger.info(f"Correct captions: {correct_captions}")
logger.info(f"Correct family: {correct_family}")
logger.info(f"Correct genus: {correct_genus}")
logger.info(f"Correct species: {correct_species}")
logger.info(f"Total images: {images}")