import torch

from transformers import (
    AutoTokenizer,
    set_seed,
)
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datasets import load_dataset
import json
import pandas as pd
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='evaluation.info')

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
index = 0

unique_captions = sorted(list(set(dataset['train']['caption'])))
logger.info(f"Unique captions: {len(unique_captions)}")

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

def preprocess_image(image_path):
    image = Image.open(image_path)
    
    return processor(images=image, return_tensors="pt",).pixel_values.squeeze()


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        file_name = os.path.basename(path)
        image_id = int(file_name.split(".")[0].split("-")[1])
        return {"pixel_values": preprocess_image(path), "image_id": image_id}


image_dataset = ImageDataset(IMAGE_PATHS)
image_loader = DataLoader(image_dataset, batch_size=8, shuffle=False)

logger.info(f"Number of images: {len(image_dataset)}")

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
images_n = 0

if __name__ == '__main__':
    
    for batch in image_loader:
        
        pixel_values = batch['pixel_values'].to(device)
        image_ids = batch['image_id']
        #logger.info(f"Image ids: {image_ids}")

        try: 
            with torch.no_grad():

                image_features = model.get_image_features(pixel_values=batch['pixel_values'].to(device))

            logits = torch.matmul(image_features, text_features.T)

            predictions = logits.argmax(dim=1)

            predicted_captions = [unique_captions[predictions[i].item()] for i in range(len(predictions))]

            predicted_categories = [map_label_to_category(caption) for caption in predicted_captions]
            
            #logger.info(f"Predicted categories: {predicted_categories}")
            #logger.info("-----")

            submission_lst += [{"image_id": image_ids[i].item(), "category_id": predicted_categories[i]} for i in range(len(image_ids))] 

            images_n += 1
            if images_n % 10000 == 0:
                logger.info(f"Processed {images_n} images")
                
        except KeyboardInterrupt:
            logger.warning(f"Keyboard interrupt -- processed {images_n} images")
            break

    submission_df = pd.DataFrame(submission_lst)
    submission_df.to_csv("submission.csv", index=False)