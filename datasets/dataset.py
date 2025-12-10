import os

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from transformers import AutoImageProcessor


class HerbariaClassificationDataset(Dataset):
    def __init__(self, annotations_file: str, image_dir: str, name: str, label: str, transform=None, image_processor=None):
        """
        Args:

            annotations_file (string): 

                Path to the csv file with annotations. 
                Columns: image_id, filename, caption, scientificName, family, genus, species, scientificNameEncoded

            image_dir (string): Directory with all the images.

            name (string): Name of the dataset.

            label (string): Label column name in the annotations file.

            transform (callable, optional): Optional transform to be applied

        """
        self.img_labels = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.name = name
        self.transform = transform

        self.images = self.img_labels['filename'].tolist()
        self.image_ids = self.img_labels['image_id'].tolist()
        self.label_column = label
        try: 
            self.labels = self.img_labels[self.label_column].tolist()
        except KeyError:
            raise KeyError(f"Failed to initialize dataset. Label '{label}' not found in annotations file. Available columns: {self.img_labels.columns.tolist()}")

        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.image_dir + self.images[idx]
        label = self.labels[idx]
        image_id = self.image_ids[idx]
        return {"pixel_values": self.preprocess_image(path), "image_id": image_id, "label": label, "path": path}

    def convert_to_jsonl(self):
        self.img_labels

    def preprocess_image(self, image_path):
        if os.path.exists(image_path) is False:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)
    
        return self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze()

    def sample(self, n):
        indices = np.random.choice(len(self), n, replace=False)
        return [self[i] for i in indices]

    def __repr__(self):
        return f"{self.name} dataset with {len(self.img_labels)} samples"

    def describe(self):
        statistics = ''

        statistics += f"Number of samples: {len(self.img_labels)}\n"
        statistics += f"Columns: {self.img_labels.columns.tolist()}\n"
        statistics += f"Label distribution:\n{self.img_labels[self.label_column].value_counts().nlargest(10)}\n"
        statistics += f"Average samples per class: {self.img_labels[self.label_column].value_counts().mean()}\n\n"

        statistics += f"Image shape: {self.preprocess_image(self.image_dir + self.images[0]).shape}\n"

        return statistics
        
if __name__ == "__main__":
    from constants import KAGGLE_HERBARIUM_21_TRAIN_CSV, KAGGLE_HERBARIUM_21_TRAIN

    MODEL_CPKT = "microsoft/swinv2-large-patch4-window12-192-22k"
    image_processor = AutoImageProcessor.from_pretrained(MODEL_CPKT)

    dataset = HerbariaClassificationDataset(KAGGLE_HERBARIUM_21_TRAIN_CSV, KAGGLE_HERBARIUM_21_TRAIN, name="Kaggle Herbaria 21 Train", label="scientificNameEncoded", image_processor=image_processor)

    print(dataset)

    print(dataset.describe())


