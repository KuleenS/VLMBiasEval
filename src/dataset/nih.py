import os

import json

from typing import List

import pandas as pd

import numpy as np

from src.dataset.base_dataset import BaseDataset

class NIHCXR(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.mode = mode

        self.protected_category_mode, self.prediction_mode = self.mode.split("_")

        self.protected_category_modes = ["age", "gender"]

        self.TASKS_NIH = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax']

        if self.protected_category_mode not in self.protected_category_modes:
            raise ValueError()

        if self.prediction_mode not in self.TASKS_NIH:
            raise ValueError()
        
        self.prompt = f"Does this patient have {self.prediction_mode}? Answer the question using a single word or phrase."
        
        self.annotations = self.get_annotations(self.input_folder)

        self.outputs = ["Yes", "No"]

        self.clip_outputs = [f"{self.prediction_mode}", f"No {self.prediction_mode}"]
    
    def bin_age(self, x):
        if pd.isnull(x): return None
        elif 0 <= x < 18: return "Child"
        elif 18 <= x < 40: return "Young"
        elif 40 <= x < 60: return "Middle-Aged"
        elif 60 <= x: return "Senior"

    
    def get_annotations(self, input_folder): 
        train_split_path = os.path.join(input_folder, "train_val_list.txt")

        with open(train_split_path, "r") as f:
            train_images = [x.strip() for x in f.readlines()]
        
        df = pd.read_csv(os.path.join(input_folder, "Data_Entry_2017_v2020.csv"))

        df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

        for label in self.TASKS_NIH:
            pathology = label if label != 'Pleural Thickening' else 'Pleural_Thickening'
            df[label] = (df['labels'].apply(lambda x: pathology in x)).astype(int)

        df['age'] = df['Patient Age'].apply(self.bin_age)

        df["gender"] = df["Patient Gender"]

        conditions = [df["Image Index"].isin(train_images), ~df["Image Index"].isin(train_images)]

        df['split'] = np.select(conditions, [1, 2])

        return df

    def generate_dataset_dict(self, prompt: str | List[str], split: int = 2):

        split_items = self.annotations[self.annotations.split == split]

        test_items = list(split_items["Image Index"])

        labels = list(split_items[self.prediction_mode])

        protected_category = list(split_items[self.protected_category_mode])

        test_images = [os.path.join(self.input_folder, "images", x)  for x in test_items]

        prompts = [prompt] * len(test_images)

        list_of_tuples = list(zip(prompts, test_images, labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        final_data = {"data": list_of_dict, "labels": self.outputs}

        return final_data

    def create_train_llava_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.prompt, split=1)

        with open(os.path.join(self.output_folder, f"zeroshot_train_nih_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
    
    def create_test_llava_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.prompt)

        with open(os.path.join(self.output_folder, f"zeroshot_test_nih_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)

    def create_train_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.clip_outputs, split=1)
        
        with open(os.path.join(self.output_folder, f"clipzeroshot_train_nih_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
        
    def create_test_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.clip_outputs)
        
        with open(os.path.join(self.output_folder, f"clipzeroshot_test_nih_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
