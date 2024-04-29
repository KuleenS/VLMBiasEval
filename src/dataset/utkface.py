from abc import ABC

import os

import json

from typing import List

import pandas as pd

from sklearn.model_selection import train_test_split

from src.dataset.base_dataset import BaseDataset

class UTKFace(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.modes = ["race_gender", "race_age", "age_gender", "age_race", "gender_race", "gender_age"]

        if mode not in self.modes:
            raise ValueError()
    
        self.mode = mode

        self.files = self.grab_pictures(self.input_folder)

        self.data = self.prepare_metadata(self.files)

        self.data["image"] = self.files

        self.predicting_mode = self.mode.split("_")[0]

        self.protected_category_mode = self.mode.split("_")[1]

        if self.predicting_mode == "race":
            self.prompt = "What is the race of the person in the picture?\nA. White\nB. Black\nC. Asian\nD. Indian\nE. Others\nAnswer with the option's letter from the given choices directly."
        elif self.predicting_mode == "age":
            self.prompt = "What is the age of the person in the picture?\nA. Child\nB. Young\nC. Middle-Aged\nD. Senior\nAnswer with the option's letter from the given choices directly."
        elif self.predicting_mode == "gender":
            self.prompt = "What is the gender of the person in the picture?\nA. Male\nB. Female\nAnswer with the option's letter from the given choices directly."

        self.train_images, self.test_eval_images = train_test_split(self.files, test_size=0.2, random_state=0)

        self.eval_images, self.test_images = train_test_split(self.test_eval_images, test_size=0.5, random_state=0)

    def grab_pictures(self, input_folder):
        total_files = []

        for root, dirs, files in os.walk(input_folder):
            for name in files:
                if ".jpg" in name:
                    total_files.append(os.path.join(root, name))
        
        return total_files

    def prepare_metadata(self, list_of_files: List[str]):
        meta_data = [os.path.basename(x).split(".")[0].split("_")[:3] for x in list_of_files]

        meta_data_normalized = [[int(y) if len(y) != 0 else None for y in x] for x in meta_data]

        meta_data = pd.DataFrame(meta_data_normalized, columns = ['age', 'gender', 'race'])

        meta_data["age"] = pd.cut(meta_data["age"], [0, 20, 40, 60, 116], right=True, include_lowest=True, labels=["Child", "Young", "Middle-Aged", "Senior"])

        meta_data["race"] = meta_data["race"].map({0: "White", 1:"Black", 2:"Asian", 3:"Indian", 4: "Others"})

        meta_data["gender"] = meta_data["gender"].map({0: "Male", 1:"Female"})

        meta_data["image"] = list_of_files

        return meta_data

    def generate_dataset_dict(self, split: str):
        if split == "train":
            test_items = self.train_images
        elif split == "eval":
            test_items = self.eval_images
        elif split == "test":
            test_items = self.test_images

        filtered_metadata = self.data[self.data.image.isin(test_items)]

        protected_category = filtered_metadata[self.protected_category_mode]

        labels = filtered_metadata[self.predicting_mode] 

        prompts = [self.prompt]*len(test_items)

        list_of_tuples = list(zip(prompts, test_items, labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        return list_of_dict
    
    def create_zero_shot_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split="test")
        
        with open(os.path.join(self.output_folder, f"zeroshot_utkface_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)
        
    def create_finetuning_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split="train")
        
        with open(os.path.join(self.output_folder, f"train_utkface_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split="eval")
        
        with open(os.path.join(self.output_folder, f"eval_utkface_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split="test")
        
        with open(os.path.join(self.output_folder, f"test_utkface_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)