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

        self.predicting_mode, self.protected_category_mode = self.mode.split("_")

        if self.predicting_mode == "race":
            self.prompt = "What is the race of the person in the picture?\nA. White\nB. Black\nC. Asian\nD. Indian\nE. Others\nAnswer with the option's letter from the given choices directly."
            self.outputs = ["A", "B", "C", "D", "E"]
            self.clip_outputs = ["white", "black", "asian", "indian", "others"]
        elif self.predicting_mode == "age":
            self.prompt = "What is the age of the person in the picture?\nA. Child\nB. Young\nC. Middle-Aged\nD. Senior\nAnswer with the option's letter from the given choices directly."
            self.outputs = ["A", "B", "C", "D"]
            self.clip_outputs = ["child", "young", "middle-aged", "senior"]

        elif self.predicting_mode == "gender":
            self.prompt = "What is the gender of the person in the picture?\nA. Male\nB. Female\nAnswer with the option's letter from the given choices directly."
            self.outputs = ["A", "B"]
            self.clip_outputs = ["male", "female"]

        self.train_images, self.test_eval_images = train_test_split(self.data, test_size=0.2, random_state=0)

        self.eval_images, self.test_images = train_test_split(self.test_eval_images, test_size=0.5, random_state=0)

    def grab_pictures(self, input_folder):
        total_files = []

        for root, dirs, files in os.walk(input_folder):
            for name in files:
                if ".jpg" in name:
                    total_files.append(os.path.join(root, name))
        
        return total_files

    def prepare_metadata(self, list_of_files: List[str]):
        meta_data = []
        for total_file in list_of_files:
            age, gender, race = os.path.basename(total_file).split("_")[:3]

            meta_data.append([total_file, int(age), int(gender), int(race)])

        meta_data_df = pd.DataFrame(meta_data, columns = ['image', 'age', 'gender', 'race'])

        meta_data_df["age"] = pd.cut(meta_data_df["age"], [0, 20, 40, 60, 116], right=True, include_lowest=True, labels=["Child", "Young", "Middle-Aged", "Senior"])

        meta_data_df["race"] = meta_data_df["race"].map({0: "White", 1:"Black", 2:"Asian", 3:"Indian", 4: "Others"})

        meta_data_df["gender"] = meta_data_df["gender"].map({0: "Male", 1:"Female"})
       
        return meta_data_df

    def generate_dataset_dict(self, prompt: str | List[str], split: int = 2):
        if split == 0:
            filtered_metadata = self.train_images
        elif split == 1:
            filtered_metadata = self.eval_images
        elif split == 2: 
            filtered_metadata = self.test_images

        protected_category = filtered_metadata[self.protected_category_mode]

        labels = filtered_metadata[self.predicting_mode] 

        prompts = [prompt]*len(filtered_metadata)

        list_of_tuples = list(zip(prompts, filtered_metadata["image"], labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        final_data = {"data": list_of_dict, "labels": self.outputs}
        
        return final_data
    
    def create_train_llava_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.prompt, split=0)

        with open(os.path.join(self.output_folder, f"zeroshot_train_utkface_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
    
    def create_test_llava_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.prompt)

        with open(os.path.join(self.output_folder, f"zeroshot_test_utkface_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)

    def create_train_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.clip_outputs, split=0)
        
        with open(os.path.join(self.output_folder, f"clipzeroshot_train_utkface_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
        
    def create_test_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(self.clip_outputs)
        
        with open(os.path.join(self.output_folder, f"clipzeroshot_test_utkface_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)