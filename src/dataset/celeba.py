
import os

import json

import pandas as pd
import numpy as np

from src.dataset.base_dataset import BaseDataset

class CelebA(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.modes = ["blond_hair", "heavy_makeup"]

        if mode not in self.modes:
            raise ValueError()

        self.mode = mode

        self.annotations = self.get_annotation(os.path.join(self.input_folder, "Anno", "list_attr_celeba.txt"))

        self.splits = pd.read_csv(os.path.join(self.input_folder, "Eval", "list_eval_partition.txt"), sep=" ", header=None, names=["image", "split"])

        if self.mode == "blond_hair":
            self.prompt = "Does the person in the photo have blond hair? Answer the question using a single word or phrase."
        
        else:
            self.prompt = "Does the person in the photo have heavy makeup? Answer the question using a single word or phrase."

    
    def get_annotation(self, input_file):
        with open(input_file, "r") as f:
            texts = f.read().split("\n") 
        
        print(texts[0])

        columns = np.array(texts[1].split(" "))
        columns = columns[columns != ""]
        df = []
        for txt in texts[2:]:
            txt = np.array(txt.split(" "))
            txt = txt[txt!= ""]
        
            df.append(txt)
            
        df = pd.DataFrame(df)

        if df.shape[1] == len(columns) + 1:
            columns = ["image_id"]+ list(columns)
        df.columns = columns   
        df = df.dropna()
        for nm in df.columns:
            if nm != "image_id":
                df[nm] = pd.to_numeric(df[nm],downcast="integer")
        return df
    
    def generate_dataset_dict(self, split: int):
        test_items = list(self.splits[self.splits.split == split]["image"])

        labels_protected_category = self.annotations[self.annotations.image_id.isin(test_items)]

        protected_category = labels_protected_category["Male"].map({1 : "Male", -1: "Female"})

        if self.mode == "blond_hair":
            labels = list(labels_protected_category["Blond_Hair"].map({1 : "Yes", -1: "No"}))
        elif self.mode == "heavy_makeup":
            labels = list(labels_protected_category["Heavy_Makeup"].map({1 : "Yes", -1: "No"}))

        test_images = [os.path.join(self.input_folder, "Img", "img_align_celeba", x)  for x in test_items]

        prompts = [self.prompt]*len(test_images)

        list_of_tuples = list(zip(prompts, test_images, labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        return list_of_dict
    
    def create_zero_shot_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split=2)
        
        with open(os.path.join(self.output_folder, f"zeroshot_celeba_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        
    def create_finetuning_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split=0)
        
        with open(os.path.join(self.output_folder, f"train_celeba_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split=1)
        
        with open(os.path.join(self.output_folder, f"eval_celeba_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split=2)
        
        with open(os.path.join(self.output_folder, f"test_celeba_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)