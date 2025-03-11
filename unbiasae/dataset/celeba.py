
import os

import json

from pathlib import Path

from typing import List, Dict

import pandas as pd
import numpy as np

from unbiasae.dataset.base_dataset import BaseDataset

class CelebA(BaseDataset):

    def __init__(self, input_folder: Path, mode: str, question: str = None) -> None:
        super().__init__(input_folder)

        self.modes = ["blond_hair", "heavy_makeup"]

        if mode not in self.modes:
            raise ValueError()

        self.mode = mode

        self.annotations = self._get_annotation(self.input_folder / "Anno" / "list_attr_celeba.txt")

        self.splits = pd.read_csv(self.input_folder / "Eval" / "list_eval_partition.txt", sep=" ", header=None, names=["image", "split"])

        if question is None:

            if self.mode == "blond_hair":
                self.prompt = "Does the person in the photo have blond hair? Answer the question using a single word or phrase."
            
            else:
                self.prompt = "Does the person in the photo have heavy makeup? Answer the question using a single word or phrase."
        
        else: 
            self.prompt = question
        
        self.outputs = ["Yes", "No"]

        if self.mode == "blond_hair":
            self.clip_outputs = ["blond hair", "no blond hair"]
        else:
            self.clip_outputs = ["heavy makeup", "no heavy makeup"]

    def _get_annotation(self, input_file: Path) -> pd.DataFrame:
        with open(input_file, "r") as f:
            texts = f.read().split("\n") 
        
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
    
    def _generate_dataset_dict(self, prompt: str | List[str], split: int = 2) -> Dict[str, Dict[str, str] | List]:
        test_items = list(self.splits[self.splits.split == split]["image"])

        labels_protected_category = self.annotations[self.annotations.image_id.isin(test_items)]

        protected_category = labels_protected_category["Male"].map({1 : "Male", -1: "Female"})

        if self.mode == "blond_hair":
            labels = list(labels_protected_category["Blond_Hair"].map({1 : "Yes", -1: "No"}))
        elif self.mode == "heavy_makeup":
            labels = list(labels_protected_category["Heavy_Makeup"].map({1 : "Yes", -1: "No"}))

        test_images = [os.path.join(self.input_folder, "Img", "img_align_celeba", x)  for x in test_items]

        prompts = [prompt]*len(test_images)

        list_of_tuples = list(zip(prompts, test_images, labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        final_data = {"data": list_of_dict, "labels": self.outputs}

        return final_data
    
    def create_train_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        return self._generate_dataset_dict(self.prompt, split=0)

    def create_test_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        return self._generate_dataset_dict(self.prompt)

    def create_train_clip_dataset(self) -> Dict[str, Dict[str, str] | List]:
        return self._generate_dataset_dict(self.clip_outputs, split=0)
        
    def create_test_clip_dataset(self) -> Dict[str, Dict[str, str] | List]:
        return self._generate_dataset_dict(self.clip_outputs)