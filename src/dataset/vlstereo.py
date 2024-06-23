import os 

import json

from typing import List

import requests

import pickle

import pandas as pd

from tqdm import tqdm 
from src.dataset.base_dataset import BaseDataset

class VLStereo(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        if mode not in ["race", "profession", "gender", "religion"]:
            raise ValueError()

        self.mode = mode

        self.question = "Which one is the correct caption of this image?\n"

        self.prompt = "Answer with the option's letter from the given choices directly."

        self.annotations = pd.read_csv(os.path.join(self.input_folder, "data", "VLStereoSet.csv"))

        if not os.path.exists(os.path.join(self.input_folder, "images")):
            url_to_path_dict = self.download_image_data(self.input_folder, list(self.annotations["Imaeg URL"]))
        else:
            with open(os.path.join(self.input_folder, "url_to_path_map.pickle"), "rb") as f:
                url_to_path_dict = pickle.load(f)
        
        self.annotations["path"] = self.annotations["Imaeg URL"].map(url_to_path_dict)

        self.annotations = self.annotations[~self.annotations.path.isna()]

        self.label_to_choice = {0: "A", 1: "B", 2: "C"}

        self.outputs = ["A", "B", "C"]

    def download_image_data(self, input_folder: str, urls: List[str]):
        image_paths = []

        os.makedirs(os.path.join(input_folder, "images"), exist_ok=True)

        for i, image_url in tqdm(enumerate(urls)):

            image_path = os.path.join(input_folder, "images", f"{i}.jpg")

            try:
                with requests.get(image_url, timeout=15) as r:
                    with open(image_path, 'wb') as f:
                        f.write(r.content)
                image_paths.append((image_url, image_path))
            except requests.exceptions.Timeout:
                print("Timed out")
                image_paths.append((image_url, None))
            except requests.exceptions.SSLError:
                print("SSL Error")
                image_paths.append((image_url, None))
            except requests.exceptions.ConnectionError:
                print("Connection Error")
                image_paths.append((image_url, None))
            except requests.exceptions.TooManyRedirects:
                print("Redirect Error")
                image_paths.append((image_url, None))
        
        url_to_path_dict = dict(image_paths)

        with open(os.path.join(input_folder, "url_to_path_map.pickle"), "wb") as f:
            pickle.dump(url_to_path_dict, f)
        
        return url_to_path_dict
    
    def generate_dataset_dict(self, model: str):
        filtered_set = self.annotations[self.annotations.bias_type == self.mode]

        list_of_tuples = []

        for row in filtered_set.itertuples(): 

            path = row.path

            prompt_label = self.label_to_choice[row.label]

            protected_category = row.target

            if model == "clip":
                prompt = [row.stereotype, getattr(row, '_5'), row.unrelated]
            else:

                prompt = self.question + f"A. {row.stereotype}\nB. {getattr(row, '_5')}\nC. {row.unrelated}\n" + self.prompt

            list_of_tuples.append([prompt, path, prompt_label, protected_category])

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        final_data = {"data": list_of_dict, "labels": self.outputs}

        return final_data

    def create_train_llava_dataset(self) -> None:
        raise NotImplementedError()
    
    def create_test_llava_dataset(self) -> None:
        final_data = self.generate_dataset_dict(model="llava")

        with open(os.path.join(self.output_folder, f"zeroshot_test_vlstereo_{self.text_mode}.json"), "w") as f:
            json.dump(final_data, f)

    def create_train_clip_dataset(self) -> None:
       raise NotImplementedError()
        
    def create_test_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(model="clip")

        with open(os.path.join(self.output_folder, f"clipzeroshot_test_vlstereo_{self.text_mode}.json"), "w") as f:
            json.dump(final_data, f)