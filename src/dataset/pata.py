import os

import json

import pickle

import requests

from typing import List

from string import ascii_uppercase

import pandas as pd

from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset

class PATA(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        if mode not in ["races", "gender", "age"]:
            raise ValueError()

        self.mode = mode

        self.question = "Which one is the correct caption of this image?\n"

        self.prompt = "Answer with the option's letter from the given choices directly."

        self.replacements = {"doctor": "helping a patient", "pilot": "flying a plane", "soldier": "being a soldier", "singer": "singing", "photographer": "taking a picture"}

        self.annotations = self.generate_annotations(self.input_folder)

        self.prompt_options, self.prompt_labels = self.generate_prompt_options_labels(self.input_folder)

        if not os.path.exists(os.path.join(self.input_folder, "images")):
            url_to_path_dict = self.download_image_data(self.input_folder, list(self.annotations["url"]))
        else:
            with open(os.path.join(self.input_folder, "url_to_path_map.pickle"), "rb") as f:
                url_to_path_dict = pickle.load(f)

        self.annotations["path"] = self.annotations["url"].map(url_to_path_dict)

        self.annotations = self.annotations[~self.annotations.path.isna()]

        self.outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    def download_image_data(self, input_folder: str, urls: List[str]):
        image_paths = []

        os.makedirs(os.path.join(input_folder, "images"), exist_ok=True)

        for i, image_url in tqdm(enumerate(urls)):

            image_path = os.path.join(input_folder, "images", f"{i}.jpg")

            try:
                r = requests.get(image_url, timeout=15) # 10 seconds
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
    
    def generate_annotations(self, input_folder):
        with open(os.path.join(input_folder, "pata_fairness.files.lst")) as f:
            data = [x.strip().split("|") for x in f.readlines()]
        
        metadata = [x[0].replace("gun_range", "gun range").replace("drinking_water", "drinking water").replace("fashion_show", "fashion show").split("_") for x in data]

        url = [x[1] for x in data]

        df = pd.DataFrame([m + [u] for m, u in zip(metadata, url)], columns=["scene", "races", "gender", "age", "url"])

        return df

    def generate_prompt_options_labels(self, input_folder):
        with open(os.path.join(input_folder, "pata_fairness.captions.json"), "r") as f:
            data = json.load(f)
        
        added_action = {x["short"]: x["long"][x["long"].rfind("} ") + 2:] for x in data}

        for action in added_action:
            if action in self.replacements:
                added_action[action] = self.replacements[action]
        
        prompts = dict()

        prompt_labels = dict()

        for scene in data:

            for prompt_type in ["races", "gender", "age"]:
                positive_examples = scene["pos"][prompt_type]
                negative_examples = scene["neg"][prompt_type]

                total_examples = len(positive_examples) + len(negative_examples)

                options = list(zip(ascii_uppercase[:total_examples], positive_examples + negative_examples))

                prompts[scene["short"] + "_" + prompt_type] = options, added_action[scene["short"]]

                prompt_labels[scene["short"] + "_" + prompt_type] = (ascii_uppercase[:len(positive_examples)], ascii_uppercase[len(positive_examples): total_examples])
            
        return prompts, prompt_labels

    def generate_dataset_dict(self, model: str):
        list_of_tuples = []

        for row in self.annotations.itertuples(): 
            key = row.scene.replace(" ", "_") +"_" + self.mode

            prompt_label = self.prompt_labels[key] 

            options, action = self.prompt_options[key]

            if len(options) != 0:

                if model == "clip":
                    prompt = [x[1] for x in options]
                else:
                    prompt_options = ""

                    for option in options:

                        prompt_options += option[0] + ". " + option[1] + " "+ action + "\n"

                    prompt = self.question + prompt_options + "\n" + self.prompt

                path = row.path

                protected_category = getattr(row, self.mode)

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

        with open(os.path.join(self.output_folder, f"zeroshot_test_pata_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)

    def create_train_clip_dataset(self) -> None:
       raise NotImplementedError()
        
    def create_test_clip_dataset(self) -> None:
        final_data = self.generate_dataset_dict(model="clip")

        with open(os.path.join(self.output_folder, f"clipzeroshot_test_pata_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)