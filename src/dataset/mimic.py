import os 

import json

import pandas as pd

from src.dataset.base_dataset import BaseDataset

class MIMIC(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.mode = mode

        self.protected_category_mode, self.prediction_mode = self.mode.split("_")

        self.protected_category_modes = ["age", "gender", "ethnicity"]

        self.TASKS_MIMIC = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
              'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
              'Pneumothorax', 'Support Devices']

        if self.protected_category_mode not in self.protected_category_modes:
            raise ValueError()

        if self.prediction_mode not in self.TASKS_MIMIC:
            raise ValueError()
        
        self.prompt = f"Does this patient have {self.prediction_mode}? Answer the question using a single word or phrase."
        
        self.annotations = self.get_annotations(self.input_folder)

        self.outputs = ["Yes", "No"]
    
    def ethnicity_mapping(self,x):
        if pd.isnull(x):
            return 3
        elif x.startswith("WHITE"):
            return 0
        elif x.startswith("BLACK"):
            return 1
        elif x.startswith("ASIAN"):
            return 2
        return 3
    
    def bin_age(self, x):
        if pd.isnull(x): return None
        elif 0 <= x < 18: return "Child"
        elif 18 <= x < 40: return "Young"
        elif 40 <= x < 60: return "Middle-Aged"
        elif 60 <= x: return "Senior"
    
    def get_annotations(self, input_folder):
        patients = pd.read_csv(os.path.join(input_folder, "mimiciv", "hosp", "patients.csv.gz"))

        ethnicities = pd.read_csv(os.path.join(input_folder, "mimiciv", "hosp", "admissions.csv.gz")).drop_duplicates(subset=['subject_id']).set_index('subject_id')['race'].to_dict()

        patients['ethnicity'] = patients['subject_id'].map(ethnicities)
       
        labels = pd.read_csv(os.path.join(input_folder, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-negbio.csv.gz'))

        meta = pd.read_csv(os.path.join(input_folder, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-metadata.csv.gz'))

        df = meta.merge(patients, on='subject_id').merge(labels, on=['subject_id', 'study_id'])

        df['age_decile'] = pd.cut(df['anchor_age'], bins=list(range(0, 101, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)

        df['age'] = df['anchor_age'].apply(self.bin_age)

        df['filename'] = df.apply(
                lambda x: os.path.join(
                    input_folder, "MIMIC-CXR-JPG",
                    'files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'
                ), axis=1)
        
        df = df[df.anchor_age > 0]
        
        for t in self.TASKS_MIMIC:
            df[t] = (df[t].fillna(0.0) == 1.0).astype(int)

        split = pd.read_csv(os.path.join(input_folder, "MIMIC-CXR-JPG", 'mimic-cxr-2.0.0-split.csv.gz'))

        df = df.merge(split, on=['dicom_id'])

        return df
    
    def generate_dataset_dict(self, split: str):

        split_items = self.annotations[self.annotations.split == split]

        test_images = list(split_items["filename"])

        labels = list(split_items[self.prediction_mode])

        prompts = [self.prompt] * len(test_images)

        protected_category = list(split_items[self.protected_category_mode])

        list_of_tuples = list(zip(prompts, test_images, labels, protected_category))

        keys = ["prompt", "image", "label", "protected_category"]

        list_of_dict = [
            dict(zip(keys, values))
            for values in list_of_tuples
        ]

        return list_of_dict 
    
    def create_zero_shot_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split="test")

        final_data = {"data": list_of_dict, "labels": self.outputs}
        
        with open(os.path.join(self.output_folder, f"zeroshot_mimic_{self.mode}.json"), "w") as f:
            json.dump(final_data, f)
        
    def create_finetuning_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split="train")
        
        with open(os.path.join(self.output_folder, f"train_mimic_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)
        
        list_of_dict = self.generate_dataset_dict(split="validate")
        
        with open(os.path.join(self.output_folder, f"eval_mimic_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split="test")
        
        with open(os.path.join(self.output_folder, f"test_mimic_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

       