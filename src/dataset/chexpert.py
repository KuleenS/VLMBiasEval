import os

import json

import pandas as pd

from pathlib import Path

from src.dataset.base_dataset import BaseDataset

class CheXpert(BaseDataset):

    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.mode = mode

        self.protected_category_mode, self.prediction_mode = self.mode.split("_")

        self.protected_category_modes = ["age", "sex", "ethnicity"]

        self.TASKS_CHEXPERT = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                 'Pneumothorax', 'Support Devices']
        
        if self.protected_category_mode not in self.protected_category_modes:
            raise ValueError()

        if self.prediction_mode not in self.TASKS_CHEXPERT:
            raise ValueError()
        
        self.prompt = f"Does this patient have {self.prediction_mode}? Answer the question using a single word or phrase."
        
        self.annotations = self.get_annotations(self.input_folder)
    
    def bin_age(self, x):
        if pd.isnull(x): return None
        elif 0 <= x < 18: return "Child"
        elif 18 <= x < 40: return "Young"
        elif 40 <= x < 60: return "Middle-Aged"
        elif 60 <= x: return "Senior"

    
    def get_annotations(self, input_folder): 
        train_df = pd.read_csv(os.path.join(input_folder, "CheXpert-v1.0-small", 'train.csv'))
        
        valid_df = pd.read_csv(os.path.join(input_folder, "CheXpert-v1.0-small", 'valid.csv'))

        test_df = pd.read_csv(os.path.join(input_folder, "CheXpert-v1.0-small", 'test.csv'))

        train_df['filename'] = train_df['Path'].astype(str).apply(lambda x: os.path.join(input_folder, "CheXpert-v1.0-small", x[x.index('/')+1:]))

        valid_df['filename'] = valid_df['Path'].astype(str).apply(lambda x: os.path.join(input_folder, "CheXpert-v1.0-small", x[x.index('/')+1:]))

        test_df['filename'] = test_df['Path'].astype(str).apply(lambda x: os.path.join(input_folder, "CheXpert-v1.0-small", x))

        df = pd.concat([train_df[test_df.columns], valid_df[test_df.columns], test_df], ignore_index=True)

        df = df.assign(split=pd.Series([0] * len(train_df) + [1] * len(valid_df) + [2] * len(test_df)))

        df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)

        details = pd.read_csv(os.path.join(input_folder, 'chexpert_demo.csv'))[['PATIENT', 'GENDER', 'AGE_AT_CXR', 'PRIMARY_RACE']]

        details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

        df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

        df = df[df.GENDER.isin(['Male', 'Female'])]

        df['sex'] = "gender"

        df['age'] = df['AGE_AT_CXR'].apply(self.bin_age)

        df['ethnicity'] = df['PRIMARY_RACE']

        for t in self.TASKS_CHEXPERT:
            # treat uncertain labels as negative
            df[t] = (df[t].fillna(0.0) == 1.0).astype(int)

        return df

    def generate_dataset_dict(self, split: int):

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
        list_of_dict = self.generate_dataset_dict(split=2)
        
        with open(os.path.join(self.output_folder, f"zeroshot_chexpert_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)
        
    def create_finetuning_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split=0)
        
        with open(os.path.join(self.output_folder, f"train_chexpert_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split=1)
        
        with open(os.path.join(self.output_folder, f"test_chexpert_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split=2)
        
        with open(os.path.join(self.output_folder, f"test_chexpert_{self.mode}.json"), "w") as f:
            json.dump(list_of_dict, f)