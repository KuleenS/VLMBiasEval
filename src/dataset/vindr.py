import os 

import json

import pandas as pd

import pydicom

from tqdm import tqdm 

from src.dataset.base_dataset import BaseDataset

class VINDR(BaseDataset):
    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.mode = mode

        self.prediction_mode, self.protected_category_mode = self.mode.split("_")

        self.protected_category_modes = ["sex", "age"]

        self.TASKS_VINDR = ['Aortic enlargement', 'Atelectasis', 'COPD', 'Calcification', 'Cardiomegaly', 'Clavicle fracture',
              'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
              'Lung cavity', 'Lung cyst', 'Lung tumor', 'Mediastinal shift', 'No finding', 'Nodule/Mass',
              'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumonia', 'Pneumothorax',
              'Pulmonary fibrosis', 'Rib fracture', 'Tuberculosis']
        
        if self.protected_category_mode not in self.protected_category_modes:
            raise ValueError()

        if self.prediction_mode not in self.TASKS_VINDR:
            raise ValueError()
        
        self.prompt = f"Task: Does this patient have {self.prediction_mode}? Answer yes or no\n Answer:"

        self.annotations = self.get_annotations(self.input_folder)

        self.annotations = self.annotations[self.annotations.exists == True]
    
    def bin_age(self, x):
        if pd.isnull(x): return None
        elif 0 <= x < 18: return "Child"
        elif 18 <= x < 40: return "Young"
        elif 40 <= x < 60: return "Middle-Aged"
        elif 60 <= x: return "Senior"
    
    def get_annotations(self, input_folder): 
        train_df = pd.read_csv(os.path.join(input_folder, "annotations", "image_labels_train.csv"))

        test_df = pd.read_csv(os.path.join(input_folder, "annotations", "image_labels_test.csv"))

        train_df['filename'] = train_df['image_id'].astype(str).apply(lambda x: os.path.join(input_folder, 'train', x+'.dicom'))
        train_df['split'] = 0
        # test data no rad_id, only ground truth
        test_df['filename'] = test_df['image_id'].astype(str).apply(lambda x: os.path.join(input_folder, 'test', x+'.dicom'))
        test_df = test_df.rename(columns={'Other disease': 'Other diseases'})
        test_df['split'] = 1

        train_df = train_df.groupby("image_id").agg(pd.Series.mode).reset_index()

        df = pd.concat([train_df, test_df], ignore_index=True)

        df['exists'] = df['filename'].apply(lambda x: os.path.exists(x))

        df['sex'] = None
        df['age_yr'] = None
        for idx, row in tqdm(df.iterrows()):
            dicom_obj = pydicom.filereader.dcmread(row['filename'])
            df.loc[idx, 'sex'] = dicom_obj[0x0010, 0x0040].value
            try:
                df.loc[idx, 'age_yr'] = int(dicom_obj[0x0010, 0x1010].value[:-1])
            except:
                # no age
                pass
        df['age'] = df['age_yr'].apply(self.bin_age)

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
        list_of_dict = self.generate_dataset_dict(split=1)
        
        with open(os.path.join(self.output_folder, f"zeroshot_vindr_{self.mode}.json")) as f:
            json.dump(list_of_dict, f)
        
    def create_finetuning_dataset(self) -> None:
        list_of_dict = self.generate_dataset_dict(split=0)
        
        with open(os.path.join(self.output_folder, f"train_vindr_{self.mode}.json")) as f:
            json.dump(list_of_dict, f)

        list_of_dict = self.generate_dataset_dict(split=1)
        
        with open(os.path.join(self.output_folder, f"test_vindr_{self.mode}.json")) as f:
            json.dump(list_of_dict, f)

