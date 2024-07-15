import os 

import json

from typing import List

import pandas as pd

from sklearn.model_selection import train_test_split

from src.dataset.base_dataset import BaseDataset

class PadChest(BaseDataset):
    def __init__(self, input_folder: str, output_folder: str, mode: str) -> None:
        super().__init__(input_folder, output_folder)

        self.mode = mode

        self.protected_category_mode, self.prediction_mode = self.mode.split("_")

        self.protected_category_modes = ["sex", "age"]

        self.TASKS_PADCHEST = ['adenopathy', 'air trapping', 'alveolar pattern', 'aortic atheromatosis', 'aortic button enlargement',
                 'aortic elongation', 'apical pleural thickening', 'artificial heart valve', 'atelectasis',
                 'axial hyperostosis', 'azygos lobe', 'bronchiectasis', 'bronchovascular markings', 'bullas',
                 'calcified adenopathy', 'calcified densities', 'calcified granuloma', 'calcified pleural thickening',
                 'callus rib fracture', 'cardiomegaly', 'cavitation', 'central venous catheter via jugular vein',
                 'central venous catheter via subclavian vein', 'chronic changes', 'consolidation', 'copd signs',
                 'costophrenic angle blunting', 'dai', 'descendent aortic elongation', 'diaphragmatic eventration',
                 'dual chamber device', 'emphysema', 'endotracheal tube', 'fibrotic band', 'flattened diaphragm',
                 'goiter', 'granuloma', 'ground glass pattern', 'gynecomastia', 'heart insufficiency',
                 'hemidiaphragm elevation', 'hiatal hernia', 'hilar congestion', 'hilar enlargement',
                 'hyperinflated lung', 'hypoexpansion', 'hypoexpansion basal', 'increased density', 'infiltrates',
                 'interstitial pattern', 'kyphosis', 'laminar atelectasis', 'lobar atelectasis', 'mammary prosthesis',
                 'mastectomy', 'mediastinal enlargement', 'mediastinic lipomatosis', 'metal',
                 'minor fissure thickening', 'multiple nodules', 'nipple shadow', 'nodule', 'normal', 'nsg tube',
                 'osteopenia', 'osteosynthesis material', 'pacemaker', 'pectum excavatum', 'pleural effusion',
                 'pleural thickening', 'pneumonia', 'pseudonodule', 'pulmonary fibrosis', 'pulmonary mass',
                 'rib fracture', 'sclerotic bone lesion', 'scoliosis', 'single chamber device', 'sternotomy',
                 'suboptimal study', 'superior mediastinal enlargement', 'supra aortic elongation', 'suture material',
                 'tracheal shift', 'tracheostomy tube', 'tuberculosis sequelae', 'unchanged',
                 'vascular hilar enlargement', 'vascular redistribution', 'vertebral anterior compression',
                 'vertebral compression', 'vertebral degenerative changes', 'vertebral fracture', 'volume loss']
        
        if self.protected_category_mode not in self.protected_category_modes:
            raise ValueError()

        if self.prediction_mode not in self.TASKS_PADCHEST:
            raise ValueError()
        
        self.prompt = f"Does this patient have {self.prediction_mode}? Answer the question using a single word or phrase."

        self.data = self.get_annotations(self.input_folder)

        self.outputs = ["Yes", "No"]

        self.clip_outputs = [f"{self.prediction_mode}", f"No {self.prediction_mode}"]

        self.train_images, self.test_eval_images = train_test_split(self.data, test_size=0.2, random_state=0)

        self.eval_images, self.test_images = train_test_split(self.test_eval_images, test_size=0.5, random_state=0)

    def bin_age(self, x):
        if pd.isnull(x): return None
        elif 0 <= x < 18: return "Child"
        elif 18 <= x < 40: return "Young"
        elif 40 <= x < 60: return "Middle-Aged"
        elif 60 <= x: return "Senior"
    
    def get_annotations(self, input_folder): 
        df = pd.read_csv(os.path.join(input_folder, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"))

        df = df[['ImageID', 'StudyID', 'PatientID', 'PatientBirth', 'PatientSex_DICOM', 'ViewPosition_DICOM', 'Projection', 'Labels', 'MethodLabel']]
        df = df[~df['Labels'].isnull()]
        df['filename'] = df['ImageID'].astype(str).apply(lambda x: os.path.join(input_folder, 'images-224', x))
        df = df[df['filename'].apply(lambda x: os.path.exists(x))]
        # df = df[df.Projection.isin(['PA', 'L', 'AP_horizontal', 'AP'])]
        df['Age'] = 2017 - df['PatientBirth']
        df = df[~df['Age'].isnull()]

        df['age'] = df['Age'].apply(self.bin_age)

        df['sex'] = df["PatientSex_DICOM"]

        # filtered df with radiologist labels
        df_filtered = df[df['MethodLabel'] == 'Physician']
        # get unique labels
        unique_labels = set()
        for index, row in df_filtered.iterrows():
            currentLabels = row['Labels']
            try:
                # convert labels str to array
                labels_arr = currentLabels.strip('][').split(', ')
                for label in labels_arr:
                    processed_label = label.split("'")[1].strip()
                    processed_label = processed_label.lower()
                    unique_labels.add(processed_label)
            except:
                continue
        unique_labels = list(unique_labels)
        # multi hot encoding for labels
        dict_list = []
        for index, row in df_filtered.iterrows():
            labels = row['Labels']
            try:
                labels_arr = labels.strip('][').split(', ')
                count_dict = dict()  # map label name to count
                count_dict['ImageID'] = row['ImageID']
                # init count dict with 0s
                for unq_label in unique_labels:
                    count_dict[unq_label] = 0

                if len(labels_arr) > 0 and labels_arr[0] != '':
                    for label in labels_arr:
                        processed_label = label.split("'")[1].strip()
                        processed_label = processed_label.lower()
                        count_dict[processed_label] = 1
                dict_list.append(count_dict)
            except:
                if labels == 'nan':
                    continue
                else:
                    print(f"{index}: error when creating labels for this img.")
                    continue
        multi_hot_labels_df = pd.DataFrame(dict_list, columns=(['ImageID'] + unique_labels))
        df_filtered = df_filtered.merge(multi_hot_labels_df, on='ImageID', how='inner')

        return df_filtered
    
    def generate_dataset_dict(self, prompt: str | List[str], split: int = 2):
        if split == 0:
            filtered_metadata = self.train_images
        elif split == 1:
            filtered_metadata = self.eval_images
        elif split == 2: 
            filtered_metadata = self.test_images
        
        test_images = list(filtered_metadata["filename"])

        labels = list(filtered_metadata[self.prediction_mode])

        prompts = [prompt] * len(test_images)

        protected_category = list(filtered_metadata[self.protected_category_mode])

        list_of_tuples = list(zip(prompts, test_images, labels, protected_category))

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

