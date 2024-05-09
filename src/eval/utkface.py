import json

from typing import Dict, List, Any

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.eval.evaluate_dataset import BaseEvaluateDataset

class UTKFaceEval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__()

        self.output_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

        self.race_label = {"white": 0, "black": 1, "asian": 2, "indian": 3, "others": 4}

        self.age_label = {"child": 0, "young": 1, "middle_aged": 2, "senior": 3}

        self.gender_label = {"male": 0, "female": 1}
    
    def evaluate(self, path: str) -> Dict[str, float]:
        with open(path) as f:
            data = json.load(f)

        protected_category = [x["protected_category"] for x in data]

        label_map = None

        if any(substring in path for substring in ["race_gender", "race_age"]):
            label_map = self.race_label
        elif any(substring in path for substring in ["age_gender", "age_race"]):
            label_map = self.age_label
        elif any(substring in path for substring in ["gender_race", "gender_age"]):
            label_map = self.gender_label
             
        label = [label_map[x["label"].lower()] for x in data]

        output = [self.output_map[x["output"].lower()] for x in data]

        result = dict()

        result["demo-difference"] = demographic_parity_difference(label, output, sensitive_features=protected_category)

        result["demo-ratio"] = demographic_parity_ratio(label, output, sensitive_features=protected_category)

        result["eq-odds-ratio"] = None

        result["eq-demo-difference"] = None

        result["accuracy"] = accuracy_score(label, output)

        precision, recall, f1, _ = precision_recall_fscore_support(label, output, labels=list(label_map.values()))

        for i, name in enumerate(label_map.keys()):

            result[f"prec-{name}"] = precision[i]
            result[f"recall-{name}"] = precision[i]
            result[f"f1-{name}"] = precision[i]

        return result
