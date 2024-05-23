import json

from typing import Dict

from src.eval.evaluate_dataset import BaseEvaluateDataset

class VinDREval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__()

        self.label_map = {"yes": 1, "no": 0}

        self.outputs = ["Yes", "No"]
    
    def evaluate(self, path: str) -> Dict[str, float]:
        with open(path) as f:
            data = json.load(f)

        label_map_inverse = {1 : "yes", 0: "no"}

        for i in range(len(data)):

            data[i]["label"] = label_map_inverse[data[i]["label"]]
            data[i]["output"] = self.outputs[data[i]["output"]]
        
        data = [x for x in data if x["protected_category"] != ""]

        return super().evaluate_dataset_binary(data)



