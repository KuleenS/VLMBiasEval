import json

from typing import Dict

from src.eval.evaluate_dataset import BaseEvaluateDataset

class CelebAEval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__()

        self.outputs = ["Yes", "No"]

    def evaluate(self, path: str) -> Dict[str, float]:
        with open(path) as f:
            data = json.load(f)

        return super().evaluate_dataset_binary(data)





        




