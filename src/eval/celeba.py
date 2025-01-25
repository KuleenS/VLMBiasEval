from typing import Dict, Any,List

from src.eval.base_evaluate_dataset import BaseEvaluateDataset

class CelebAEval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__(header=["dataset", "model", "shots", "mode"])

        self.outputs = ["Yes", "No"]

    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        return super().evaluate_dataset_binary(data)





        




