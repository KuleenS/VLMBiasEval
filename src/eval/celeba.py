from typing import Dict, List, Any

from src.eval.evaluate_dataset import BaseEvaluateDataset

class CelebAEval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__()

    def evaluate_dataset_binary(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        return super().evaluate_dataset_binary(data)





        




