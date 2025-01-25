from typing import Dict, List, Any

from src.eval.evaluate_dataset import BaseEvaluateDataset

class NIHCXREval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__(header=["dataset", "model", "shots", "mode", "protected_category"])

        self.label_map = {"yes": 1, "no": 0}

        self.outputs = ["Yes", "No"]
    
    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        label_map_inverse = {1 : "yes", 0: "no"}

        for i in range(len(data)):
            data[i]["label"] = label_map_inverse[data[i]["label"]]

        return super().evaluate_dataset_binary(data)




        




