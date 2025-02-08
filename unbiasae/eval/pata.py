import json

from typing import Dict, List, Any

from unbiasae.eval.base_evaluate_dataset import BaseEvaluateDataset

class PATAEval(BaseEvaluateDataset):

    def __init__(self) -> None:
        super().__init__(header=["dataset", "model", "shots", "mode"])

        self.label_map = {"yes": 1, "no": 0}

        self.outputs = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    def evaluate(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        labels = [x["label"] for x in data]

        outputs = [x["output"] for x in data]

        true_outputs = [1 if output in label[0] else 0 for label, output in zip(labels, outputs)]

        true_labels = [1]*len(true_outputs)

        label_map_inverse = {1 : "yes", 0: "no"}

        for i in range(len(data)):

            data[i]["label"] = label_map_inverse[true_labels[i]] 

            data[i]["output"] = label_map_inverse[true_outputs[i]] 

        return super().evaluate_dataset_binary(data)