from abc import ABC

from typing import List, Any, Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio

class BaseEvaluateDataset(ABC):

    def __init__(self) -> None:
        super().__init__()

        self.label_map = {"yes": 1, "no": 0}

    def evaluate(self, path: str) -> Dict[str, float]:
        pass

    def evaluate_direct(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        pass
    
    def evaluate_dataset_binary(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        result = dict()

        protected_category = [x["protected_category"] for x in data]

        label = [self.label_map[x["label"].lower()] for x in data]

        output = [self.label_map[x["output"].lower()] for x in data]

        result["demo-difference"] = demographic_parity_difference(label, output, sensitive_features=protected_category)

        result["demo-ratio"] = demographic_parity_ratio(label, output, sensitive_features=protected_category)

        result["eq-odds-ratio"] = equalized_odds_ratio(label, output, sensitive_features=protected_category)

        result["eq-demo-difference"] = equalized_odds_difference(label, output, sensitive_features=protected_category)

        result["accuracy"] = accuracy_score(label, output)

        precision, recall, f1, _ = precision_recall_fscore_support(label, output, labels=list(self.label_map.values()))

        result["prec-yes"] = precision[0]

        result["prec-no"] = precision[1]

        result["recall-yes"] = recall[1]

        result["recall-no"] = recall[0]

        result["f1-yes"] = f1[1]

        result["f1-no"] = f1[0]

        return result