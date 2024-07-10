import json

from src.safety.steering.steering_datasets.steering_dataset import SteeringDataset


class ProtectedCategoryDataset(SteeringDataset):

    def __init__(self, input_dataset: str):
        with open(input_dataset, "r") as f:
            data = json.loads(f.read())

        self.questions = data["data"]
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):

        q = self.questions[index]["prompt"]

        protected_category = self.questions[index]["protected_category"]

        return q, protected_category, self.questions[index]["image"]
