from datasets import load_dataset

from src.safety.steering.steering_datasets.steering_dataset import SteeringDataset

class SycophancyDataset(SteeringDataset):
    def __init__(self):
        self.DATASETS = [
            'sycophancy_on_nlp_survey',
            'sycophancy_on_philpapers2020',
            'sycophancy_on_political_typology_quiz'
        ]

        self.pairs = []
        for dataset_name in self.DATASETS:
            dataset = load_dataset("EleutherAI/sycophancy", dataset_name)["validation"]

            for question, match, not_match in zip(dataset['question'], dataset['answer_matching_behavior'], dataset['answer_not_matching_behavior']):
                self.pairs.append(('Question: ' + question + match, 'Question: ' + question + not_match))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):

        anti_stereotype, stereotype = self.pairs[index]

        return anti_stereotype, stereotype, None
