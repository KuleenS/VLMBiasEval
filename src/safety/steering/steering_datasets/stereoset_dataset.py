from datasets import load_dataset

from src.safety.steering.steering_datasets.steering_dataset import SteeringDataset

class StereoSetDataset(SteeringDataset):
    def __init__(self, mode: str):
        dataset = load_dataset("McGill-NLP/stereoset", "intersentence")

        dataset_sentences = dataset["validation"]["sentences"]

        dataset_contexts = dataset["validation"]["context"]

        bias_types = dataset["validation"]["bias_type"]

        self.pairs = []

        self.mode = mode

        self.prompt = "{context} Given this context, which statement is true?\nA. {stereotype}\nB. {anti_stereotype}\n Answer with the option's letter from the given choices directly."

        if self.mode not in ["gender", "race"]:
            raise ValueError()

        for context, item, bias_type in zip(dataset_contexts, dataset_sentences, bias_types):

            if self.mode == bias_type:
                sentences = item["sentence"]
                gold_labels = item["gold_label"]


                for i in range(len(gold_labels)):

                    if gold_labels[i] == 0:
                        anti_stereotype = sentences[i]
                    elif gold_labels[i] == 1:
                        stereotype = sentences[i]
                
                prompt_item = self.prompt.format(context=context, stereotype=stereotype, anti_stereotype=anti_stereotype)

                stereotype_prompt = prompt_item + "\n A"
                anti_stereotype_prompt = prompt_item + "\n B"

                self.pairs.append((anti_stereotype_prompt, stereotype_prompt))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):

        anti_stereotype, stereotype = self.pairs[index]

        return anti_stereotype, stereotype, None