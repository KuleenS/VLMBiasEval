from abc import ABC

class BaseDataset(ABC):

    def __init__(self, input_folder: str, output_folder: str) -> None:
        super().__init__()

        self.input_folder = input_folder

        self.output_folder = output_folder
    
    def create_zero_shot_dataset(self) -> None:
        pass
        
    def create_finetuning_dataset(self) -> None:
        pass