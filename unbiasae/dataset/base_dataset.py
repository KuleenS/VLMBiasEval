from abc import ABC

from pathlib import Path

from typing import List, Dict

class BaseDataset(ABC):

    def __init__(self, input_folder: Path) -> None:
        super().__init__()

        self.input_folder = input_folder

    def create_train_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        pass
    
    def create_test_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        pass

    def create_train_clip_dataset(self) -> Dict[str, Dict[str, str] | List]:
        pass
        
    def create_test_clip_dataset(self) -> Dict[str, Dict[str, str] | List]:
        pass