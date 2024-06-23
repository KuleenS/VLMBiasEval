"""Tools for tokenizing and manipulating text datasets."""
from PIL import Image

from torch.utils.data import Dataset

class LLaVaDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data 

        self.processor = processor
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image = Image.open(self.data[index]["image"])

        prompt = self.data[index]["prompt"]

        return self.processor(prompt, image, return_tensors="pt")