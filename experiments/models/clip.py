import torch

from typing import List

import PIL

from PIL import Image

from transformers import CLIPProcessor, CLIPModel

from experiments.models.model import EvalModel

class CLIPEvalModel(EvalModel):

    def __init__(self, model_name: str) -> None:

        super().__init__(model_name)

        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model = CLIPModel.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    def _process_images(self, image_files: List[str]):
        images = []

        for image_file in image_files:
            try:
                images.append(Image.open(image_file))
            except PIL.UnidentifiedImageError:
                continue

        return images

    def _get_outputs(self, prompt: str, images, include_image: bool):
        if include_image:
            inputs = self.processor(text=prompt, images=images, return_tensors="pt", padding=True).to("cuda")

            with torch.inference_mode():
                outputs = self.model(**inputs)
            
                logits = outputs.logits_per_image.cpu().detach()
        else:
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True).to("cuda")

            with torch.inference_mode():
                outputs = self.model(**inputs)
            
                logits = outputs.logits_per_text.cpu().detach()
        
        return logits.argmax(dim=1).tolist()

    def predict(self, prompt: str, image_files: List[str], include_image: bool):

        images = self._process_images(image_files)

        if len(images) != 0:
            return self._get_outputs(prompt, images, include_image)


        