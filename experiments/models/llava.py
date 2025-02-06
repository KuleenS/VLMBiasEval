from typing import List

import torch

import PIL

from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from experiments.models.model import EvalModel

class LLaVaEvalModel(EvalModel):

    def __init__(self, model_name: str):

        super().__init__(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, low_cpu_mem_usage=True).to("cuda")

        self.processor.tokenizer.padding_side = "left"

        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
    
    def _tokenize_images_and_prompts(self, prompts: List[str], images, include_image: bool):
        if "paligemma" in self.model_name:
            inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt").to(torch.bfloat16).to("cuda:0")
            
        else:
            if include_image:

                inputs = self.processor(prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
            
            else:
                inputs = self.processor(prompts, padding=True, return_tensors="pt").to("cuda:0")
        
        return inputs
    
    def _process_images_and_prompts(self, questions: List[str], images: List[str], include_image: bool):
        prompts = []

        loaded_images = []

        for q, image_file in zip(questions, images):
            try:

                image =  Image.open(image_file)
            
            except PIL.UnidentifiedImageError:
                continue
        
            if "paligemma" in self.model_name:
                image = image.convert('RGB')

                if not include_image:
                    width, height = image.size

                    loaded_images.append(Image.new('RGB', (width, height)))
                else:
                    loaded_images.append(image)
            else:
                loaded_images.append(image)
            
            if "paligemma" in self.model_name:
                prompts.append(f"<image>\n{q}")
            elif self.model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                prompts.append(f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:")
            elif self.model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                prompts.append(f"[INST] <image>\n{q} [/INST]")
            elif self.model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                prompts.append(f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n")
            
        return prompts, loaded_images

    def _get_outputs(self, inputs, output_labels):
        with torch.inference_mode():
            output = self.model.generate(**inputs,
                        max_new_tokens=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                    )
            
            g = output['scores'][0]

            preds = []

            for i in g:
                pred_options_logits = torch.stack([i[self.tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                pred = pred_options_logits.argmax(dim=-1).item()

                preds.append(pred)
        
        return preds

    def predict(self, qs: List[str], image_files: List[str], output_labels: List[str], include_image: bool):
        prompts, images = self._process_images_and_prompts(qs, image_files, include_image)

        if len(images) != 0 and len(prompts) != 0:
            
            inputs = self._tokenize_images_and_prompts(prompts, images, include_image)

            return self._get_outputs(inputs, output_labels)