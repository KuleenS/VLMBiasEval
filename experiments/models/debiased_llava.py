from typing import List

import torch

import PIL

from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from unbiasae.debias.wrapper import InterventionWrapper

from experiments.models.model import EvalModel

class DeBiasedLLaVaEvalModel(EvalModel):

    def __init__(self, model_name: str, include_image: bool):
        super().__init__(model_name)

        self.include_image = include_image

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, low_cpu_mem_usage=True).to("cuda")

        self.processor.tokenizer.padding_side = "left"

        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        self.wrapper = InterventionWrapper(self.model, self.processor, self.model_name, self.include_image, device="cuda:0")

        self.model_params = None

        self.module_and_hook_fn = None

    def load_wrapper(self, sae_release: str, sae_id: int, sae_layer: str, feature_idx: int):
        self.wrapper.load_sae(release=sae_release, sae_id=sae_id, layer_idx=sae_layer)

        self.model_params =  {
            "sae_release": sae_release,
            "sae_id": sae_id,
            "targ_layer": sae_layer,
            "feature_idx": feature_idx,
        }
    
    def load_intervention(self, intervention_type: str, scaling_factor: float):
        self.module_and_hook_fn = self.wrapper.get_hook(intervention_type, self.model_params, scaling_factor)
    
    def _process_image_and_prompt(self, prompt: str, image_file: str):
        try:
            processed_image = Image.open(image_file).convert('RGB')

        except PIL.UnidentifiedImageError:
            processed_image =  None 
        
        if not self.include_image:
            width, height = processed_image.size

            processed_image = Image.new('RGB', (width, height))
    
        processed_prompt = f"<image>\n{prompt}"

        return processed_prompt, processed_image
    
    def _get_outputs(self, prompt, image, output_labels, max_new_tokens):
        g = self.wrapper.generate(prompt, image, self.module_and_hook_fn, max_new_tokens)

        if max_new_tokens is None:

            preds = []

            for i in g:
                pred_options_logits = torch.stack([i[self.tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                pred = pred_options_logits.argmax(dim=-1).item()

                preds.append(pred)
            
            return preds

        else:
            return self.tokenizer.batch_decode(g.cpu(), skip_special_tokens=True).strip()

    def predict(self, q: str, image_file: str, output_labels: List[str], max_new_tokens: int = None):
        prompt, image = self._process_image_and_prompt(q, image_file)

        if image is not None:
            outputs = self._get_outputs([prompt], [image], output_labels, max_new_tokens)

            if max_new_tokens is None:
                return outputs[0]
            else:
                return outputs
    
        else:
            print(q, image_file, " fail")
            return None