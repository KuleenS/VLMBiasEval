import base64
import io

import PIL

from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

from experiments.models.model import EvalModel

class GeminiEvalModel(EvalModel):

    def __init__(self, model_name: str) -> None:

        super().__init__(model_name)
        
        vertexai.init(project="api2024-421420", location="us-central1")

        self.model = GenerativeModel(
            self.model_name,
        )
    
    def image_to_base64(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _process_images_and_prompts(self, prompt: str, image_file: str, include_image: bool):

        image = None

        if include_image:
            try:
                image = Image.open(image_file)
                image.verify()  # Verify that the image is not corrupted
                image = Image.open(image_file)  # Reopen image after verification
            except (PIL.UnidentifiedImageError, IOError) as e:
                image = None
        
        else:
            prompt = prompt.replace('the correct caption of this image?', 'most likely the correct sentence?')

        prompt += " Answer with just the letter that corresponds to the correct answer."

        return prompt, image

    def _generate(self, prompt, image):
        image = self.image_to_base64(image)
        try:
            responses = self.model.generate_content(
                [prompt, image],
                generation_config={
                    "max_output_tokens": 1,
                }
            )

            # Check if the content is blocked
            if responses.candidates[0].finish_reason == FinishReason.SAFETY:
                print("Content blocked by safety filters")
                return 'Content blocked by safety filters'

            return responses.text
        except Exception as e:
            print(f"Error in generate function: {e}")
            # Handle other unexpected errors
            return 'Error'
    
    def _xgenerate(self, prompt):
        try:
            safe = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
            responses = self.model.generate_content(
                [prompt],
                generation_config={
                    "max_output_tokens": 1,
                },
                safety_settings=safe
            )

            # Check if the content is blocked
            if responses.candidates[0].finish_reason == FinishReason.SAFETY:
                print("Content blocked by safety filters")
                return 'Content blocked by safety filters'

            return responses.text
        except Exception as e:
            print(f"Error in generate function: {e}")
            # Handle other unexpected errors
            return 'Error'

    def predict(self, prompt: str, image_file: str, include_image: bool):

        prompt, image = self._process_images_and_prompts(prompt, image_file, include_image)

        if include_image and image is not None:
            return self._generate(prompt, image)
        elif not include_image:
            return self._xgenerate(prompt, image)
        