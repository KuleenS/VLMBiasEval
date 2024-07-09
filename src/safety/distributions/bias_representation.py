import argparse
import torch
import os
import json
from tqdm import tqdm

import numpy as np

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

import PIL

from PIL import Image

from src.safety.distributions.model_wrapper import LLaVaModelWrapper


def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_model(args):
    model_name = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = LlavaNextProcessor.from_pretrained(model_name)

    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    model.eval()

    layer_range = list(range(args.start, args.end))

    wrapped_model = LLaVaModelWrapper(model, processor)

    for question_file in question_files:

        with open(question_file, "r") as f:
            data = json.loads(f.read())

        questions = data["data"]

        model_outputs = dict()

        for layer in layer_range:
            model_outputs[layer] = []

        for i,batch in tqdm(enumerate(questions)):
            image_file = batch["image"]

            q = batch["prompt"]

            try:
                image = Image.open(image_file)

                if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:"

                elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                    prompt = f"[INST] <image>\n{q} [/INST]"

                elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                    prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"

            except PIL.UnidentifiedImageError:
                continue
            
            inputs = processor(prompt, images=image, return_tensors="pt").to("cuda:0")

            with torch.inference_mode():
                output = wrapped_model.generate_text(inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        num_beams=1,
                        )

            for layer in layer_range:
                model_outputs[layer].append(wrapped_model.get_last_activations(layer)[0, -1, :].cpu().numpy())
            
            if len(model_outputs[layer_range[0]]) > 500:
                for layer in layer_range:
                    combined_output = np.concatenate(model_outputs[layer])

                    model_name_clean = args.model_path.replace("/", "-")

                    output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_layer_{layer}_activations_batch_{i}.npz"

                    np.savez_compressed(os.path.join(args.output_folder, output_file_name), activations = combined_output)

                    model_outputs[layer] = []
        
        for layer in layer_range:
            combined_output = np.concatenate(model_outputs[layer])

            model_name_clean = args.model_path.replace("/", "-")

            output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_layer_{layer}_activations_batch_{i}.npz"

            np.savez_compressed(os.path.join(args.output_folder, output_file_name), activations = combined_output)

            model_outputs[layer] = []
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    args = parser.parse_args()

    eval_model(args)