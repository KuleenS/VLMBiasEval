import argparse

import json

import os

import PIL

from PIL import Image

from transformers import (
    LlavaNextForConditionalGeneration, LlavaNextProcessor
)

import torch

from tqdm import tqdm

import numpy as np

from sae import Sae

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main(args):
    sae_model = Sae.load_from_disk(path=args.sae_path, device="cuda")

    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    with open(args.question_file, "r") as f:
        data = json.loads(f.read())
    
    dataset = data["data"]    
  
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    dl = batch_iterable(dataset, n=args.batch_size) 

    pbar = tqdm(dl, desc="Training")

    sae_model.eval()

    model.eval()

    outputs = []

    for i, inputs in enumerate(pbar):
        image_files = [x["image"] for x in inputs]

        qs = [x["prompt"] for x in inputs]

        images = []

        prompts = []

        for image_file, q in zip(image_files, qs):
            try:
                images.append(Image.open(image_file))

                prompts.append(f"[INST] <image>\n{q} [/INST]")

            except PIL.UnidentifiedImageError:
                continue
        
        batch = processor(prompts, images=images, padding=True, return_tensors="pt")

        # Forward pass on the model to get the next batch of activations
        with torch.no_grad():
            hidden_list = model(
                **batch.to("cuda"), output_hidden_states=True
            ).hidden_states[:-1]

            hidden_list = hidden_list[args.layer]

        outputs.append(sae_model.encode(hidden_list).detach().cpu().numpy())

        if len(outputs) > 500:
            combined_output = np.concatenate(outputs)

            model_name_clean = "llava-hf/llava-v1.6-mistral-7b-hf".replace("/", "-")

            output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{model_name_clean}_layer_{args.layer}_sae_activations_batch_{i}.npz"

            np.savez_compressed(os.path.join(args.output_folder, output_file_name), activations = combined_output)

            outputs = []
        
    combined_output = np.concatenate(outputs)

    model_name_clean = "llava-hf/llava-v1.6-mistral-7b-hf".replace("/", "-")

    output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{model_name_clean}_layer_{args.layer}_sae_activations_batch_{i}.npz"

    np.savez_compressed(os.path.join(args.output_folder, output_file_name), activations = combined_output)

    outputs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=str)
    parser.add_argument("--question_file", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layer", type=int)

    args = parser.parse_args()

    main(args)
