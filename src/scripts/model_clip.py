import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

import PIL

from PIL import Image

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_model(args):
    model_name = args.model_path

    processor = CLIPProcessor.from_pretrained(model_name)

    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    
    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    for question_file in question_files:

        with open(question_file, "r") as f:
            data = json.loads(f.read())
        
        output_labels = data["labels"]

        questions = data["data"]

        model_outputs = []

        questions_batched = batch_iterable(questions, args.batch_size)

        for batch in tqdm(questions_batched):
            image_files = [x["image"] for x in batch]

            prompts = [x["prompt"] for x in batch]

            images = []

            for image_file in zip(image_files):
                try:
                    images.append(Image.open(image_file))
                except PIL.UnidentifiedImageError:
                    continue
            
            if len(images) != 0:

                inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda")

                with torch.inference_mode():
                    outputs = model(**inputs)
                
                logits_per_image = outputs.logits_per_image.cpu().detach()
                
                preds = logits_per_image.argmax(dim=1).tolist()

                for line, prompt, pred in zip(batch, prompts, preds):
                    line["prompt"] = prompt

                    line["model_id"] = model_name

                    line["output"] = output_labels[pred]

                    model_outputs.append(line)
        
        model_name_clean = args.model_path.replace("/", "-")
        
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
            json.dump(model_outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
