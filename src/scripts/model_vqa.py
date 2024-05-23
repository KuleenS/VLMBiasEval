import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

import PIL

from PIL import Image

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_model(args):
    model_name = args.model_path

    processor = LlavaNextProcessor.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"

    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

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

            qs = [x["prompt"] for x in batch]

            images = []

            prompts = []

            for image_file, q in zip(image_files, qs):
                try:
                    images.append(Image.open(image_file))
                    prompts.append("[INST] <image>\n"+ q +" [/INST]")
                except PIL.UnidentifiedImageError:
                    continue
            
            if len(images) != 0 and len(prompts) != 0:

                inputs = processor(prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")

                with torch.inference_mode():
                    output = model.generate(**inputs,
                            max_new_tokens=1,
                            output_scores=True,
                            return_dict_in_generate=True,
                            do_sample=False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            )
                
                g = output['scores'][0]

                preds = []

                for i in g:
                    pred_options_logits = torch.stack([i[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                    pred = pred_options_logits.argmax(dim=-1).item()

                    preds.append(pred)
                
                for line, prompt, pred in zip(batch, prompts, preds):
                    line["prompt"] = prompt

                    line["model_id"] = model_name

                    line["output"] = pred

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
