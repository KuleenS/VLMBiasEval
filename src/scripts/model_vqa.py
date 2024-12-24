import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

import PIL

from PIL import Image

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        

def eval_model(args):
    model_name = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    if "paligemma" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)

        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    
    else:
        processor = LlavaNextProcessor.from_pretrained(model_name)

        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    model.eval()

    for question_file in question_files:

        model_name_clean = args.model_path.replace("/", "-")
        
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        if not os.path.exists(os.path.join(args.output_folder, output_file_name)):

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
                        if "paligemma" in model_name:
                            image = Image.open(image_file).convert('RGB')

                            width, height = image.size

                            images.append(Image.new('RGB', (width, height)))
                        else:
                            images.append(Image.open(image_file))

                        if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                            prompts.append(f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:")
                        elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                            prompts.append(f"[INST] <image>\n{q} [/INST]")
                        elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                            prompts.append(f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n")
                        elif "paligemma" in model_name:
                            prompts.append(f"<image>\n{q}")

                    except PIL.UnidentifiedImageError:
                        continue
                
                if len(images) != 0 and len(prompts) != 0:

                    if "paligemma" in model_name:
                        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(torch.bfloat16).to(model.device)
                        
                    else:

                        if args.include_image:

                            inputs = processor(prompts, images=images, padding=True, return_tensors="pt").to("cuda:0")
                        
                        else:
                            inputs = processor(prompts, padding=True, return_tensors="pt").to("cuda:0")
                        

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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--include_image", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
