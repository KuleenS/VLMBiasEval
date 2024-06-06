import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

import PIL

from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_med_llava(args):

    model_path = os.path.expanduser(args.model_path)

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    tokenizer.padding_side = "left"

    tokenizer.pad_token_id = tokenizer.eos_token_id

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    for question_file in question_files:

        with open(question_file, "r") as f:
            data = json.loads(f.read())
        
        questions = data["data"]

        output_labels = data["labels"]

        model_outputs = []

        for line in tqdm(questions):
            image_file = line["image"]
            qs = line["prompt"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates["mistral_instruct"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            try:
                with Image.open(image_file) as img:
                    image = img.convert('RGB')
            except PIL.UnidentifiedImageError:
                continue
            
            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                with torch.inference_mode():
                    output = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            max_new_tokens=1,
                            output_scores=True,
                            return_dict_in_generate=True,
                            do_sample=False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            )

            g = output['scores'][0][0]

            pred_options_logits = torch.stack([g[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
            pred = pred_options_logits.argmax(dim=-1).item()

            line["prompt"] = prompt

            line["model_id"] = model_name

            line["output"] = output_labels[pred]

            model_outputs.append(line)
        
        model_name_clean = args.model_path.replace("/", "-")
        
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
            json.dump(model_outputs, f)
        

def eval_model(args):
    if args.model_path == "microsoft/llava-med-v1.5-mistral-7b":
        
        eval_med_llava(args)
    
    else:
        model_name = args.model_path

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token_id = tokenizer.eos_token_id

        processor = LlavaNextProcessor.from_pretrained(model_name)

        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    
        processor.tokenizer.padding_side = "left"

        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

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

                        if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                            prompts.append(f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:")
                        elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                            prompts.append(f"[INST] <image>\n{q} [/INST]")
                        elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                            prompts.append(f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n")

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
    args = parser.parse_args()

    eval_model(args)
