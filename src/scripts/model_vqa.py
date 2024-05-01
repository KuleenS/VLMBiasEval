import argparse
import torch
import os
import json
from tqdm import tqdm

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import PIL
from PIL import Image

import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    for question_file in question_files:

        with open(question_file, "r") as f:
            questions = json.loads(f.read())

        model_outputs = []

        for line in tqdm(questions):
            image_file = line["image"]
            qs = line["prompt"]
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
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
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            line["prompt"] = cur_prompt

            line["model_id"] = model_name

            line["output"] = outputs

            model_outputs.append(line)
        
        model_name_clean = args.model_path.replace("/", "-")
        
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
            json.dump(model_outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
