import argparse
from typing import List

import os

import json

import torch

from tqdm import tqdm

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

import PIL

from PIL import Image

from src.debias.wrapper import InterventionWrapper

FAIRNESS_PROMPT = {
    "base": "",
    "pos": "Remember to take into account both the data in the image and the text to be as fair as possible.",
    "neg": ""
}

def process_image(q, image_file, args):
    try:
        if "paligemma" in args.model_name:
            image = Image.open(image_file).convert('RGB')

            width, height = image.size

            processed_image = Image.new('RGB', (width, height))
        else:
           processed_image = Image.open(image_file)

        if args.model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
            processed_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:"
        elif args.model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            processed_prompt =  f"[INST] <image>\n{q} [/INST]"
        elif args.model_name in ["llava-hf/llava-v1.6-34b-hf"]:
            processed_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"
        elif "paligemma" in args.model_name:
            processed_prompt = f"<image>\n{q}"

    except PIL.UnidentifiedImageError:
        return None, None 

    return processed_prompt, processed_image


def sample_training_data(questions): 
    protected_categories = set([x["protected_category"] for x in questions])

    sampled_data = dict()

    for x in protected_categories:
        sampled_data[x] = []

    i = 0 
    while True:

        if all(len(x) == 2 for x in sampled_data):
            break 
        else:
            pc = questions[i]["protected_category"]

            if len(sampled_data[pc]) != 2: 
                sampled_data[pc].append(questions[i])
            i += 1
    
    return sum([sampled_data[x] for x in sampled_data], [])


def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_individual_model(model, processor, tokenizer, intervention_type: str, feature_idx: int, args):
    wrapper = InterventionWrapper(model, processor, args.include_image, device="cuda:0")

    wrapper.load_sae(release=args.sae_release, sae_id=args.sae_id, layer_idx=args.sae_layer)

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    model_params =  {
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "targ_layer": args.sae_layer,
        "feature_idx": feature_idx,
    }

    for question_file in question_files:
        with open(question_file, "r") as f:
            data = json.loads(f.read())
            
        output_labels = data["labels"]

        questions = data["data"]

        model_outputs = []

        sampled_data = sample_training_data(questions)

        train_image_files = [x["image"] for x in sampled_data]

        train_qs = [x["prompt"] for x in sampled_data]

        train_images = []

        train_prompts = []

        for image_file, q in zip(train_image_files, train_qs):
            processed_image, procssed_prompt = process_image(q, image_file, args)

            train_images.append(processed_image)

            train_prompts.append(procssed_prompt)

        examples = (train_prompts, train_images)

        for scaling_factor in args.scaling_factors:
            module_and_hook_fn = wrapper.get_hook(intervention_type, model_params, scaling_factor, FAIRNESS_PROMPT, examples, args.steering_vector_threshold_bias)

            model_name_clean = args.model_name.replace("/", "-")
            
            output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_{scaling_factor}_answers.json"

            if not os.path.exists(os.path.join(args.output_folder, output_file_name)):
                questions_batched = batch_iterable(questions, args.batch_size)

                for batch in tqdm(questions_batched):
                    image_files = [x["image"] for x in batch]

                    qs = [x["prompt"] for x in batch]

                    images = []

                    prompts = []

                    for image_file, q in zip(image_files, qs):
                        processed_image, procssed_prompt = process_image(q, image_file, args)

                        images.append(processed_image)

                        prompts.append(procssed_prompt)
                    
                    if len(images) != 0 and len(prompts) != 0:
                        g = wrapper.generate(prompts, images, module_and_hook_fn)

                        preds = []

                        for i in g:
                            pred_options_logits = torch.stack([i[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                            pred = pred_options_logits.argmax(dim=-1).item()

                            preds.append(pred)
                        
                        for line, prompt, pred in zip(batch, prompts, preds):
                            line["prompt"] = prompt

                            line["model_id"] = args.model_name

                            line["output"] = output_labels[pred]

                            model_outputs.append(line)
                
                model_name_clean = args.model_path.replace("/", "-")
            
                output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_scaling_factor_{scaling_factor}_intervention_{intervention_type}_feature_idx_{args.feature_idx}_answers.json"

                with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                    json.dump(model_outputs, f)


def eval_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    if "paligemma" in args.model_name:
        processor = AutoProcessor.from_pretrained(args.model_name)

        model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    
    else:
        processor = AutoProcessor.from_pretrained(args.model_name)

        model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    
    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.eval()

    for intervention in args.interventions: 
        for feature_idx in args.feature_idxs: 
            eval_individual_model(model, processor, tokenizer, intervention, feature_idx, args)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--include_image", action="store_true", default=False)
    parser.add_argument("--scaling_factors", nargs="+", type=float, default=[ -2, -3, -4, -5, -6, -7, -8, -10, -20, -40])
    parser.add_argument("--interventions", nargs="+", type=str, default=["constant_sae","constant_steering_vector","conditional_per_input","conditional_per_token","conditional_steering_vector","clamping","conditional_clamping","probe_steering_vector","probe_sae","probe_sae_clamping","probe_steering_vector_clamping","sae_steering_vector"])
    parser.add_argument("--steering_threshold_bias", type=float)
    parser.add_argument("--feature_idxs", nargs="+", type=int)

    args = parser.parse_args()

    eval_model(args)