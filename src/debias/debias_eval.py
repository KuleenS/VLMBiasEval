import argparse

import csv

from typing import Dict, Type, Any, List

import os

import json

import torch

from tqdm import tqdm

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

import PIL

from PIL import Image

from src.debias.wrapper import InterventionWrapper

from src.eval import VisoGenderEval

def evaluate_output(data: List[Dict[str, Any]], dataset: str):

    evaluator = VisoGenderEval()

    return evaluator.evaluate_direct(data, mode=dataset)

def process_image(image_file):
    try:
        processed_image = Image.open(image_file).convert('RGB')

    except PIL.UnidentifiedImageError:
        return None 

    return processed_image


def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_individual_model(model, processor, tokenizer, intervention_type: str, sae_release: str, sae_id: str, sae_layer: str, feature_idx: int, args, writer):
    wrapper = InterventionWrapper(model, processor, args.model_name, args.include_image, device="cuda:0")

    wrapper.load_sae(release=sae_release, sae_id=sae_id, layer_idx=sae_layer)

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    model_params =  {
        "sae_release": sae_release,
        "sae_id": sae_id,
        "targ_layer": sae_layer,
        "feature_idx": feature_idx,
    }

    for question_file in question_files:
        with open(question_file, "r") as f:
            data = json.loads(f.read())
            
        output_labels = data["labels"]

        questions = data["data"]

        model_outputs = []

        for scaling_factor in args.scaling_factors:
            module_and_hook_fn = wrapper.get_hook(intervention_type, model_params, scaling_factor)

            model_name_clean = args.model_name.replace("/", "-")

            sae_release_clean = sae_release.replace("/", "-")

            sae_id_clean = sae_id.replace("/", "-")
        
            output_file_name = os.path.basename(question_file).split(".")[0] + f"_{model_name_clean}_scaling_factor_{scaling_factor}_intervention_{intervention_type}_feature_idx_{feature_idx}_sae_layer_{sae_layer}_sae_id_{sae_id_clean}_sae_release_{sae_release_clean}_answers.json"

            if not os.path.exists(os.path.join(args.output_folder, output_file_name)):
                for item in tqdm(questions):

                    images = []

                    prompts = []

                    processed_image = process_image(item["image"])

                    if item['prompt'] is not None and processed_image is not None:
                        images.append(processed_image)

                        prompts.append(f"<image>\n{item['prompt']}")
                    
                    if len(images) != 0 and len(prompts) != 0:
                        g = wrapper.generate(prompts, images, module_and_hook_fn)

                        preds = []

                        for i in g:
                            pred_options_logits = torch.stack([i[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                            pred = pred_options_logits.argmax(dim=-1).item()

                            preds.append(pred)
                        
                        item["model_id"] = args.model_name

                        item["output"] = output_labels[pred]

                        model_outputs.append(item)
                
                with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                    json.dump(model_outputs, f)

                if "OO" in question_file:
                    eval_output = evaluate_output(model_outputs, "OO")
                else:
                    eval_output = evaluate_output(model_outputs, "OP")
                
                sorted_keys = sorted(eval_output.keys())
                
                sorted_dict = {key: eval_output[key] for key in sorted_keys}

                row = [args.model_name, sae_release, sae_id, sae_layer, feature_idx, intervention_type, scaling_factor] + list(sorted_dict.values())

                writer.writerow(row)
                

        

def eval_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(args.model_name)

    model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    
    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.eval()

    with open(os.path.join(args.output_folder, "results.csv"), "w") as f:
        writer = csv.writer(f)

        header = ["model", "sae_release", "sae_id", "sae_layer", "feature_idx", "intervention", "scale"]

        writer.writerow(header)

        for sae_release, sae_id, sae_layer, feature_idx in zip(args.sae_releases, args.sae_ids, args.sae_layers, args.feature_idxs): 
            for intervention in args.interventions: 
                eval_individual_model(model, processor, tokenizer, intervention, sae_release, sae_id, sae_layer, feature_idx, args, writer)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--include_image", action="store_true", default=False)
    parser.add_argument("--sae_releases", nargs="+", type=str)
    parser.add_argument("--sae_ids", nargs="+", type=str)
    parser.add_argument("--sae_layers", nargs="+", type=int)
    parser.add_argument("--feature_idxs", nargs="+", type=int)
    parser.add_argument("--scaling_factors", nargs="+", type=float, default=[ 2, 3, 4, 5, 6, 7, 8, 10, 20, 40])
    parser.add_argument("--interventions", nargs="+", type=str, default=["constant_sae", "conditional_per_input","conditional_per_token","clamping","conditional_clamping"])
    

    args = parser.parse_args()

    eval_model(args)