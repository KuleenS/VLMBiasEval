import argparse

import os

import json

from datasets import load_dataset

import PIL

from PIL import Image

import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

from tqdm import tqdm

from src.safety.steering.control import ControlModel
from src.safety.steering.extract import ControlVector

from src.safety.steering.steering_datasets.demographic_dataset import DemographicDataset
from src.safety.steering.steering_datasets.stereoset_dataset import StereoSetDataset
from src.safety.steering.steering_datasets.sycophancy_dataset import SycophancyDataset
from src.safety.steering.steering_datasets.protected_category_dataset import ProtectedCategoryDataset



def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main(args):
    model_name = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = LlavaNextProcessor.from_pretrained(model_name)

    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")

    dataset = None

    if args.dataset == "demographic":
        dataset = DemographicDataset(args.demographic_file)
    elif args.dataset == "sycophancy":
        dataset = SycophancyDataset()
    elif args.dataset == "stereoset":
        dataset = StereoSetDataset(args.stereoset_mode)
    elif args.dataset == "protected_category":
        dataset = ProtectedCategoryDataset(args.demographic_file)
    
    if args.dataset == "protected_category":

        with open(args.protected_category_weights) as f:
            data = json.load(f)

        control_vectors = ControlVector.train(model, processor, dataset, hidden_layers=args.layers, method=args.reduction_method, bias=True, protected_category_weights=data)
    else:
        control_vectors = ControlVector.train(model, processor, dataset, hidden_layers=args.layers, method=args.reduction_method)

    for steering_vector_layer in args.layers:

        wrapped_model = ControlModel(model, args.layers)

        for multiplier in [0.5, 1, 2, 5, 10, 20, 50]:
            wrapped_model.reset()
            wrapped_model.set_control(control_vectors, multiplier)

            with open(args.question_file, "r") as f:
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
                        output = wrapped_model.generate_text(inputs,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                                do_sample=False,
                                temperature=0,
                                top_p=None,
                                num_beams=1,
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

                    if args.demographic_file:
                        demographic_file = os.path.basename(args.demographic_file).split(".")[0]

                        output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{demographic_file}_{args.dataset}_steered_{model_name_clean}_answers_{steering_vector_layer}_{multiplier}.json"
                    if args.stereoset_mode:
                        output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{args.stereoset_mode}_{args.dataset}_steered_{model_name_clean}_answers_{steering_vector_layer}_{multiplier}.json"
                    else:
                        output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{args.dataset}_steered_{model_name_clean}_answers_{steering_vector_layer}_{multiplier}.json"

                    with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                        json.dump(model_outputs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--demographic_file", type=str, required=False)
    parser.add_argument("--stereoset_mode", type=str, required=False)
    parser.add_argument("--protected_category_weights", type=str, required=False)


    parser.add_argument("--question_file", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--layers", type=int, nargs = "+")
    parser.add_argument("--reduction_method", type=str)


    args = parser.parse_args()

    main(args)
