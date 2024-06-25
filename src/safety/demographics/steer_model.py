import argparse

import os

import json

from datasets import load_dataset

import PIL

from PIL import Image

import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

from tqdm import tqdm

from model_wrapper import LLaVaModelWrapper


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

    wrapped_model = LLaVaModelWrapper(model, processor)

    with open(args.train_file, "r") as f:
        data = json.loads(f.read())
    
    output_labels = data["labels"]

    questions = data["data"]    

    if not os.path.exists(f"demographics_all_diffs_layer_{args.layer}.pt"):

        layers = list(range(21, 30))

        filenames = dict([(layer, f'demographics_all_diffs_layer_{layer}.pt') for layer in layers])
        diffs = dict([(layer, []) for layer in layers])


        for batch in tqdm(questions):
            image_file = batch["image"]

            q = batch["prompt"]

            protected_category = batch["protected_category"]

            postive_prompt = q + f"Remember that the person in the patient is {protected_category}"

            negative_prompt = q

            try:
                image = Image.open(image_file)

                if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                    postive_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{postive_prompt} ASSISTANT:"
                    negative_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{negative_prompt} ASSISTANT:"

                elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                    postive_prompt = f"[INST] <image>\n{postive_prompt} [/INST]"
                    negative_prompt = f"[INST] <image>\n{negative_prompt} [/INST]"

                elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                    postive_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{postive_prompt}<|im_end|><|im_start|>assistant\n"
                    negative_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{negative_prompt}<|im_end|><|im_start|>assistant\n"

            except PIL.UnidentifiedImageError:
                continue
        
            matching_input = processor(postive_prompt, image, return_tensors="pt").to("cuda")

            not_matching_input = processor(negative_prompt, image, return_tensors="pt").to("cuda")

            s_out = wrapped_model.get_logits(matching_input)

            for layer in layers:
                s_activations = wrapped_model.get_last_activations(layer)
                s_activations = s_activations[0, -2, :].detach().cpu()
                diffs[layer].append(s_activations)
            
            n_out = wrapped_model.get_logits(not_matching_input)
        
            for layer in layers:
                n_activations = wrapped_model.get_last_activations(layer)
                n_activations = n_activations[0, -2, :].detach().cpu()
                diffs[layer][-1] -= n_activations
                    
        for layer in layers:
            diffs[layer] = torch.stack(diffs[layer])
            torch.save(diffs[layer], filenames[layer])
    
    for steering_vector_layer in args.layers:

        vec_data = torch.load(f"demographics_all_diffs_layer_{steering_vector_layer}.pt")

        vec = vec_data.mean(dim=0)
        unit_vec = vec / torch.norm(vec, p=2)
    
        for multiplier in [0.5, 1, 2, 5, 10]:
            wrapped_model.reset_all()
            wrapped_model.set_add_activations(steering_vector_layer, multiplier * unit_vec.cuda())
            
            with open(args.test_file, "r") as f:
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
                    
                    output_file_name = os.path.basename(args.test_file).split(".")[0] + f"demographic_steered_{model_name_clean}_answers_{steering_vector_layer}_{multiplier}.json"

                    with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                        json.dump(model_outputs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs = "+")

    args = parser.parse_args()

    main(args)
