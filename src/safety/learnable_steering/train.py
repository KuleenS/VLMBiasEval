import argparse

import json

import os

import PIL

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import LlavaNextProcessor, LlavaNextConfig, AutoTokenizer

from src.safety.learnable_steering.learnable_control import LlavaNextForLearnableControl
from src.safety.learnable_steering.fairness_loss import fairness_loss, computeBatchCounts

from src.safety.learnable_steering.stochastic_count_model import StochasticCountModel

def main(args):

    model_name = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = LlavaNextConfig.from_pretrained(model_name)

    processor = LlavaNextProcessor.from_pretrained(model_name)

    processor.tokenizer.padding_side = "left"

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaNextForLearnableControl(config=config, num_regression_layers=args.hidden_layers, dropout=args.dropout, layers=args.layers).to("cuda:0")

    model.train()

    with open(args.protected_category_weights) as f:
            data = json.load(f)
        
    output_labels = data["labels"]

    train_data = data["data"]

    intersectional_groups = set([x["protected_category"] for x in train_data])

    VB_CountModel = StochasticCountModel(len(intersectional_groups),len(train_data), 1)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.regression_model.parameters(), lr = args.lr)

    for epoch in range(args.burn_in_epochs):
        for batch in train_data:

            image_file = batch["image"]

            q = batch["prompt"]

            label = batch["label"]

            try:

                image = Image.open(image_file)
            
            except PIL.UnidentifiedImageError:
                continue
     

            if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:"
            elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                prompt = f"[INST] <image>\n{q} [/INST]"
            elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"

            if image and prompt:
                batch = processor(prompt, images=image, padding=True, return_tensors="pt").to("cuda:0")

                # forward + backward + optimize
                outputs = model.generate(**batch,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                                do_sample=False,
                                temperature=0,
                                top_p=None,
                                num_beams=1,
                            )
                
                g = outputs['scores'][0]

                pred_options_logits = torch.stack([g[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])

                tot_loss = criterion(pred_options_logits, label)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                tot_loss.backward()

                optimizer.step()

    for epoch in range(args.epochs):
        for batch in train_data:
            image_file = batch["image"]

            q = batch["prompt"]

            label = batch["label"]

            protected_category = batch["protected_category"]

            try:

                image = Image.open(image_file)
            
            except PIL.UnidentifiedImageError:
                continue
     

            if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:"
            elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
                prompt = f"[INST] <image>\n{q} [/INST]"
            elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
                prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"

            if image and prompt:
                batch = processor(prompt, images=image, padding=True, return_tensors="pt").to("cuda:0")

            
                VB_CountModel.countClass_hat.detach_()
                VB_CountModel.countTotal_hat.detach_()
            # forward + backward + optimize

                outputs = model.generate(**batch,
                                    max_new_tokens=1,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    do_sample=False,
                                    temperature=0,
                                    top_p=None,
                                    num_beams=1,
                                )
                
                g = outputs['scores'][0]

                pred_options_logits = torch.stack([g[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])

                loss = criterion(outputs, label)

            # update Count model 
                countClass, countTotal = computeBatchCounts(protected_category, intersectional_groups,outputs)
                #thetaModel(stepSize,theta_batch)
                VB_CountModel(args.step_size ,countClass, countTotal,len(train_data), 1)
                
                # fairness constraint 
                lossDF = fairness_loss(args.epsilon_base,VB_CountModel)            
                tot_loss = loss+args.l_fair*lossDF
                
                # zero the parameter gradients
                optimizer.zero_grad()
                tot_loss.backward()
                optimizer.step() 
    
    model.eval()

    with open(args.test_file) as f:
        data = json.load(f)
        
    output_labels = data["labels"]

    test_data = data["data"]

    model_outputs = []

    for batch in test_data:
        image_file = batch["image"]

        q = batch["prompt"]

        label = batch["label"]

        images = []

        prompts = []

        image_file = batch["image"]

        q = batch["prompt"]

        label = batch["label"]

        protected_category = batch["protected_category"]

        try:

            image = Image.open(image_file)
        
        except PIL.UnidentifiedImageError:
            continue
     

        if model_name in ["llava-hf/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-13b-hf"]:
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{q} ASSISTANT:"
        elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            prompt = f"[INST] <image>\n{q} [/INST]"
        elif model_name in ["llava-hf/llava-v1.6-34b-hf"]:
            prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{q}<|im_end|><|im_start|>assistant\n"

        if image and prompt:
            batch = processor(prompt, images=image, padding=True, return_tensors="pt").to("cuda:0")

            outputs = model.generate(**batch,
                                    max_new_tokens=1,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    do_sample=False,
                                    temperature=0,
                                    top_p=None,
                                    num_beams=1,
                                )
            
            g = outputs['scores'][0]

            pred_options_logits = torch.stack([g[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])

            pred = pred_options_logits.argmax(dim=-1).item()
        
            batch["prompt"] = prompt

            batch["model_id"] = model_name

            batch["output"] = output_labels[pred]

            model_outputs.append(batch)
        
        model_name_clean = args.model_path.replace("/", "-")

        model_name_clean = args.model_path.replace("/", "-")
            
        output_file_name = os.path.basename(args.question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
            json.dump(model_outputs, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_folder", type=str)

    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)

    parser.add_argument("--layers", type=int, nargs = "+")
    parser.add_argument("--hidden_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--burn_in_epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--l_fair", type=float, default=0.01)
    parser.add_argument("--epsilon_base", type=float, default=0.2231)
    parser.add_argument("--step_size", type=float, default=0.1)


    args = parser.parse_args()

    main(args)
