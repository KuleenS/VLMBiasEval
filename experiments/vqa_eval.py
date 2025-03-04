import argparse

import json

from typing import List, Dict

from pathlib import Path

from tqdm import tqdm

import toml

from unbiasae.dataset import dataset_eval_generator

from unbiasae.eval import *

from experiments.models import *

def model_factory(model_name: str):
    if "clip" in model_name:
        model = CLIPEvalModel(model_name)
    elif "gemini" in model_name:
        model = GeminiEvalModel(model_name)
    else:
        model = LLaVaEvalModel(model_name)
    
    return model

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
def evaluate_model(model: EvalModel, dataset: Dict[str, Dict[str, str] | List], eval: BaseEvaluateDataset, include_image: bool, batch_size: int, mode: str):
    output_labels = dataset["labels"]

    questions = dataset["data"]

    model_outputs = []

    questions_batched = batch_iterable(questions, batch_size)

    for batch in tqdm(questions_batched):
        image_files = [x["image"] for x in batch]

        qs = [x["prompt"] for x in batch]

        if isinstance(model, LLaVaEvalModel):
            preds = model.predict(qs, image_files, output_labels, include_image)
        else:
            preds = model.predict(qs, image_files, include_image)
           
        for line, pred in zip(batch, preds):
            line["model_id"] = model.model_name

            line["output"] = output_labels[pred]

            model_outputs.append(line)
    
    if isinstance(eval, UTKFaceEval) or isinstance(eval, VisoGenderEval):
        return eval.evaluate(model_outputs, mode=mode)
    else:
        return eval.evaluate(model_outputs)

def main(args):
    with open(args.config, "r") as f:
        data = toml.load(f)
    
    model_name = data["model_name"]

    datasets = data["datasets"]

    include_image = data["include_image"]

    batch_size = data["batch_size"]

    output_dir = Path(data["output_dir"])

    output = []

    model = model_factory(model_name)

    model_type = "clip" if isinstance(model, CLIPEvalModel) else "llava"

    for dataset in datasets:
        
        dataset_config = data[dataset]

        modes = dataset_config["modes"]

        prompts = dataset_config.get("prompts", [None])

        input_folder = Path(dataset_config["input_folder"])

        for prompt in prompts: 
            for mode in modes:
                data_examples, eval_class = dataset_eval_generator(dataset, input_folder, mode, model_type, prompt)

                evaluate_output = evaluate_model(model, data_examples, eval_class, include_image, batch_size, mode)

                evaluate_output["mode"] = mode

                evaluate_output["prompt"] = prompt

                evaluate_output["model"] = model_name

                evaluate_output["dataset"] = dataset

                evaluate_output["include_image"] = include_image

                print(evaluate_output)

                output.append(evaluate_output)
    
    output_file = f"{model_name.replace('/', '-')}_{'image_included' if include_image else 'no_image'}.ndjson"
    
    with open(output_dir / output_file, "w") as f:
        for item in output:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    main(args)