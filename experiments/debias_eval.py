import argparse

import json

from io import TextIOWrapper

from pathlib import Path

import toml

from unbiasae.dataset import dataset_eval_generator

from unbiasae.eval import *

from experiments.models.debiased_llava import DeBiasedLLaVaEvalModel

def evaluate_dataset(model: DeBiasedLLaVaEvalModel, dataset, mode: str, eval_class: BaseEvaluateDataset, config, file_out: TextIOWrapper, max_new_tokens: int):

    sae_releases, sae_ids, sae_layers, feature_idxs = config["sae_releases"], config["sae_ids"], config["sae_layers"], config["feature_idxs"]

    interventions = config["interventions"]

    scaling_factors = config["scaling_factors"]

    output_labels = dataset["labels"]

    questions = dataset["data"]

    for sae_release, sae_id, sae_layer, feature_idx in zip(sae_releases, sae_ids, sae_layers, feature_idxs): 

        model.load_wrapper(sae_release, sae_id, sae_layer, feature_idx)

        for intervention in interventions: 
            for scaling_factor in scaling_factors: 
                
                model_outputs = []

                model.load_intervention(intervention, scaling_factor)

                for item in questions:
                    prompt, image  = item["prompt"], item["image"]

                    pred = model.predict(prompt, image, output_labels, max_new_tokens)

                    if pred is not None:

                        item["model_id"] = model.model_name

                        if max_new_tokens is None:
                            item["output"] = output_labels[pred]

                        else:
                            item["output"] = pred

                        model_outputs.append(item)
                
                if max_new_tokens is None:
                    if isinstance(eval_class, UTKFaceEval) or isinstance(eval_class, VisoGenderEval):
                        evaluate_output =  eval_class.evaluate(model_outputs, mode=mode)
                    else:
                        evaluate_output =  eval_class.evaluate(model_outputs)
                        
                    evaluate_output["model_id"] = model.model_name

                    evaluate_output["sae_release"] = sae_release

                    evaluate_output["sae_id"] = sae_id

                    evaluate_output["sae_layer"] = sae_layer

                    evaluate_output["feature_idx"] = feature_idx

                    evaluate_output["interventions"] = intervention

                    evaluate_output["scaling_factor"] = scaling_factor

                    evaluate_output["mode"] = mode

                    evaluate_output["include_image"] = model.include_image
            
                    file_out.write(json.dumps(evaluate_output) + "\n")

                    file_out.flush()
                
                else:
                    output = {"outputs": model_outputs}

                    output["model_id"] = model.model_name

                    output["sae_release"] = sae_release

                    output["sae_id"] = sae_id

                    output["sae_layer"] = sae_layer

                    output["feature_idx"] = feature_idx

                    output["interventions"] = intervention

                    output["scaling_factor"] = scaling_factor

                    output["mode"] = mode

                    output["include_image"] = model.include_image
            
                    file_out.write(json.dumps(output) + "\n")

                    file_out.flush()

def main(args):
    with open(args.config, "r") as f:
        data = toml.load(f)

    datasets = data["datasets"]

    output_dir = Path(data["output_dir"])

    model_name = data["model_name"]

    include_image = data["include_image"]

    max_new_tokens = data.get("max_new_tokens", None)

    model = DeBiasedLLaVaEvalModel(model_name, include_image)

    for dataset in datasets:
        
        dataset_config = data[dataset]

        modes = dataset_config["modes"]

        input_folder = Path(dataset_config["input_folder"])

        for mode in modes:
            data_examples, eval_class = dataset_eval_generator(dataset, input_folder, mode, "llava")

            output_file = f"{dataset}_{mode}_{model_name.replace('/', '-')}_{'image_included' if include_image else 'no_image'}_debias.ndjson"

            with open(output_dir / output_file, "w") as f:
                evaluate_dataset(model, data_examples, mode, eval_class, data, f, max_new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    main(args)