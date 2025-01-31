import argparse

import json

from pathlib import Path

import toml

from unbiasae.dataset import dataset_eval_generator

from unbiasae.eval import BaseEvaluateDataset

from experiments.models.debiased_llava import DeBiasedLLaVaEvalModel

def evaluate_dataset(model: DeBiasedLLaVaEvalModel, dataset, mode: str, eval_class: BaseEvaluateDataset, config):

    sae_releases, sae_ids, sae_layers, feature_idxs = config["sae_releases"], config["sae_ids"], config["sae_layers"], config["feature_idxs"]

    interventions = config["interventions"]

    scaling_factors = config["scaling_factors"]

    output_labels = dataset["labels"]

    questions = dataset["data"]

    outputs = []

    for sae_release, sae_id, sae_layer, feature_idx in zip(sae_releases, sae_ids, sae_layers, feature_idxs): 

        model.load_wrapper(sae_release, sae_id, sae_layer, feature_idx)

        for intervention in interventions: 
            for scaling_factor in scaling_factors: 
                
                model_outputs = []

                model.load_intervention(intervention, scaling_factor)

                for item in questions:
                    image, prompt = item["prompt"], item["prompt"]

                    pred = model.predict(image, prompt, output_labels)

                    item["model_id"] = model.model_name

                    item["output"] = output_labels[pred]

                    model_outputs.append(item)
                
                evaluate_output = eval_class.evaluate(model_outputs)

                evaluate_output["model_id"] = model.model_name

                evaluate_output["sae_release"] = sae_release

                evaluate_output["sae_id"] = sae_id

                evaluate_output["sae_layer"] = sae_layer

                evaluate_output["feature_idx"] = feature_idx

                evaluate_output["interventions"] = intervention

                evaluate_output["scaling_factor"] = scaling_factor

                evaluate_output["mode"] = mode

                evaluate_output["dataset"] = dataset

                evaluate_output["include_image"] = model.include_image

                outputs.append(evaluate_output)
    
    return outputs


def main(args):
    with open(args.config, "r") as f:
        data = toml.load(f)

    datasets = data["datasets"]

    output_dir = Path(data["output_dir"])

    model_name = data["model_name"]

    include_image = data["include_image"]

    model = DeBiasedLLaVaEvalModel(model_name, include_image)

    for dataset in datasets:
        
        dataset_config = data[dataset]

        modes = dataset_config["modes"]

        input_folder = Path(dataset_config["input_folder"])

        for mode in modes:
            data_examples, eval_class = dataset_eval_generator(dataset, input_folder, mode, "llava")

            evaluate_output = evaluate_dataset(model, data_examples, eval_class)

            output_file = f"{dataset}_{mode}_{model_name}_{'image_included' if include_image else 'no_image'}.ndjson"
    
            with open(output_dir / output_file, "w") as f:
                for item in evaluate_output:
                    f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    main(args)