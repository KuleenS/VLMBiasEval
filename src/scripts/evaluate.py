import argparse

import os

from typing import Dict, Type

from src.eval import *


def evaluate_output(input_file: str):
    evaluator_map: Dict[str, Type[BaseEvaluateDataset]] = {
        "celeba": CelebAEval,
        "chexpert": CheXpertEval,
        "mimic": MIMICEval,
        "nih": NIHCXREval,
        "padchest": PadChestEval,
        "pata": PATAEval,
        "utkface": UTKFaceEval,
        "vindr": VinDREval,
        "visogender": VisoGenderEval,
        "vlstereo": VLStereoEval,
    }

    for key in evaluator_map:
        if key in input_file:
            evaluator: BaseEvaluateDataset = evaluator_map[key]()
            break

    return evaluator.evaluate(input_file)

def main(args):
    folder = args.input_folder

    model_outputs = [os.path.join(folder, x) for x in os.listdir(folder)]

    for model_output in model_outputs:
        eval_results = evaluate_output(model_output)

        total_path = os.path.basename(model_output)

        print(model_output,  ",".join(str(x) for x in list(eval_results.values())))

            if "pata" in total_path or "visogender" in total_path or "vlstereo" in total_path:
                shots, dataset, mode_model_name, _ = total_path.split("_")

                index = mode_model_name.find("llava")

                mode, model = mode_model_name[:index], mode_model_name[index:]

                header = ["dataset", "model", "shots", "mode"] +  list(eval_results.keys())

                values = [dataset, model, shots, mode] +  list(eval_results.values())
            elif "celeba" in total_path:
                shots, dataset, first_half_mode, second_half_mode_model_name, _ = total_path.split("_")

                index = second_half_mode_model_name.find("llava")

                mode, model = second_half_mode_model_name[:index], second_half_mode_model_name[index:]

                header = ["dataset", "model", "shots", "mode"] +  list(eval_results.keys())

                values = [dataset, model, shots, first_half_mode+mode] +  list(eval_results.values())

            else:
                shots, dataset, protected_category, mode_model_name, _ = total_path.split("_")

                index = mode_model_name.find("llava")

                mode, model = mode_model_name[:index], mode_model_name[index:]

                header = ["dataset", "model", "shots", "mode", "protected_category"] +  list(eval_results.keys())

                values = [dataset, model, shots, mode, protected_category] +  [str(x) for x in list(eval_results.values())]
            
            print(",".join(header))
            print(",".join(str(x) for x in values))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")

    args = parser.parse_args()

    main(args)