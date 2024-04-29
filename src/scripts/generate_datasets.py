import argparse

import toml

from typing import Dict

from src.dataset import *


def generate_zeroshot_dataset(dataset_name: str, input_folder: str, output_folder: str, mode: str):
    dataset_map: Dict[str, BaseDataset] = {
        "celeba": CelebA,
        "chexpert": CheXpert,
        "mimic": MIMIC,
        "nih": NIHCXR,
        "padchest": PadChest,
        "pata": PATA,
        "utkface": UTKFace,
        "vindr": VINDR,
        "visogender": VisoGender,
        "vlstereo": VLStereo,
    }

    dataset: BaseDataset = dataset_map[dataset_name](input_folder, output_folder, mode)

    dataset.create_zero_shot_dataset()

def main(args):
    with open(args.config, "r") as f:
        data = toml.load(f)

    datasets_to_generate = data["datasets"]

    output_folder = data["output_folder"]
    
    for dataset_name in datasets_to_generate:
        modes = data[dataset_name]["modes"]

        for mode in modes:
            generate_zeroshot_dataset(dataset_name, data[dataset_name]["input_folder"], output_folder, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    main(args)