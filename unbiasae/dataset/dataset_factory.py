from typing import Dict, List, Tuple

from pathlib import Path

from unbiasae.dataset import *
from unbiasae.eval import *

def dataset_eval_generator(dataset_name: str, input_folder: Path, mode: str, type_of_dataset: str, question: str = None) -> Tuple[Dict[str, Dict[str, str] | List], BaseEvaluateDataset]:
    dataset_map: Dict[str, Tuple[BaseDataset, BaseEvaluateDataset]] = {
        "celeba": (CelebA, CelebAEval),
        "pata": (PATA, PATAEval),
        "utkface": (UTKFace, UTKFaceEval),
        "visogender": (VisoGender, VisoGenderEval),
        "adv_visogender": (AdversarialVisoGender, VisoGenderEval),
        "vlstereo": (VLStereo, VLStereoEval),
    }

    dataset_class, eval_class = dataset_map[dataset_name]

    dataset: BaseDataset = dataset_class(input_folder, mode, question)

    dataset_eval: BaseEvaluateDataset = eval_class()

    if type_of_dataset == "llava":
        return dataset.create_test_llava_dataset(), dataset_eval
    else:
        return dataset.create_test_clip_dataset(), dataset_eval
