from pathlib import Path

from typing import Any, Dict, List

from unbiasae.dataset.visogender import VisoGender

class AdversarialVisoGender(VisoGender):

    def __init__(self, input_folder: Path, mode: str) -> None:
        super().__init__(input_folder, mode)
    
    def _perturb_oo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        counter_factual_occ = [x["label"] for x in data["data"]]

        for i, occ in enumerate(counter_factual_occ):

            not_prepared_prompt = data["data"][i]["prompt"]

            if occ == "masculine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a woman?")

            elif occ == "feminine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a man?")
            
            data["data"][i]["prompt"] = finished_prompt
        
        return data

    def _perturb_op(self, data: Dict[str, Dict[str, str] | List]) -> Dict[str, Dict[str, str] | List]:
        counter_factual_occ = [x["label"] for x in data["data"]]
        counter_factual_participant = [x["other_gender"] for x in data["data"]]

        for i, (occ, part) in enumerate(zip(counter_factual_occ, counter_factual_participant)):

            not_prepared_prompt = data["data"][i]["prompt"]

            if occ == "masculine" and part == "masculine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a woman helping a woman?")
            elif occ == "feminine" and part == "feminine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a man helping a man?")
            elif occ == "masculine" and part == "feminine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a woman helping a man?")
            elif occ == "feminine" and part == "masculine":
                finished_prompt = not_prepared_prompt.replace("image?", " image of a man helping a woman?")
            
            data["data"][i]["prompt"] = finished_prompt
        
        return data
    
    def create_train_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        raise NotImplementedError()
    
    def create_test_llava_dataset(self) -> Dict[str, Dict[str, str] | List]:
        final_data = self._generate_dataset_dict(model="llava")

        if self.text_mode == "OO":
            final_data = self._perturb_oo(final_data)
        else:
            final_data = self._perturb_op(final_data)

        return final_data

    def create_train_clip_dataset(self) -> Dict[str, Dict[str, str] | List]:
       raise NotImplementedError()
        
    def create_test_clip_dataset(self) ->  Dict[str, Dict[str, str] | List]:
        final_data = self._generate_dataset_dict(model="clip")

        if self.text_mode == "OO":
            final_data = self._perturb_oo(final_data)
        else:
            final_data = self._perturb_op(final_data)

        return final_data