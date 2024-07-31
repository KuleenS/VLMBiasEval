import argparse
import torch
import os
import json
from tqdm import tqdm

import PIL

from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_med_llava(args):

    model_path = os.path.expanduser(args.model_path)

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    tokenizer.padding_side = "left"

    tokenizer.pad_token_id = tokenizer.eos_token_id

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    model.eval()

    for question_file in question_files:

        model_name_clean = args.model_path.replace("/", "-")
            
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        if not os.path.exists(os.path.join(args.output_folder, output_file_name)):

            with open(question_file, "r") as f:
                data = json.loads(f.read())
            
            questions = data["data"]

            output_labels = data["labels"]

            model_outputs = []

            for line in tqdm(questions):
                image_file = line["image"]
                qs = line["prompt"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                if args.include_image:
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs   

                conv = conv_templates["mistral_instruct"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                if args.include_image:
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                    try:
                        with Image.open(image_file) as img:
                            image = img.convert('RGB')
                    except PIL.UnidentifiedImageError:
                        continue
                    
                    image_tensor = process_images([image], image_processor, model.config)[0]

                    with torch.inference_mode():
                        output = model.generate(
                                input_ids,
                                images=image_tensor.unsqueeze(0).half().cuda(),
                                image_sizes=[image.size],
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                                do_sample=False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                )
                
                else:
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
                    input_ids = input_ids.unsqueeze(0).cuda()

                    with torch.inference_mode():
                        output = model.generate(
                                input_ids,
                                max_new_tokens=1,
                                output_scores=True,
                                return_dict_in_generate=True,
                                do_sample=False,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                )


                g = output['scores'][0][0]

                pred_options_logits = torch.stack([g[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                pred = pred_options_logits.argmax(dim=-1).item()

                line["prompt"] = prompt

                line["model_id"] = model_name

                line["output"] = output_labels[pred]

                model_outputs.append(line)
            
            model_name_clean = args.model_path.replace("/", "-")
            
            output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

            with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                json.dump(model_outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--include_image", action="store_true", default=False)
    args = parser.parse_args()

    eval_med_llava(args)
