import argparse
import torch
import os
import json
from tqdm import tqdm

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import PIL
from PIL import Image

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

    for question_file in question_files:

        with open(question_file, "r") as f:
            data = json.loads(f.read())
        
        output_labels = data["labels"]

        questions = data["data"]

        model_outputs = []

        questions_batched = batch(questions)

        for batch in tqdm(questions_batched):
            image_files = [x["image"] for x in batch]
            qs = [x["prompt"] for x in batch]
            cur_prompts = qs
            
            for i in range(len(qs)):

                if model.config.mm_use_im_start_end:
                    qs[i] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs[i]
                else:
                    qs[i] = DEFAULT_IMAGE_TOKEN + '\n' + qs[i]
            
            prompts = []

            for i in range(len(qs)):

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs[i])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                prompts.append(prompt)

            input_ids = tokenizer_image_token(prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

            images = []

            for image_file in image_files:
                with Image.open(image_file) as img:
                    image = img.convert('RGB')
                    image.append(images)
            
            image_tensor = process_images(images, image_processor, model.config)

            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    image_sizes=[image.size],
                    do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                    use_cache=True)
            
            g = output['scores'][0]

            preds = []

            for i in g:
                pred_options_logits = torch.stack([i[tokenizer.convert_tokens_to_ids(y_label)] for y_label in output_labels])
                pred = pred_options_logits.argmax(dim=-1).item()

                preds.append(pred)
            
            for cur_prompt, pred, line in zip(batch, cur_prompts, preds):
                model_outputs.append(line)
        
                line["prompt"] = cur_prompt

                line["model_id"] = model_name

                line["output"] = pred

                model_outputs.append(line)
        
        model_name_clean = args.model_path.replace("/", "-")
        
        output_file_name = os.path.basename(question_file).split(".")[0] + f"{model_name_clean}_answers.json"

        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
            json.dump(model_outputs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
