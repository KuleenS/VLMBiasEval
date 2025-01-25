import argparse
import os
import json
from tqdm import tqdm
import base64
import io

import PIL

from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def batch_iterable(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def extract_test_name(question_file):
    start = "zeroshot_test_"
    end = ".json"
    start_index = question_file.find(start) + len(start)
    end_index = question_file.find(end)
    return question_file[start_index:end_index]


def image_to_base64(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate(prompt, image, model, temp = 0, top_p = None):
    image = image_to_base64(image)
    try:
        vertexai.init(project="api2024-421420", location="us-central1")
        model = GenerativeModel(
            model,
        )
        responses = model.generate_content(
            [prompt, image],
            generation_config={
                "max_output_tokens": 1,
                "temperature": temp,
                "top_p": top_p,
            }
        )

        # Check if the content is blocked
        if responses.candidates[0].finish_reason == FinishReason.SAFETY:
            print("Content blocked by safety filters")
            return 'Content blocked by safety filters'

        return responses.text
    except Exception as e:
        print(f"Error in generate function: {e}")
        # Handle other unexpected errors
        return 'Error'
    

def xgenerate(prompt, model, temp = 0, top_p = None):
    try:
        vertexai.init(project="api2024-421420", location="us-central1")
        model = GenerativeModel(
            model,
        )
        safe = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}
        responses = model.generate_content(
            [prompt],
            generation_config={
                "max_output_tokens": 1,
                "temperature": temp,
                "top_p": top_p,
            },
            safety_settings=safe
        )

        # Check if the content is blocked
        if responses.candidates[0].finish_reason == FinishReason.SAFETY:
            print("Content blocked by safety filters")
            return 'Content blocked by safety filters'

        return responses.text
    except Exception as e:
        print(f"Error in generate function: {e}")
        # Handle other unexpected errors
        return 'Error'


def eval_model(args):
    try:
        
        model_name = args.model_path
        temp = args.temperature
        top_p = args.top_p
        
        question_files = [os.path.join(args.question_folder, x) for x in os.listdir(args.question_folder)]

        for question_file in tqdm(question_files):

            model_name_clean = args.model_path.replace("/", "-")
            test_name = extract_test_name(question_file)
            output_file_name = f"{model_name_clean}_{test_name}_answers.json"

            # if os.path.exists(answer_file_path):
            #     print(f'{answer_file_path} already exists.')
            #     continue

            with open(question_file, "r") as f:
                data = json.loads(f.read())

            questions = data["data"]
            index_to_start = int(data["to_start"])

            if index_to_start > 0:
                if index_to_start >= len(questions):
                    print(f'Skipping: {question_file} already completed.')
                    continue
                else:
                    answer_file_path = os.path.join(args.output_folder, f"{model_name_clean}_{test_name}_answers.json")
                    with open(answer_file_path, 'r') as file:
                        old_data = json.load(file)
                    model_outputs = old_data[:index_to_start]
            
            else:
                model_outputs = []

            questions_batched = batch_iterable(questions[index_to_start:], args.batch_size)
            for batch in tqdm(questions_batched):
                for q in batch:
                        if args.include_image:
                            try:
                                prompt = q['prompt']
                                image_file = q['image']
                                keys = list(q.keys())
                                special_topic = keys[-1]
                                try:
                                    image = Image.open(image_file)
                                    image.verify()  # Verify that the image is not corrupted
                                    image = Image.open(image_file)  # Reopen image after verification
                                except (PIL.UnidentifiedImageError, IOError) as e:
                                    print(f"Cannot identify image file {image_file}: {e}")
                                    continue

                                prompt += " Answer with just the letter that corresponds to the correct answer."

                                response_text = generate(prompt, image, model_name, temp=temp, top_p=top_p)
                            except FileNotFoundError:
                                print(f"Image file {image_file} not found")
                                continue
                            except PIL.UnidentifiedImageError:
                                print(f"Cannot identify image file {image_file}")
                                continue
            

                        else:
                            prompt = q['prompt'].replace('the correct caption of this image?', 'most likely the correct sentence?')
                            image_file = q['image']
                            keys = list(q.keys())
                            special_topic = keys[-1]

                            prompt += " Answer with just the letter that corresponds to the correct answer."

                            response_text = xgenerate(prompt, model_name, temp=temp, top_p=top_p)
                        
                        model_output = {
                            "prompt": prompt,
                            "image": image_file,  # Store image file path instead of image object
                            "label": q['label'],
                            special_topic: q[special_topic],
                            "model_id": model_name,
                            "output": response_text[0]
                        }

                        model_outputs.append(model_output)

                        index_to_start+=1
                        if index_to_start % 20 == 0:
                            with open(question_file, 'r') as f:
                                test_data = json.load(f)
                            test_data["to_start"] = index_to_start
                            with open(question_file, 'w') as f:
                                json.dump(test_data, f)

                        with open(os.path.join(args.output_folder, output_file_name), "w") as f:
                            json.dump(model_outputs, f)


    except Exception as e:
        print(f"Error in eval_model function: {e}")
        return 'Error'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="gemini-1.5-flash-001")
    parser.add_argument("--question_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--include_image", action="store_true", default=True)
    parser.add_argument("--exclude_image", action="store_false", dest="include_image")
    args = parser.parse_args()

    eval_model(args)