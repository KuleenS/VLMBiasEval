# VLMBiasEval

## Description
"VLMBiasEval" is a research project exploring the fairness and bias of Visual Language Models (VLLMs) like LLaVa and ways to debiasing them using SAE based methods

## Installation
Clone the repository to your local machine:
```
git clone https://github.com/KuleenS/VLMBiasEval.git
```
Navigate to the project directory and install the required packages:
```
cd VLMBiasEval
conda env create -f vlmbiaseval.yaml
```

## Scripts

To run any script 

```
python -m src.scripts.<script_name> <args>
```

### evaluate.py
Runs evaluation of output

### generate_datasets.py
Generates datasets from config.toml

### model_clip.py
Evaluate CLIP model

To use medclip you must install it from the pip package or the repo (https://github.com/RyanWangZf/MedCLIP)
However, it does have older versioning of some packages so please be mindful of conflicts that could occur

### model_vqa.py
Evaluate VQA model

### model_adversarial.py
Evaluate VQA model on adversarial VisoGender 

### model_gemini.py
Evaluate gemini

## Models
- **LLaVa**: Set of LLaVa 1.6 models from 7B to 34B from llava-hf
- **CLIP**: CLIP Large 224 and CLIP Large 336
- **Gemini**: Gemini Flash 001 
- **PaliGemma**

### General Datasets
1. **VisoGender**
   - **Task**: Pronoun Resolution Caption Task. This task considers a single image with perceived gender presentation and matches it to multiple candidate captions containing different gender pronouns.
   - **Original Article**: [VisoGender](https://arxiv.org/abs/2306.12424)

2. **VLStereoSet**
   - **Task**: Caption Selection Task. Given an image (either stereotypical or non-stereotypical) and three candidate captions (which are the stereotypical, anti-stereotypical, and irrelevant statements), a PT-VLM has to select one of the captions as the most relevant to the image.
   - **Original Article**: [VLStereoSet](https://aclanthology.org/2022.aacl-main.40.pdf)

3. **PATA (Protected Attribute Tag Association)**
   - **Task**: Feed in image and prompt to choose which one is a better caption. Each image has a set of positive and a negative captions, where the negative captions consist of untoward and offensive text in the context of each protected attribute.
   - **Original Article**: [PATA](https://arxiv.org/pdf/2303.10431.pdf)

### Face Datasets
4. **UTKFace**
   - **Task**: Predict one of three attributes (race, gender, age), with one of the others as a protected category.
   - **Original Article**: [UTKFace](https://arxiv.org/pdf/1702.08423.pdf)

5. **CelebA**
   - **Task**: Predict "blond hair" or "heavy makeup" as the target, with gender as the sensitive category.
   - **Original Article**: [CelebA](https://arxiv.org/pdf/1411.7766.pdf), [Gender Parity in CelebA](https://arxiv.org/pdf/2206.10843.pdf)

## Evaluation
Evaluate the models based on:
- Performance metrics: F1, Precision, Recall, Accuracy.
- Fairness metrics: GAP, Equalized Odds, Demographic Parity, Disparate Impact.

## Contributing
Contributions to "VLMBiasEval" are welcome. Please submit a pull request or open an issue to discuss potential changes or additions.
