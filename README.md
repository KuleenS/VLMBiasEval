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

## Library
Our library unbiassae can be used for development and experiments. We have 3 folders

### eval
Output dataset evaluation

### debias
SAE tools to debias

### dataset
Dataset generators

## Scripts

We have two scripts

### VQA Eval
Evaluates CLIP and LLaVa models

```
python -m experiments.vqa_eval
```
Parameters
- datasets: list of datasets to use from ["celeba", "pata", "utkface", "visogender", "vlstereo", "adv_visogender"]

- include_image: include the image in the evaluation

- batch_size: batch size to evaluate at

- model_name: huggingface tag or path to model

- output_dir: output dir for ndjson of outputs

Full example config in `config224.toml`

### Debias Eval
Evaluates Paligemma with SAE based debiasing

```
python -m experiments.debias_eval
```

Parameters
- datasets: list of datasets to use from ["celeba", "pata", "utkface", "visogender", "vlstereo", "adv_visogender"]

- include_image: include the image in the evaluation

- batch_size: batch size to evaluate at

- model_name: huggingface tag or path to model

- output_dir: output dir for ndjson of outputs

- interventions: list of sae based interventions to try ["constant_sae", "conditional_per_input","conditional_per_token","clamping","conditional_clamping"]

- scaling_factors: list of how to scale these interventions [-40, -20, -10, -5 , 0, 5, 10, 20, 40]

- sae_layers: list of layers to target

- sae_releases: list of sae releases to target e.g. gemma-scope-2b-pt-mlp-canonical

- sae_ids: list of sae ids to use e.g layer_0/width_65k/canonical

- feature_idxs: features to steer on 

Full example config in `debiasconfig224.toml`

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
