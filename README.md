# SafetyTunedLLaVa

## Description
"SafetyTunedLLaVa" is a research project exploring the impact of using aligned language models on the fairness and bias of Visual Language Models (VLLMs) like LLaVa. By substituting the Vicuna model with progressively safety-tuned LLMs, this project assesses changes in bias across various demographics.

## Installation
Clone the repository to your local machine:
```
git clone https://github.com/your-repository/safety-tuned-llava.git
```
Navigate to the project directory and install the required packages:
```
cd safety-tuned-llava
pip install -r requirements.txt
```

## Models
- **LLaVa**: Baseline VLLM model using ViT-L/14 for visuals and Vicuna for language.
- **Safety Tuned LLaMas**: Models progressively tuned for safety. [More about Safety Tuned LLaMas](https://github.com/vinid/safety-tuned-llamas).

## Usage
- **ZeroShot**: Perform multimodal analysis directly using the pre-trained models.
- **Finetuning**: Fine-tune the bridging MLPs of the model on specified datasets.
- **Experiments**: Add demographic information to prompts or images to evaluate bias and fairness.

### Main Datasets
1. **VisoGender**
   - **Task**: Pronoun Resolution Caption Task. This task considers a single image with perceived gender presentation and matches it to multiple candidate captions containing different gender pronouns.
   - **Original Article**: [VisoGender](https://arxiv.org/abs/2306.12424)

2. **VLStereoSet**
   - **Task**: Caption Selection Task. Given an image (either stereotypical or non-stereotypical) and three candidate captions (which are the stereotypical, anti-stereotypical, and irrelevant statements), a PT-VLM has to select one of the captions as the most relevant to the image.
   - **Original Article**: [VLStereoSet](https://aclanthology.org/2022.aacl-main.40.pdf)

3. **Chest X-Rays Datasets**
   - **Included Datasets**: MIMIC-CXR, CheXpert, NIH, PadChest, VinDr.
   - **Task**: Zero-Shot Diagnosis. Given a chest X-ray and a prompt, predict findings. Protected attributes may include age, demographic, income, etc.
   - **Original Article**: [Chest X-Rays](https://arxiv.org/ftp/arxiv/papers/2402/2402.14815.pdf)

4. **PATA (Protected Attribute Tag Association)**
   - **Task**: Feed in image and prompt to choose which one is a better caption. Each image has a positive and a negative caption, where the negative captions consist of untoward and offensive text in the context of each protected attribute.
   - **Original Article**: [PATA](https://arxiv.org/pdf/2303.10431.pdf)

### Face Datasets
5. **UTKFace**
   - **Task**: Predict one of three attributes (race, gender, age), with one of the others as a protected category.
   - **Original Article**: [UTKFace](https://arxiv.org/pdf/1702.08423.pdf)

6. **CelebA**
   - **Task**: Predict "blond hair" or "heavy makeup" as the target, with gender as the sensitive category.
   - **Original Article**: [CelebA](https://arxiv.org/pdf/1411.7766.pdf), [Gender Parity in CelebA](https://arxiv.org/pdf/2206.10843.pdf)

7. **FairFace**
   - **Task**: Fair and balanced facial dataset for various demographics.
   - **Original Article**: [FairFace](https://arxiv.org/pdf/1908.04913.pdf)

8. **Chicago Face Database**
   - **Task**: Various tasks involving demographic studies with facial images.
   - **Original Article**: [Chicago Face Database](https://www.wittenbrink.org/cfd/mcw2015.pdf)

### Other Possible Datasets
9. **MSCOCO-Bias**
   - **Task**: Quantifying societal bias amplification in image captioning.
   - **Original Article**: [MSCOCO-Bias](https://openaccess.thecvf.com/content/CVPR2022/papers/Hirota_Quantifying_Societal_Bias_Amplification_in_Image_Captioning_CVPR_2022_paper.pdf)

## Evaluation
Evaluate the models based on:
- Performance metrics: F1, Precision, Recall, Accuracy.
- Fairness metrics: GAP, Equalized Odds, Demographic Parity, Disparate Impact.

## Contributing
Contributions to "SafetyTunedLLaVa" are welcome. Please submit a pull request or open an issue to discuss potential changes or additions.

## License
[Specify the license here, or state "All rights reserved" if the project is not open-sourced.]

## Related Works
- [Related work 1](https://arxiv.org/pdf/2402.02207.pdf)
- [Related work 2](https://arxiv.org/pdf/2303.10431.pdf)
- [Related work 3](https://arxiv.org/pdf/2303.12734.pdf)
