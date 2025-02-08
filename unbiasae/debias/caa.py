from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Dict, Tuple, Optional, Any, List
import json
import torch
import einops
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import unbiasae.debias.utils as utils
from unbiasae.debias.eval_config import EvalConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from PIL.Image import Image


def format_contrastive_prompt(
    contrastive_prompts: Dict[str, str],
    example: str,
    prompt_polarity: str,
    processor,
) -> str:
    """Format a contrastive prompt with code for model input.

    Args:
        contrastive_prompts: Dictionary of prompt templates
        code_block: Code snippet to insert into prompt
        prompt_type: Type of prompt to use (must exist in contrastive_prompts)
        prompt_polarity: Either "pos" or "neg" for positive/negative prompt
        tokenizer: Tokenizer for formatting chat template

    Returns:
        Formatted prompt string ready for model input

    """

    prompt = f"{contrastive_prompts[prompt_polarity]}\n{example}"

    chat = [{"role": "user", "content": prompt}]
    # Handle different tokenizer types
    if hasattr(processor, "apply_chat_template"):
        result = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return result if isinstance(result, str) else result[0]

    return f"{prompt}\n"


def get_layer_activations(
    model: AutoModelForCausalLM, target_layer: int, inputs: torch.Tensor
) -> torch.Tensor:
    """Extract activations from a specific transformer layer.

    Args:
        model: The causal language model
        target_layer: Index of layer to extract from
        inputs: Input token ids tensor of shape (batch_size, seq_len)

    Returns:
        Tensor of activations with shape (batch_size, seq_len, hidden_dim)

    Raises:
        AttributeError: If model architecture is not supported
        RuntimeError: If no activations were captured
    """
    acts_BLD: Optional[torch.Tensor] = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal acts_BLD
        acts_BLD = outputs[0]
        return outputs

    # Support different model architectures
    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        raise AttributeError("Model architecture not supported")

    handle = layers[target_layer].register_forward_hook(gather_target_act_hook)
    with torch.no_grad():
        _ = model(**inputs.to(model.device))
    handle.remove()

    if acts_BLD is None:
        raise RuntimeError("No activations were captured")
    return acts_BLD


def get_layer_activations_with_generation(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get layer activations during text generation.

    Args:
        model: The causal language model
        target_layer: Index of layer to extract from
        inputs: Input token ids tensor
        **generation_kwargs: Arguments passed to model.generate()

    Returns:
        Tuple containing:
        - Tensor of activations during generation
        - Generated token ids tensor
    """
    acts_BLD: List[torch.Tensor] = []

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal acts_BLD
        acts_BLD.append(outputs[0])
        return outputs

    if hasattr(model, "transformer"):
        layers = model.transformer.h
    elif hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.model.layers
    else:
        raise AttributeError("Model architecture not supported")

    handle = layers[target_layer].register_forward_hook(gather_target_act_hook)
    with torch.no_grad():
        tokens = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    handle.remove()

    return torch.cat(acts_BLD, dim=1), tokens


@torch.no_grad()
def calculate_probe_vector(
    prompts_dict: Dict[str, Dict[str, str]],
    examples: Tuple[List[str], List[Image]],
    model,
    processor,
    prompt_type: str,
    layer: int,
    n_samples: int = 50,
    max_new_tokens: int = 1,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the steering vector for a given layer.
    Follows this general methodology: https://arxiv.org/abs/2410.12877"""
    train_pos_activations = []
    train_neg_activations = []
    test_pos_activations = []
    test_neg_activations = []
    n_train = int(n_samples * 0.8)

    for example, image in tqdm(examples):
        pos_prompt = format_contrastive_prompt(
            prompts_dict, prompts_dict, prompt_type, "pos", processor
        )
        neg_prompt = format_contrastive_prompt(
            prompts_dict, prompts_dict, prompt_type, "neg", processor
        )
        pos_tokens = processor(
            [pos_prompt] * n_samples, images = [image]*n_samples, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        neg_tokens = processor(
            [neg_prompt] * n_samples, images = [image]*n_samples, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)

        pos_activations, new_pos_tokens = get_layer_activations_with_generation(
            model, layer, pos_tokens, max_new_tokens=max_new_tokens, temperature=temperature
        )
        neg_activations, new_neg_tokens = get_layer_activations_with_generation(
            model, layer, neg_tokens, max_new_tokens=max_new_tokens, temperature=temperature
        )

        # Get generated tokens only. : -1 because we don't have activations for the final generated token
        new_pos_tokens = new_pos_tokens[:, pos_tokens.shape[1] : -1]
        new_neg_tokens = new_neg_tokens[:, neg_tokens.shape[1] : -1]

        # Get activations for generated tokens only
        pos_activations = pos_activations[:, pos_tokens.shape[1] :]
        neg_activations = neg_activations[:, neg_tokens.shape[1] :]
        # Split activations into train and test based on n_samples first
        train_pos_activations_batch = pos_activations[:n_train]
        test_pos_activations_batch = pos_activations[n_train:]
        train_neg_activations_batch = neg_activations[:n_train]
        test_neg_activations_batch = neg_activations[n_train:]

        # Create masks for non-padding and non-eos tokens
        pos_mask = (new_pos_tokens != processor.tokenizer.pad_token_id) & (
            new_pos_tokens != processor.tokenizer.eos_token_id
        )
        neg_mask = (new_neg_tokens != processor.tokenizer.pad_token_id) & (
            new_neg_tokens != processor.tokenizer.eos_token_id
        )

        # Split masks into train and test
        train_pos_mask = pos_mask[:n_train]
        test_pos_mask = pos_mask[n_train:]
        train_neg_mask = neg_mask[:n_train]
        test_neg_mask = neg_mask[n_train:]

        # Reshape activations and masks to match
        train_pos_activations_batch = einops.rearrange(
            train_pos_activations_batch, "B L D -> (B L) D"
        )
        test_pos_activations_batch = einops.rearrange(
            test_pos_activations_batch, "B L D -> (B L) D"
        )
        train_neg_activations_batch = einops.rearrange(
            train_neg_activations_batch, "B L D -> (B L) D"
        )
        test_neg_activations_batch = einops.rearrange(
            test_neg_activations_batch, "B L D -> (B L) D"
        )

        train_pos_mask = einops.rearrange(train_pos_mask, "B L -> (B L)")
        test_pos_mask = einops.rearrange(test_pos_mask, "B L -> (B L)")
        train_neg_mask = einops.rearrange(train_neg_mask, "B L -> (B L)")
        test_neg_mask = einops.rearrange(test_neg_mask, "B L -> (B L)")

        # Filter out padding and eos tokens
        train_pos_activations.append(train_pos_activations_batch[train_pos_mask])
        test_pos_activations.append(test_pos_activations_batch[test_pos_mask])
        train_neg_activations.append(train_neg_activations_batch[train_neg_mask])
        test_neg_activations.append(test_neg_activations_batch[test_neg_mask])

    # Combine all activations
    X_train = (
        torch.cat(
            [torch.cat(train_pos_activations, dim=0), torch.cat(train_neg_activations, dim=0)],
            dim=0,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    y_train = (
        torch.cat(
            [
                torch.ones(sum(len(x) for x in train_pos_activations)),
                torch.zeros(sum(len(x) for x in train_neg_activations)),
            ]
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    X_test = (
        torch.cat(
            [torch.cat(test_pos_activations, dim=0), torch.cat(test_neg_activations, dim=0)], dim=0
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    y_test = (
        torch.cat(
            [
                torch.ones(sum(len(x) for x in test_pos_activations)),
                torch.zeros(sum(len(x) for x in test_neg_activations)),
            ]
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    # Fit model and get probe vector
    linreg_model = LogisticRegression().fit(X_train, y_train)
    probe_vector = torch.tensor(linreg_model.coef_[0]).to(device=model.device, dtype=model.dtype)
    probe_bias = torch.tensor(linreg_model.intercept_[0]).to(device=model.device, dtype=model.dtype)

    # Report test accuracy
    test_acc = linreg_model.score(X_test, y_test)
    print(f"Test accuracy on {len(X_test)} points: {test_acc:.3f}")
    train_acc = linreg_model.score(X_train, y_train)
    print(f"Train accuracy on {len(X_train)} points: {train_acc:.3f}")

    return probe_vector, probe_bias


@torch.no_grad()
def calculate_steering_vector(
    prompts_dict: Dict[str, Dict[str, str]],
    examples: Tuple[List[str], List[Image]],
    model,
    processor,
    layer: int,
) -> torch.Tensor:
    """Calculate the steering vector for a given layer.
    Follows this general methodology: https://arxiv.org/abs/2410.12877"""
    all_pos_activations = []
    all_neg_activations = []

    for example, image in examples:
        pos_prompt = format_contrastive_prompt(
            prompts_dict, example, "pos", processor
        )
        neg_prompt = format_contrastive_prompt(
            prompts_dict, example, "neg", processor
        )
        pos_tokens = processor(pos_prompt, images=[image], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device)
        neg_tokens = processor(neg_prompt, images=[image], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device)

        pos_activations = get_layer_activations(model, layer, pos_tokens)
        neg_activations = get_layer_activations(model, layer, neg_tokens)
        pos_activations = get_layer_activations(model, layer, pos_tokens)[:, -1, :].squeeze()
        neg_activations = get_layer_activations(model, layer, neg_tokens)[:, -1, :].squeeze()

        all_pos_activations.append(pos_activations)
        all_neg_activations.append(neg_activations)

    pos_activations = torch.stack(all_pos_activations).mean(dim=0)
    neg_activations = torch.stack(all_neg_activations).mean(dim=0)

    return pos_activations - neg_activations


def get_threshold(
    examples: Tuple[List[str], List[Image]],
    model_params: dict,
    wrapper,
    steering_vector_D: torch.Tensor,
    encoder_vector_D: torch.Tensor,
    encoder_threshold: float,
) -> float:

    average_threshold = 0

    for example, image in examples:

        chat = [{"role": "user", "content": example}]

        if hasattr(wrapper.processor, "apply_chat_template"):
            result = wrapper.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            example = result if isinstance(result, str) else result[0]
        
        tokens = wrapper.processor(example, images = [image], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(wrapper.model.device)

        resid_BLD = get_layer_activations(wrapper.model, model_params["targ_layer"], tokens)

        feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)
        above_threshold = feature_acts_BL > encoder_threshold
        k = above_threshold.sum().item()

        steering_vector_acts_BL = torch.einsum("BLD, D->BL", resid_BLD, steering_vector_D)

        topk_values = torch.topk(steering_vector_acts_BL.flatten(), k, largest=True)[0]
        threshold = topk_values[-1].item()
        average_threshold += threshold

    average_threshold /= len(examples)

    return average_threshold
