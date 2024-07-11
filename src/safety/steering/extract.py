import dataclasses
import os
import typing
import warnings

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm

import PIL 

from PIL import Image

from src.safety.steering.control import ControlModel, model_layer_list


@dataclasses.dataclass
class DatasetEntry:
    positive: str
    negative: str
    image: str
    

@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[int, np.ndarray]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel | ControlModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        bias: bool,
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                max_batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Can be either
                    "pca_diff" or "pca_center". Defaults to "pca_diff".

        Returns:
            ControlVector: The trained vector.
        """
        if bias:
            dirs = read_representations_bias(
                model,
                tokenizer,
                dataset,
                **kwargs,
            )
        else:
            dirs = read_representations(
                model,
                tokenizer,
                dataset,
                **kwargs,
            )
        return cls(model_type=model.config.model_type, directions=dirs)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        directions: dict[int, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.int_ | np.float_) -> "ControlVector":
        return self.__mul__(1 / other)

def read_representations_bias(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    protected_category_weights = dict[str, float],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_diff",
) -> dict[int, np.ndarray]:
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [(ex[0], ex[2]) for ex in inputs]

    protected_categories = [ex[1] for ex in inputs]

    set_protected_categories = set(protected_categories)

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        h = layer_hiddens[layer]

        vectors = dict()

        for protected_category in set_protected_categories:
            train = np.array([x for x,y in zip(h, protected_categories) if y == protected_category])

            if method != "umap":
                # shape (1, n_features)
                pca_model = PCA(n_components=1, whiten=False).fit(train)
                # shape (n_features,)
                vectors[protected_category] = pca_model.components_.astype(np.float32).squeeze(axis=0)
            else:
                # still experimental so don't want to add this as a real dependency yet
                import umap  # type: ignore

                umap_model = umap.UMAP(n_components=1)
                embedding = umap_model.fit_transform(train).astype(np.float32)
                vectors[protected_category] =  np.sum(train * embedding, axis=0) / np.sum(embedding)
        
        directions[layer] = sum([protected_category_weights[x]*vectors[x] for x in protected_category_weights.keys()])

    return directions
    

def read_representations(
    model: "PreTrainedModel | ControlModel",
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[DatasetEntry],
    hidden_layers: typing.Iterable[int] | None = None,
    batch_size: int = 32,
    method: typing.Literal["pca_diff", "pca_center", "umap"] = "pca_diff",
) -> dict[int, np.ndarray]:
    """
    Extract the representations based on the contrast dataset.
    """
    if not hidden_layers:
        hidden_layers = range(-1, -model.config.num_hidden_layers, -1)

    # normalize the layer indexes if they're negative
    n_layers = len(model_layer_list(model))
    hidden_layers = [i if i >= 0 else n_layers + i for i in hidden_layers]

    # the order is [positive, negative, positive, negative, ...]
    train_strs = [(s, ex[2]) for ex in inputs for s in (ex[0], ex[1])]

    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size
    )

    # get directions for each layer using PCA
    directions: dict[int, np.ndarray] = {}
    for layer in tqdm.tqdm(hidden_layers):
        h = layer_hiddens[layer]
        assert h.shape[0] == len(inputs) * 2

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = (h[::2] + h[1::2]) / 2
            train = h
            train[::2] -= center
            train[1::2] -= center
        elif method == "umap":
            train = h
        else:
            raise ValueError("unknown method " + method)

        if method != "umap":
            # shape (1, n_features)
            pca_model = PCA(n_components=1, whiten=False).fit(train)
            # shape (n_features,)
            directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        else:
            # still experimental so don't want to add this as a real dependency yet
            import umap  # type: ignore

            umap_model = umap.UMAP(n_components=1)
            embedding = umap_model.fit_transform(train).astype(np.float32)
            directions[layer] = np.sum(train * embedding, axis=0) / np.sum(embedding)

        # calculate sign
        projected_hiddens = project_onto_direction(h, directions[layer])

        # order is [positive, negative, positive, negative, ...]
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )

        if positive_smaller_mean > positive_larger_mean:  # type: ignore
            directions[layer] *= -1

    return directions


def batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    """
    Using the given model and tokenizer, pass the inputs through the model and get the hidden
    states for each layer in `hidden_layers` for the last token.

    Returns a dictionary from `hidden_layers` layer id to an numpy array of shape `(n_inputs, hidden_dim)`
    """
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layers}
    
    for batch in tqdm.tqdm(batched_inputs):

        qs = [x[0] for x in batch]

        image_files = [x[1] for x in batch]

        images = []

        prompts = []

        for image_file, q in zip(image_files, qs):
            try:
                if image_file:
                    images.append(Image.open(image_file))
                else:
                    images.append(None)

                prompts.append(f"[INST] <image>\n{q} [/INST]")

            except PIL.UnidentifiedImageError:
                continue
            
        if len(images) != 0 and len(prompts) != 0:

            if all([x is None for x in images]):
                inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            else:
                inputs = tokenizer(prompts, images=images, padding=True, return_tensors="pt")
            
            with torch.no_grad():
                out = model(
                    **inputs.to(model.device),
                    output_hidden_states=True,
                )
            
            for layer in hidden_layers:
                # if not indexing from end, account for embedding hiddens
                hidden_idx = layer + 1 if layer >= 0 else layer
                for batch in out.hidden_states[hidden_idx]:
                    hidden_states[layer].append(batch[-1, :].squeeze().cpu().numpy())
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = np.linalg.norm(direction)
    assert not np.isinf(mag)
    return (H @ direction) / mag