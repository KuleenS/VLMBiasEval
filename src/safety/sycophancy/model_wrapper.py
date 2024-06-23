import torch

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:]
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None

class LLaVaModelWrapper:
    def __init__(self, model, processor):
        self.processor = processor
        self.model = model
        for i, layer in enumerate(self.model.language_model.model.layers):
            self.model.language_model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, inputs, **kwargs):
        generate_ids = self.model.generate(**inputs, **kwargs)
        return self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def get_logits(self, inputs):
        with torch.no_grad():
            logits = self.model(**inputs).logits
            return logits
    
    def get_last_activations(self, layer):
        return self.model.language_model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.language_model.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.language_model.model.layers:
            layer.reset()