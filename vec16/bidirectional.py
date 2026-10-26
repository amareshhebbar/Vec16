import torch
from transformers import LlamaModel, Qwen2Model, Qwen3Model

try:
    from transformers import Gemma3TextModel
except ImportError:
    Gemma3TextModel = None


def bidirectional_mask(attention_mask, input_ids=None, inputs_embeds=None, dtype=torch.float32):
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    if attention_mask is None:
        ref = input_ids if input_ids is not None else inputs_embeds
        attention_mask = torch.ones(ref.shape[0], ref.shape[1], dtype=torch.long, device=ref.device)
    n = attention_mask.shape[1]
    keep = attention_mask[:, None, None, :].to(dtype)
    mask = (1.0 - keep) * torch.finfo(dtype).min
    return mask.expand(-1, 1, n, -1)


def disable_causal(model):
    model.config.is_causal = False
    for m in model.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False
    return model


class BidirectionalMixin:
    def __init__(self, config, *args, **kwargs):
        config.is_causal = False
        super().__init__(config, *args, **kwargs)
        disable_causal(self)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        kwargs.setdefault("use_cache", False)
        kwargs.pop("past_key_values", None)
        dtype = self.get_input_embeddings().weight.dtype
        if torch.is_autocast_enabled():
            dtype = torch.float32
        mask = bidirectional_mask(attention_mask, input_ids, inputs_embeds, dtype)
        return super().forward(input_ids=input_ids, attention_mask=mask, inputs_embeds=inputs_embeds, **kwargs)


class Vec16LlamaModel(BidirectionalMixin, LlamaModel):
    pass


class Vec16Qwen2Model(BidirectionalMixin, Qwen2Model):
    pass


class Vec16Qwen3Model(BidirectionalMixin, Qwen3Model):
    pass


CLASSES = {
    "LlamaModel": Vec16LlamaModel,
    "Qwen2Model": Vec16Qwen2Model,
    "Qwen3Model": Vec16Qwen3Model,
}

if Gemma3TextModel is not None:

    class Vec16Gemma3TextModel(BidirectionalMixin, Gemma3TextModel):
        pass

    CLASSES["Gemma3TextModel"] = Vec16Gemma3TextModel


def to_bidirectional(model):
    name = type(model).__name__
    if name in {c.__name__ for c in CLASSES.values()}:
        return disable_causal(model)
    if name not in CLASSES:
        raise ValueError(f"no bidirectional class for {name}; supported: {sorted(CLASSES)}")
    model.__class__ = CLASSES[name]
    return disable_causal(model)
