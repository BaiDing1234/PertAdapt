import torch.nn as nn

from typing import Union


def init_linear_xavier_(linear: Union[nn.Linear, nn.Embedding, None]):
    if linear is None:
        return linear
    nn.init.xavier_uniform_(linear.weight, gain=1)
    if hasattr(linear, "bias") and linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def init_linear_zero_(linear: Union[nn.Linear, nn.Embedding, None], eps: float = 1e-6):
    if linear is None:
        return linear
    nn.init.normal_(linear.weight, 0, eps)
    if hasattr(linear, "bias") and linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def init_linear_(
    linear: Union[nn.Linear, nn.Embedding, None],
    init_type: str = "xavier",
    eps: float = 1e-6,
):
    if init_type == "xavier":
        return init_linear_xavier_(linear)
    elif init_type == "zero":
        return init_linear_zero_(linear, eps=eps)
    else:
        raise ValueError(f"Unknown init_type {init_type}")


def init_layer_norm_(layer_norm: Union[nn.LayerNorm, None]):
    if layer_norm is None:
        return layer_norm
    nn.init.ones_(layer_norm.weight)
    nn.init.zeros_(layer_norm.bias)
    return layer_norm


def init_params_recursively_(module: nn.Module):
    for name, child in module.named_children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
        elif isinstance(child, nn.Linear):
            init_linear_(child)
        elif isinstance(child, nn.LayerNorm):
            init_layer_norm_(child)
        elif isinstance(child, nn.Embedding):
            init_linear_(child)
        elif isinstance(child, nn.Dropout):
            pass
        else:
            try:
                init_params_recursively_(child)
            except Exception as e:
                print(f"Failed to init {name} with {e}")
