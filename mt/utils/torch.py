from torch import Tensor
import torch.nn as nn
import torch
from itertools import chain


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def to_device(data: dict, device) -> dict:
    for k, v in data.items():
        if isinstance(v, Tensor):
            data[k] = v.to(device)
    return data


# def mse(x, y):
#     return ((x - y) ** 2).mean()


def save_checkpoint(model: nn.Module, path) -> None:
    """Save model into a checkpoint."""
    torch.save(model.state_dict(), path)


def load(model: nn.Module, path: str) -> nn.Module:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model
