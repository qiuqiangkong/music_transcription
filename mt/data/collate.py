from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


def collate_fn(batch: list[dict]) -> dict:
    """Collate samples into a batch, padding variable-length NumPy arrays."""
    out = {}

    for key in batch[0]:
        values = [item[key] for item in batch]
        x = values[0]

        if isinstance(x, str):
            out[key] = values

        elif isinstance(x, list):
            out[key] = values

        elif isinstance(x, Path):
            out[key] = values

        elif key in ["token", "token_mask", "tokens_num"]:
            out[key] = torch.from_numpy(pad_sequence(values, dim=-1))

        elif x is None:
            out[key] = None

        else:
            out[key] = torch.utils.data.default_collate(values)
            
    return out


def pad_sequence(xs, dim=0):
    max_len = max(x.shape[dim] for x in xs)
    outs = []
    for x in xs:
        pad_width = [(0, 0)] * x.ndim
        pad_width[dim] = (0, max_len - x.shape[dim])
        outs.append(np.pad(x, pad_width))

    return np.stack(outs)