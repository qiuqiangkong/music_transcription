from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from .attention import Block
from ..layers.rope import build_rope


class Transformer(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        in_dim: int,
        dim: int,
        mlp_ratio=4.0,
        n_layers=12,
        n_heads=12,
        rope_len=8192,
        **kwargs
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = dim // n_heads

        self.in_proj = nn.Linear(in_dim, dim)
        self.embedder = nn.Embedding(n_vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads) for _ in range(n_layers))
        self.out_proj = nn.Linear(dim, n_vocab)
        
    
    def forward(
        self, 
        token: LongTensor, 
        seq: Tensor,
        self_attn_mask=None,
        cross_attn_mask=None,
    ) -> Tensor:
        r"""DiT.

        b: batch_size
        d: dim
        
        Args:
            x: (b, l_q)

        Outputs:
            out: (b, l_q, n_vocab)
        """

        device = seq.device
        rope_q = build_rope(head_dim=self.head_dim).to(device)
        rope_k = build_rope(head_dim=self.head_dim).to(device)

        x = self.embedder(token)
        seq = self.in_proj(seq)

        for block in self.blocks:
            x = block(x, seq, rope_q, rope_k, self_attn_mask, cross_attn_mask)

        out = self.out_proj(x)
        return out
