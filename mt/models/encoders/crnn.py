import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram
from einops import rearrange
import numpy as np


class CRNN(nn.Module):
    def __init__(self, in_dim: int, dim: int, out_dim: int, n_layers: int, **kwargs):
        super(CRNN, self).__init__()
        self.conv1 = ConvBlock(in_dim, 48)
        self.conv2 = ConvBlock(48, 64)
        self.conv3 = ConvBlock(64, 96)
        self.conv4 = ConvBlock(96, 128)

        self.gru = nn.GRU(
            input_size=128 * 8, 
            hidden_size=dim // 2, 
            num_layers=n_layers, 
            bias=True, 
            batch_first=True, 
            dropout=0., 
            bidirectional=True
        )
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        b: batch_size
        c: audio_channels
        t: n_frames
        f: n_freqs
        d: dim

        Args:
          in: (b, c, t, f)

        Outputs:
          out: (b, t, d)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = rearrange(x, 'b c t f -> b t (c f)')
        x, _ = self.gru(x)  # (b, t, d)
        x = self.proj(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=out_dim, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_dim, 
            out_channels=out_dim, 
            kernel_size=(3, 3), 
            padding=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        b: batch_size
        c: audio_channels
        t: n_frames
        f: n_freqs

        Args:
            x: (b, c, t, f)

        Returns:
            output: (b, d, t/2, f/2)
        """
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x))) 
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        return x