import torchaudio
import torch.nn as nn
from torch import Tensor
import torch
from einops import rearrange


class LogMel(nn.Module):
    def __init__(self, sr: int, n_fft: int, hop_length: int, n_mels: int, **kwargs):
        super().__init__()
        self.melsp = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft, 
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        self.eps = 1e-8

    def forward(self, x: Tensor) -> Tensor:
        r"""
        b: batch_size
        c: n_channels
        l: n_samples
        t: n_frames
        f: n_freq_bins

        Args:
            x: (b, c, l)

        Returns:
            out: (b, c, t, f)
        """
        x = self.melsp(x)
        x = torch.log10(torch.clamp(x, min=self.eps))
        x = rearrange(x, 'b c f t -> b c t f')
        return x