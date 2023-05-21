import torch
import torch.nn as nn
import torchaudio
import numpy as np

class Spectrogram(nn.Module):
    def __init__(self, n_fft=400, win_length=400, hop_length=160, power=2.0):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.power = power

    def forward(self, x):
        x = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
        )(x)
        return x
