import torch
import torch.nn as nn
import torchaudio
import numpy as np
import yaml

params = yaml.safe_load(open("params.yaml"))["features"]

class Spectrogram(nn.Module):
    def __init__(self):
        super(Spectrogram, self).__init__()
        self.n_fft = params["n_fft"]
        self.win_length = params["win_length"]
        self.hop_length = params["hop_length"]
        self.power = params["power"]

    def forward(self, x):
        x = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
        )(x)
        return x
