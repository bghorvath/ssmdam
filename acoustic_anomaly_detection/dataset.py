import os
import random
import yaml
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram

params = yaml.safe_load(open("params.yaml"))

class AudioDataset(Dataset):
    def __init__(self, test: bool = False, fast_dev_run: bool = False) -> None:
        self.audio_dirs = params["data"]["audio_dirs"]

        self.file_list = []
        for audio_dir in self.audio_dirs:
            audio_path = os.path.join(audio_dir, "test" if test else "train")
            file_list = [os.path.join(audio_path, file) for file in os.listdir(audio_path)]
            self.file_list += file_list

        if fast_dev_run:
            random.seed(params["train"]["seed"])
            self.file_list = random.sample(self.file_list, 1000)

        self.sr = params["transform"]["params"]["sr"]
        self.duration = params["transform"]["params"]["duration"]

        self.transform = {
            "mel_spectrogram": MelSpectrogram,
            "mfcc": MFCC,
            "spectrogram": Spectrogram,
        }[params["transform"]["type"]]

        transform_params = {k:v for k, v in params["transform"]["params"].items() if k in self.transform.__init__.__code__.co_varnames}

        self.transform = self.transform(**transform_params)

    def __len__(self) -> int:
        return len(self.file_list)

    def _cut(self, signal: torch.Tensor) -> torch.Tensor:
        length = self.sr * self.duration
        if signal.shape[1] > length:
            signal = signal[:, :length]
        elif signal.shape[1] < length:
            signal = torch.nn.functional.pad(signal, (0, length - signal.shape[1]))
        return signal

    def _resample(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def __getitem__(self, idx) -> tuple:
        audio_path = self.file_list[idx]
        label = os.path.basename(p=audio_path).split("_")[0]
        label = torch.tensor(1) if label == "anomaly" else torch.tensor(0)
        signal, sr = torchaudio.load(audio_path) # type: ignore
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self.transform(signal)
        return signal, label
