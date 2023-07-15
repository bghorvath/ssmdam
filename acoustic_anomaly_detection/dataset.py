import os
import random
import yaml
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram, AmplitudeToDB

params = yaml.safe_load(open("params.yaml"))


class AudioDataset(Dataset):
    def __init__(
        self,
        file_list: list,
        fast_dev_run: bool = False,
    ) -> None:
        self.file_list = file_list
        self.sr = params["transform"]["params"]["sr"]
        self.duration = params["transform"]["params"]["duration"]
        self.seed = params["train"]["seed"]

        if fast_dev_run:
            random.seed(self.seed)
            self.file_list = random.sample(self.file_list, 100)

        self.transform_func = {
            "mel_spectrogram": MelSpectrogram,
            "mfcc": MFCC,
            "spectrogram": Spectrogram,
        }[params["transform"]["type"]]
        transform_params = {
            k: v
            for k, v in params["transform"]["params"].items()
            if k in self.transform_func.__init__.__code__.co_varnames
        }
        self.transform_func = self.transform_func(**transform_params)

        # self.data, self.labels = self.create_data()

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

    def slide_window(self, signal: torch.Tensor) -> torch.Tensor:
        window_size = 5
        stride = 1
        num_windows = (signal.shape[2] - window_size) // stride + 1
        windows = []
        for i in range(num_windows):
            window = signal[:, :, i * stride : i * stride + window_size].squeeze(0)
            windows.append(window)
        return torch.stack(windows)

    def transform(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self.transform_func(signal)
        signal = AmplitudeToDB(stype="power")(signal)
        signal = self.slide_window(signal)
        return signal

    # def create_data(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     data = torch.tensor([])
    #     labels = torch.tensor([])

    #     for file_path in self.file_list:
    #         signal, sr = torchaudio.load(file_path) # type: ignore
    #         signal = self.transform(signal, sr)
    #         label = int(file_path.split("/")[-2])
    #         data = torch.cat([data, signal])
    #         labels = torch.cat([labels, torch.tensor([label])])
    #     return data, labels

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.file_list[idx]
        label = os.path.basename(p=audio_path).split("_")[0]
        label = torch.tensor(1) if label == "anomaly" else torch.tensor(0)
        signal, sr = torchaudio.load(audio_path)  # type: ignore
        signal = self.transform(signal, sr)
        return signal, label
