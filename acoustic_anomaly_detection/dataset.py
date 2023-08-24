import os
import random
import yaml
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram, AmplitudeToDB
from transformers import AutoProcessor

from acoustic_anomaly_detection.utils import get_attributes

params = yaml.safe_load(open("params.yaml"))


class ASTProcessor(torch.nn.Module):
    """
    Audio Spectrogram Transformer AutoEncoder
    Source: https://huggingface.co/transformers/model_doc/audio-spectrogram-transformer.html
    """

    def __init__(self):
        super().__init__()
        self.ast = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ast(
            x.squeeze(0),
            sampling_rate=params["transform"]["params"]["sr"],
            return_tensors="pt",
        )
        return x["input_values"]


class AudioDataset(Dataset):
    def __init__(
        self,
        file_list: list,
        fast_dev_run: bool = False,
    ) -> None:
        self.file_list = file_list
        self.seed = params["train"]["seed"]
        self.data_sources = params["data"]["data_sources"]
        self.transform_type = params["transform"]["type"]
        self.segment = params["transform"]["segment"]
        self.sr = params["transform"]["params"]["sr"]
        self.duration = params["transform"]["params"]["duration"]
        self.length = self.sr * self.duration

        self.transform_func = {
            "mel_spectrogram": MelSpectrogram,
            "mfcc": MFCC,
            "spectrogram": Spectrogram,
            "ast": ASTProcessor,
        }[self.transform_type]
        transform_params = {
            k: v
            for k, v in params["transform"]["params"].items()
            if k in self.transform_func.__init__.__code__.co_varnames
        }
        self.transform_func = self.transform_func(**transform_params)

        if fast_dev_run:
            random.seed(self.seed)
            self.file_list = random.sample(self.file_list, 100)

    def __len__(self) -> int:
        return len(self.file_list)

    def cut(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[1] > self.length:
            signal = signal[:, : self.length]
        elif signal.shape[1] < self.length:
            signal = torch.nn.functional.pad(signal, (0, self.length - signal.shape[1]))
        return signal

    def resample(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def transform(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        if self.segment:
            pass
            # signal = self.segment(signal)
        else:
            signal = self.cut(signal)
        signal = self.transform_func(signal)
        signal = signal.squeeze(0)
        if self.transform_type == "ast":
            return signal
        signal = AmplitudeToDB(stype="power")(signal)
        signal = signal.transpose(0, 1)
        return signal

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, str]]:
        file_path = self.file_list[idx]
        signal, sr = torchaudio.load(file_path)
        attributes = get_attributes(os.path.basename(file_path))
        return self.transform(signal, sr), attributes
