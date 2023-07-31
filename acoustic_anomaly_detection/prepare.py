import os
import yaml
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram, AmplitudeToDB
from transformers import AutoProcessor

params = yaml.safe_load(open("params.yaml"))


class ASTProcessor(torch.nn.Module):
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


class Preparator:
    """
    Class to prepare data for training.

    Possible transforms:
    - MelSpectrogram
    - MFCC
    - Spectrogram
    - Huggingface ASTFeatureExtractor
    """

    def __init__(self):
        self.transform_func = {
            "mel_spectrogram": MelSpectrogram,
            "mfcc": MFCC,
            "spectrogram": Spectrogram,
            "ast": ASTProcessor,
        }[params["transform"]["type"]]
        transform_params = {
            k: v
            for k, v in params["transform"]["params"].items()
            if k in self.transform_func.__init__.__code__.co_varnames
        }
        # if params["transform"]["type"] != "ast":
        self.transform_func = self.transform_func(**transform_params)

        self.data_sources = params["data"]["data_sources"]
        self.segment = params["transform"]["segment"]

        self.sr = params["transform"]["params"]["sr"]
        self.duration = params["transform"]["params"]["duration"]

        self.length = self.sr * self.duration

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

    def slide_window(self, signal: torch.Tensor) -> torch.Tensor:
        window_size = 5
        stride = 1
        num_windows = (signal.shape[2] - window_size) // stride + 1
        windows = []
        for i in range(num_windows):
            window = signal[:, :, i * stride : i * stride + window_size].squeeze(0)
            windows.append(window)
        return torch.stack(windows)

    def transform(self, file_path: str, prepared_file_path: str) -> None:
        signal, sr = torchaudio.load(file_path)
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        if self.segment:
            pass
            # signal = self.segment(signal)
        else:
            signal = self.cut(signal)
        signal = self.transform_func(signal)
        if params["transform"]["type"] != "ast":
            signal = AmplitudeToDB(stype="power")(signal)
        signal = self.slide_window(signal)
        torch.save(signal, prepared_file_path)

    def __call__(self):
        for data_source in self.data_sources:
            raw_dir = os.path.join("data", "raw", data_source)
            for root, _, files in os.walk(raw_dir):
                for file in files:
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)
                        relative_file_path = os.path.relpath(file_path, raw_dir)
                        prepared_file_path = os.path.join(
                            "data", "prepared", "dcase2023t2", relative_file_path
                        )
                        prepared_file_path = prepared_file_path.replace(".wav", ".pt")
                        prepared_dir = os.path.dirname(prepared_file_path)
                        os.makedirs(prepared_dir, exist_ok=True)
                        self.transform(file_path, prepared_file_path)


if __name__ == "__main__":
    Preparator()()
