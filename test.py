import os
import yaml
import torch
from torch import nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchaudio
from transformers import AutoProcessor, ASTModel
from torchaudio.transforms import MelSpectrogram, MFCC

from acoustic_anomaly_detection.dataset import AudioDataset

params = yaml.safe_load(open("params.yaml"))

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchaudio
from acoustic_anomaly_detection.utils import get_attributes

class ASTProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ast = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            max_length=998
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
    ) -> None:
        self.file_list = file_list
        self.seed = params["train"]["seed"]
        self.data_sources = params["data"]["data_sources"]
        self.transform_type = params["transform"]["type"]
        self.segment = params["transform"]["segment"]
        self.sr = params["transform"]["params"]["sr"]
        self.duration = params["transform"]["params"]["duration"]
        self.length = self.sr * self.duration
        self.transform_func = ASTProcessor()

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
        signal = self.cut(signal)
        signal = self.transform_func(signal)
        return signal
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, str]]:
        file_path = self.file_list[idx]
        signal, sr = torchaudio.load(file_path)
        attributes = get_attributes(os.path.basename(file_path))
        return self.transform(signal, sr), attributes
    

audio_dir = os.path.join("data", "raw", "dcase2023t2", "dev", "bearing", "train")
file_list = [
    os.path.join(audio_dir, file)
    for file in os.listdir(audio_dir)
]

dataset = AudioDataset(
    file_list=file_list,
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=8,
    shuffle=True,
    drop_last=True,
)

input_size = dataset[0][0].shape[1:].numel()

for batch in train_loader:
    x, attributes = batch
    x = nn.Flatten(0, 1)(x)
    print(x.shape)
    break

model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")#, max_length=998)

with torch.no_grad():
    z = model(x, return_dict=True).last_hidden_state

print(z.shape)