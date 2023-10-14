import os
import random
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, random_split
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram, AmplitudeToDB
import lightning.pytorch as pl
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
        attributes = get_attributes(file_path)
        return self.transform(signal, sr), attributes


class MachineTypeSampler(Sampler):
    def __init__(self, dataset, batch_size, mix_machine_types):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mix_machine_types = mix_machine_types

        # Group indices by machine type
        self.machine_type_indices = {}
        for idx, (_, attributes) in enumerate(dataset):
            machine_type = attributes["machine_type"]
            if machine_type not in self.machine_type_indices:
                self.machine_type_indices[machine_type] = []
            self.machine_type_indices[machine_type].append(idx)

    def __iter__(self):
        batch = []
        machine_types = list(self.machine_type_indices.keys())

        if self.mix_machine_types:
            all_indices = list(range(len(self.dataset)))
            random.shuffle(all_indices)
            return iter(all_indices)
        else:
            for idx in range(len(self.dataset)):
                # Select a machine type
                machine_type = random.choice(machine_types)

                # Select an instance of that machine type
                machine_idx = random.choice(self.machine_type_indices[machine_type])
                batch.append(machine_idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

    def __len__(self):
        return len(self.dataset)


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list
        self.fast_dev_run = params["data"]["fast_dev_run"]
        self.train_split = params["data"]["train_split"]
        self.batch_size = params["train"]["batch_size"]
        self.num_workers = params["misc"]["num_workers"]
        self.mix_machine_types = params["train"]["mix_machine_types"]

    def setup(self, stage=None):
        if stage == "fit":
            dataset = AudioDataset(
                file_list=self.file_list, fast_dev_run=self.fast_dev_run
            )
            train_size = int(len(dataset) * self.train_split)
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size]
            )
        elif stage == "test":
            self.test_dataset = AudioDataset(
                file_list=self.file_list, fast_dev_run=self.fast_dev_run
            )

    def train_dataloader(self):
        train_sampler = MachineTypeSampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        val_sampler = MachineTypeSampler(
            self.val_dataset, self.batch_size, self.mix_machine_types
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )
