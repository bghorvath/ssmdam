import os
import random
import yaml
import torch
from torch.utils.data import Dataset
import torchaudio

from acoustic_anomaly_detection.utils import get_attributes

params = yaml.safe_load(open("params.yaml"))


class AudioDataset(Dataset):
    def __init__(
        self,
        file_list: list,
        fast_dev_run: bool = False,
    ) -> None:
        self.file_list = file_list
        self.seed = params["train"]["seed"]

        if fast_dev_run:
            random.seed(self.seed)
            self.file_list = random.sample(self.file_list, 100)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, str]]:
        file_path = self.file_list[idx]
        attributes = get_attributes(os.path.basename(file_path))
        signal = torch.load(file_path)
        return signal, attributes
