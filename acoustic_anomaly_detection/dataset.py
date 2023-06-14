import os
import yaml
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram

params = yaml.safe_load(open("params.yaml"))

class AudioDataset(Dataset):
    def __init__(self) -> None:
        self.audio_dirs = params["data"]["audio_dirs"]

        self.file_list = []
        for audio_dir in self.audio_dirs:
            audio_path = os.path.join(audio_dir, "train")
            file_list = [os.path.join(audio_path, file) for file in os.listdir(audio_path)]
            self.file_list += file_list

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

    def __getitem__(self, idx) -> tuple:
        audio_path = self.file_list[idx]
        label = os.path.basename(audio_path).split("_")[0]
        signal, _ = torchaudio.load(audio_path)
        signal = self.transform(signal)
        return signal, label
