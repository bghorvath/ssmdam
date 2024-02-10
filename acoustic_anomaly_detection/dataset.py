import os
import random
import pickle
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torchaudio
from lightning.pytorch import LightningDataModule
import mlflow
from acoustic_anomaly_detection.utils import get_attributes, load_params


def get_file_list(stage: str) -> list or tuple[str, list]:
    params = load_params()
    data_sources = params["data"]["data_sources"]
    dev_eval = "dev" if stage in ("fit", "test") else "eval"
    train_test = "train" if stage in ("fit", "finetune") else "test"
    data_paths = [
        os.path.join("data", data_source, dev_eval, data_dir, train_test)
        for data_source in data_sources
        for data_dir in os.listdir(os.path.join("data", data_source, dev_eval))
    ]
    if dev_eval == "eval":
        for data_path in data_paths:
            machine_type = data_path.split("/")[-2]
            file_paths = [
                os.path.join(data_path, file) for file in os.listdir(data_path)
            ]
            yield machine_type, file_paths
    else:
        file_list = [
            os.path.join(data_path, file)
            for data_path in data_paths
            for file in os.listdir(data_path)
        ]
        yield file_list


def get_label_list(stage: str = None) -> list:
    fit_paths = [path for data_path in get_file_list("fit") for path in data_path]
    finetune_paths = [
        path for _, data_paths in get_file_list("finetune") for path in data_paths
    ]
    all_paths = fit_paths + finetune_paths
    if stage == "fit":
        file_paths = fit_paths
    elif stage == "finetune":
        file_paths = finetune_paths
    else:
        file_paths = all_paths

    attributes_list = [get_attributes(path) for path in file_paths]
    return [create_label(attributes) for attributes in attributes_list]


def create_label(attributes: list[dict[str, str]]) -> list[str]:
    return "_".join([f"{k}_{v}" for k, v in attributes.items()])


def fit_label_encoder() -> None:
    label_list = get_label_list()
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    mlflow.log_artifact("label_encoder.pkl")


class AudioDataset(Dataset):
    def __init__(
        self,
        file_list: list,
    ) -> None:
        self.file_list = file_list
        params = load_params()
        # self.fast_dev_run = params["data"]["fast_dev_run"]
        self.seed = params["data"]["seed"]
        self.segment = params["data"]["transform"]["segment"]
        self.sr = params["data"]["transform"]["sr"]
        self.duration = params["data"]["transform"]["duration"]
        self.length = self.sr * self.duration

        # if self.fast_dev_run and len(self.file_list) > 100:
        #     random.seed(self.seed)
        #     self.file_list = random.sample(self.file_list, 100)

        self.attributes_list = self.get_attributes_list()

        with open("label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

    def get_attributes_list(self):
        attributes_list = [get_attributes(fp) for fp in self.file_list]

        # all_keys = set().union(*(attr.keys() for attr in attributes_list))
        # for attr in attributes_list:
        #     for key in all_keys:
        #         attr.setdefault(key, "None")

        return attributes_list

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
        # signal = signal.squeeze(0)
        return signal

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        file_path = self.file_list[idx]
        attributes = self.attributes_list[idx]
        label = create_label(attributes)
        label = torch.nn.functional.one_hot(
            torch.tensor(self.label_encoder.transform([label])),
            num_classes=len(self.label_encoder.classes_),
        ).squeeze(0)
        signal, sr = torchaudio.load(file_path)
        if signal is None:
            raise ValueError(
                f"Signal loading failed for index: {idx}, file_path: {file_path}"
            )
        if attributes is None:
            raise ValueError(
                f"Attributes loading failed for index: {idx}, file_path: {file_path}"
            )

        transformed_data = self.transform(signal, sr)
        if transformed_data is None or any(v is None for v in transformed_data):
            raise ValueError(f"Data loading failed for index: {idx}")
        return transformed_data, label, attributes


class AudioDataModule(LightningDataModule):
    def __init__(self, file_list: list) -> None:
        super().__init__()
        params = load_params()
        self.train_split = params["data"]["train_split"]
        self.batch_size = params["data"]["batch_size"]
        self.num_workers = params["data"]["num_workers"]
        self.mix_machine_types = params["data"]["mix_machine_types"]
        self.window_size = params["data"]["transform"]["window_size"]
        self.seed = params["data"]["seed"]
        self.file_list = file_list

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            self.dataset, self.val_dataset = self.train_val_split(
                file_list=self.file_list
            )
        elif stage == "test":
            self.dataset = AudioDataset(file_list=self.file_list)

        self.train_batch_sampler = MachineTypeBatchSampler(
            dataset=self.dataset,
            batch_size=self.batch_size,
            seed=self.seed,
            mix_machine_types=self.mix_machine_types,
        )

    def train_val_split(self, file_list: list) -> tuple[Dataset, Dataset]:
        random.seed(self.seed)
        random.shuffle(file_list)
        train_size = int(len(file_list) * self.train_split)
        train_file_list = file_list[:train_size]
        val_file_list = file_list[train_size:]
        train_dataset = AudioDataset(file_list=train_file_list)
        val_dataset = AudioDataset(file_list=val_file_list)
        return train_dataset, val_dataset

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            batch_sampler=self.train_batch_sampler,
        )
        return dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def calculate_input_size(self):
        recon_dataset = AudioDataset(file_list=self.file_list[:1])
        return recon_dataset[0][0].shape[1] * self.window_size

    def reshuffle_train_batches(self) -> None:
        self.seed += 1
        self.train_batch_sampler.shuffle_batches(seed=self.seed)


class MachineTypeBatchSampler(BatchSampler):
    def __init__(
        self, dataset: Dataset, batch_size: int, seed: int, mix_machine_types: bool
    ) -> None:
        self.batch_size = batch_size
        self.mix_machine_types = mix_machine_types

        self.dataset_length = len(dataset)

        # Group the indices by machine type
        self.indices_by_type = {}
        for idx, attributes in enumerate(dataset.attributes_list):
            machine_type = attributes["machine_type"]
            if machine_type not in self.indices_by_type:
                self.indices_by_type[machine_type] = []
            self.indices_by_type[machine_type].append(idx)

        self.shuffle_batches(seed=seed)

    # Create shuffled index batches
    def shuffle_batches(self, seed: int) -> None:
        random.seed(seed)

        # Shuffle indices within each machine type group
        for indices in self.indices_by_type.values():
            random.shuffle(indices)

        # Create batches of indices
        batches = []
        if self.mix_machine_types:
            num_batches = self.dataset_length // self.batch_size
            last_index = num_batches * self.batch_size
            all_indices = [
                idx for indices in self.indices_by_type.values() for idx in indices
            ]
            random.shuffle(all_indices)
            batches = [
                all_indices[i : i + self.batch_size]
                for i in range(0, last_index, self.batch_size)
            ]
        else:
            for machine_type, indices in self.indices_by_type.items():
                num_batches = len(indices) // self.batch_size
                for i in range(num_batches):
                    batches.append(
                        indices[i * self.batch_size : (i + 1) * self.batch_size]
                    )

        # Shuffle the batches
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
