import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from acoustic_anomaly_detection.dataset import AudioDataset
from acoustic_anomaly_detection.model import get_model

params = yaml.safe_load(open("params.yaml"))


def train():
    data_sources = params["data"]["data_sources"]
    seed = params["train"]["seed"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    log_dir = params["misc"]["log_dir"]
    num_workers = params["misc"]["num_workers"]
    ckpt_dir = params["misc"]["ckpt_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]
    train_split = params["data"]["train_split"]
    window_size = params["transform"]["params"]["window_size"]

    with Live(dir=log_dir, save_dvc_exp=False) as live:
        for i, data_source in enumerate(tqdm(data_sources)):
            print(f"Training ({i+1}/{len(data_sources)} data source: {data_source})")
            audio_dirs_path = os.path.join("data", "raw", data_source, "dev")
            audio_dirs = [
                os.path.join(audio_dirs_path, dir)
                for dir in os.listdir(audio_dirs_path)
            ]
            for j, audio_dir in enumerate(tqdm(audio_dirs)):
                machine_type = audio_dir.split("/")[-1]
                print(
                    f"Training ({j+1}/{len(audio_dirs)} machine type: {machine_type})"
                )
                audio_dir = os.path.join(audio_dir, "train")
                file_list = [
                    os.path.join(audio_dir, file) for file in os.listdir(audio_dir)
                ]

                dataset = AudioDataset(
                    file_list=file_list,
                    fast_dev_run=fast_dev_run,
                )
                train_split = train_split
                train_size = int(len(dataset) * train_split)
                val_size = len(dataset) - train_size

                generator = torch.Generator().manual_seed(seed)
                train_dataset, val_dataset = random_split(
                    dataset, [train_size, val_size], generator=generator
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=True,
                )

                input_size = train_dataset[0][0].shape[1] * window_size

                model = get_model(model_name=machine_type, input_size=input_size)

                ckpt_path = os.path.join(ckpt_dir, f"{machine_type}.ckpt")
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)

                checkpoint = ModelCheckpoint(
                    dirpath=ckpt_dir,
                    monitor=f"{machine_type}_val_loss",
                    filename=machine_type,
                    save_top_k=1,
                )

                trainer = Trainer(
                    log_every_n_steps=1,
                    logger=DVCLiveLogger(
                        experiment=live,
                    ),
                    max_epochs=epochs,
                    callbacks=checkpoint,
                )
                trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
