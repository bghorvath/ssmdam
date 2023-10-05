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

from dataset import AudioDataset
from model import get_model

params = yaml.safe_load(open("params.yaml"))


def finetune():
    data_sources = params["data"]["data_sources"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    log_dir = params["misc"]["log_dir"]
    num_workers = params["misc"]["num_workers"]
    ckpt_dir = params["misc"]["ckpt_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with Live(
        dir=log_dir,
        resume=True,
    ) as live:
        for i, data_source in enumerate(tqdm(data_sources)):
            print(f"Finetuning ({i+1}/{len(data_sources)} data source: {data_source})")
            audio_dirs_path = os.path.join("data", "prepared", data_source, "eval")
            audio_dirs = [
                os.path.join(audio_dirs_path, dir)
                for dir in os.listdir(audio_dirs_path)
            ]
            for j, audio_dir in enumerate(tqdm(audio_dirs)):
                machine_type = audio_dir.split("/")[-1]
                print(
                    f"Finetuning ({j+1}/{len(audio_dirs)} machine type: {machine_type})"
                )
                audio_dir = os.path.join(audio_dir, "train")
                file_list = [
                    os.path.join(audio_dir, file)
                    for file in os.listdir(audio_dir)
                ]

                dataset = AudioDataset(
                    file_list=file_list,
                    fast_dev_run=fast_dev_run,
                )

                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                )

                input_size = dataset[0][0].shape[1:].numel()

                ckpt_path = os.path.join(ckpt_dir, machine_type + ".ckpt")

                if not os.path.exists(ckpt_path):
                    print(f"Model for {machine_type} not found. Skipping...")
                    continue

                if not os.path.exists(audio_dir):
                    print(f"Test data for {machine_type} not found. Skipping...")
                    continue

                checkpoint = ModelCheckpoint(
                    dirpath=ckpt_dir,
                    monitor=f"{machine_type}_val_loss",
                    filename=machine_type,
                    save_top_k=1,
                )

                logger = DVCLiveLogger(experiment=live)
                
                trainer = Trainer(
                    log_every_n_steps=1,
                    logger=DVCLiveLogger(
                        experiment=live,
                    ),
                    max_epochs=epochs,
                    callbacks=checkpoint,
                )
                trainer.fit(model, train_loader, val_loader)