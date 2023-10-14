import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import get_train_status, update_train_status


def train(run_id: str):
    params = yaml.safe_load(open("params.yaml"))

    seed = params["train"]["seed"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    num_workers = params["misc"]["num_workers"]
    run_dir = params["misc"]["run_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]
    train_split = params["data"]["train_split"]
    window_size = params["transform"]["params"]["window_size"]

    artifact_uri = mlflow.get_artifact_uri()
    ckpt_dir = os.path.join(artifact_uri, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    generator = torch.Generator().manual_seed(seed)

    train_status = get_train_status(run_id)
    data_paths = [
        os.path.join(data_path, "train")
        for data_path, status in train_status["trained"].items()
        if not status
    ]
    file_list = [
        os.path.join(data_path, file)
        for data_path in data_paths
        for file in os.listdir(data_path)
    ]
    data_module = AudioDataModule(file_list=file_list)

    input_size = train_dataset[0][0].shape[1] * window_size
    model = get_model(input_size=input_size)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        filename="best",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    logger = MLFlowLogger(
        experiment_name="Default",
        run_id=run_id,
    )

    trainer = Trainer(
        log_every_n_steps=1,
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path="last" if os.path.exists(ckpt_path) else None,
    )
