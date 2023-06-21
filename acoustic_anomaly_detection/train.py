import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import mlflow
from model import get_model
from dataset import AudioDataset
from utils import get_runs_scheduled, get_previous_runs


params = yaml.safe_load(open("params.yaml"))


def train_run(
    machine_type: str,
    machine_id: str,
    run_id: str,
):
    epochs = params["train"]["epochs"]
    log_dir = params["misc"]["log_dir"]
    seed = params["train"]["seed"]
    batch_size = params["train"]["batch_size"]
    num_workers = params["misc"]["num_workers"]

    train_dataset = AudioDataset(
        machine_type=machine_type,
        machine_id=machine_id,
        fast_dev_run=params["data"]["fast_dev_run"],
    )
    train_split = params["data"]["train_split"]
    train_size = int(len(train_dataset) * train_split)
    val_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], generator=generator
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

    input_size = train_dataset[0][0].shape[1:].numel()
    model = get_model(input_size=input_size)

    checkpoint = ModelCheckpoint(
        dirpath=log_dir,
        monitor="val_loss",
        filename="best",
    )

    exp_name = f"{machine_type}_{machine_id}"

    trainer = Trainer(
        logger=mlflow.pytorch.MLFlowLogger(
            experiment_name=exp_name,
            tracking_uri=log_dir,
        ),
        max_epochs=epochs,
        callbacks=checkpoint,
    )

    ckpt_path = None
    if run_id is not None:
        ckpt_path = os.path.join(log_dir, run_id, "checkpoints", "best.ckpt")
        model = model.load_from_checkpoint(ckpt_path)

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


def train():
    log_dir = params["misc"]["log_dir"]
    mlflow.set_tracking_uri(log_dir)

    fresh_start = params["train"]["fresh_start"]

    runs_scheduled = get_runs_scheduled()

    parent_run_id = ""
    if not fresh_start:
        completed_runs, incomplete_runs, parent_run_id = get_previous_runs()
        runs_scheduled = {
            k: v for k, v in runs_scheduled.items() if k not in completed_runs
        }
        runs_scheduled.update(incomplete_runs)

    if len(runs_scheduled) == 0:
        print("No runs scheduled")
        return

    with mlflow.start_run(run_id=parent_run_id) as run:
        for (machine_type, machine_id), child_run_id in tqdm(runs_scheduled.items()):
            with mlflow.start_run(
                run_name=f"{machine_type}_{machine_id}",
                run_id=child_run_id,
                nested=True,
            ) as child_run:
                train_run(machine_type, machine_id, run_id=child_run_id)
