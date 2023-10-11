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
from acoustic_anomaly_detection.dataset import AudioDataset
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import get_train_status, update_train_status

params = yaml.safe_load(open("params.yaml"))


def train(run_id: str):
    seed = params["train"]["seed"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    num_workers = params["misc"]["num_workers"]
    run_dir = params["misc"]["run_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]
    train_split = params["data"]["train_split"]
    window_size = params["transform"]["params"]["window_size"]

    input_size = None

    with mlflow.start_run(run_id=run_id):
        train_status = get_train_status(run_id)

        artifact_uri = mlflow.get_artifact_uri()
        ckpt_dir = os.path.join(artifact_uri, "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

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
            experiment_name=mlflow.active_run().info.experiment_name,
            run_id=mlflow.active_run().info.run_id,
        )

        trainer = Trainer(
            log_every_n_steps=1,
            logger=logger,
            max_epochs=epochs,
            callbacks=[checkpoint_callback],
        )

        data_paths = [
            data_path
            for data_path, status in train_status["trained"].items()
            if not status
        ]
        print(
            f"Training: {', '.join([data_path.split('/')[-1] for data_path in data_paths])}"
        )

        for i, data_path in enumerate(tqdm(data_paths)):
            _, data_source, _, machine_type = data_path.split("/")
            print(
                f"Training ({i+1}/{len(audio_dirs)} - data_source: {data_source} machine type: {machine_type})"
            )
            audio_dir = os.path.join(data_path, "train")
            file_list = [
                os.path.join(audio_dir, file) for file in os.listdir(audio_dir)
            ]

            dataset = AudioDataset(
                file_list=file_list,
                fast_dev_run=fast_dev_run,
            )
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

            if not input_size:
                input_size = train_dataset[0][0].shape[1] * window_size
                model = get_model(input_size=input_size)

            logger._prefix = machine_type

            trainer.fit(
                model,
                train_loader,
                val_loader,
                ckpt_path="last" if os.path.exists(ckpt_path) else None,
            )

            update_train_status(run_id, "dev", data_path, "trained")


if __name__ == "__main__":
    train()
