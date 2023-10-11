import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataset
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import (
    get_groupings,
    get_train_status,
    update_train_status,
)

params = yaml.safe_load(open("params.yaml"))


def test(run_id: str):
    num_workers = params["misc"]["num_workers"]
    data_sources = params["data"]["data_sources"]
    run_dir = params["misc"]["run_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    train_status = get_train_status(run_id)

    artifact_uri = mlflow.get_artifact_uri()
    ckpt_dir = os.path.join(artifact_uri, "checkpoints", machine_type)
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    train_status = get_train_status(run_id)
    data_paths = [
        os.path.join(data_path, "test")
        for data_path, status in train_status["tested"].items()
        if not status
    ]
    print(
        f"Testing: {', '.join([data_path.split('/')[-2] for data_path in data_paths])}"
    )
    file_list = [
        os.path.join(data_path, file)
        for data_path in data_paths
        for file in os.listdir(data_path)
    ]
    dataset = AudioDataset(
        file_list=file_list,
        fast_dev_run=fast_dev_run,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )

    input_size = dataset[0][0].shape[1:].numel()
    model = get_model(input_size=input_size)
    model = model.load_from_checkpoint(ckpt_path)

    logger = MLFlowLogger(
        experiment_name="Default",
        run_id=run_id,
    )

    trainer = Trainer(logger=mlflow_logger)

    with mlflow.start_run(run_id=run_id):
        trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
