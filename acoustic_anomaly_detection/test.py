import os
import yaml
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model


def test(run_id: str):
    params = yaml.safe_load(open("params.yaml"))

    num_workers = params["train"]["num_workers"]
    data_sources = params["data"]["data_sources"]
    run_dir = params["train"]["run_dir"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints", "train"
        )
        ckpt_path = os.path.join(ckpt_dir, "best.ckpt")

        data_module = AudioDataModule()
        file_list = get_file_list(stage="test")
        data_module.setup(file_list=file_list)

        input_size = data_module.compute_input_size()
        model = get_model(input_size=input_size)

        logger = MLFlowLogger(run_id=run_id)

        trainer = Trainer(logger=logger)

        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
