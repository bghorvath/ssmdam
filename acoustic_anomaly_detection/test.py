import os
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import save_metrics, load_params


def test(run_id: str):
    params = load_params()

    run_dir = params["log"]["run_dir"]
    model = params["model"]["name"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        artifacts_dir = os.path.join(run_dir, experiment_id, run_id, "artifacts")
        ckpt_dir = os.path.join(artifacts_dir, "checkpoints", "train")
        ckpt_path = os.path.join(ckpt_dir, "best.ckpt")

        file_list = next(get_file_list("test"))
        data_module = AudioDataModule(file_list=file_list)
        input_size = data_module.calculate_input_size()
        model = get_model(model=model, input_size=input_size)

        logger = MLFlowLogger(run_id=run_id)

        trainer = Trainer(logger=logger)

        trainer.test(
            model=model,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )

    metrics_dict = trainer.model.performance_metrics
    save_metrics(metrics_dict, artifacts_dir, "test")
