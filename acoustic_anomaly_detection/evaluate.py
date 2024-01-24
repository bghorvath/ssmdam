import os
from tqdm import tqdm
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import load_params, save_metrics, plot_roc_curves


def evaluate(run_id: str):
    params = load_params()

    run_dir = params["log"]["run_dir"]
    model_name = params["model"]["name"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        artifacts_dir = os.path.join(run_dir, experiment_id, run_id, "artifacts")
        ckpt_root_dir = os.path.join(artifacts_dir, "checkpoints", "finetune")

        logger = MLFlowLogger(run_id=run_id)

        file_list_iter = get_file_list("evaluate")

        metrics_dict = {}
        roc_dict = {}
        for machine_type, file_list in tqdm(file_list_iter):
            ckpt_path = os.path.join(ckpt_root_dir, machine_type, "best.ckpt")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_path} does not exist. Skipping.")
                continue

            data_module = AudioDataModule(file_list=file_list)
            input_size = data_module.calculate_input_size()
            model = get_model(model=model_name, stage="evaluate", input_size=input_size)

            trainer = Trainer(logger=logger)

            trainer.test(
                model=model,
                datamodule=data_module,
                ckpt_path=ckpt_path,
            )
            metrics_dict.update(trainer.model.performance_metrics)
            roc_dict.update(trainer.model.roc)

        save_metrics(metrics_dict, artifacts_dir, "evaluate")
        plot_roc_curves(roc_dict, artifacts_dir, "evaluate")
