import os
import yaml
from tqdm import tqdm
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model


def evaluate(run_id: str):
    params = yaml.safe_load(open("params.yaml"))
    run_dir = params["train"]["run_dir"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints", "finetune"
        )

        logger = MLFlowLogger(run_id=run_id)

        file_list_iter = get_file_list(stage="evaluate")

        for machine_type, file_list in tqdm(file_list_iter):
            data_module = AudioDataModule(file_list=file_list)
            data_module.setup(stage="evaluate")
            input_size = data_module.compute_input_size()

            model = get_model(input_size=input_size)
            ckpt_path = os.path.join(ckpt_dir, machine_type, "best.ckpt")
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_path} does not exist. Skipping.")
                continue
            model.load_from_checkpoint(ckpt_path)

            trainer = Trainer(logger=logger)

            trainer.test(
                model=model,
                datamodule=data_module,
            )
