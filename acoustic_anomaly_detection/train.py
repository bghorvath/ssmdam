import os
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import load_params


def train(run_id: str):
    params = load_params()

    run_dir = params["log"]["run_dir"]
    epochs = params["train"]["epochs"]
    model = params["model"]["name"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints", "train"
        )
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

        file_list = next(get_file_list("fit"))
        data_module = AudioDataModule(file_list=file_list)
        input_size = data_module.calculate_input_size()
        model = get_model(model=model, stage="train", input_size=input_size)

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss_epoch",
            mode="min",
            filename="best",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )

        logger = MLFlowLogger(run_id=run_id)

        trainer = Trainer(
            log_every_n_steps=1,
            logger=logger,
            max_epochs=epochs,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path="last" if os.path.exists(ckpt_path) else None,
        )
