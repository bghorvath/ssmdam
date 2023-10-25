import os
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule
from acoustic_anomaly_detection.model import get_model


def train(run_id: str):
    params = yaml.safe_load(open("params.yaml"))

    epochs = params["train"]["epochs"]
    run_dir = params["train"]["run_dir"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints", "train"
        )
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

        data_module = AudioDataModule()
        data_module.setup("fit")

        input_size = data_module.compute_input_size()
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
