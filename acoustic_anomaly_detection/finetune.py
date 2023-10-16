import os
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule
from acoustic_anomaly_detection.model import get_model


def finetune(run_id: str):
    params = yaml.safe_load(open("params.yaml"))

    epochs = params["train"]["epochs"]
    num_workers = params["misc"]["num_workers"]
    data_sources = params["data"]["data_sources"]
    run_dir = params["misc"]["run_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints"
        )
        train_ckpt_path = os.path.join(ckpt_dir, "train", "best.ckpt")
        finetune_ckpt_dir = os.path.join(ckpt_dir, "finetune")

        data_module = AudioDataModule()
        data_module.setup("finetune")

        input_size = data_module.compute_input_size()
        model = get_model(input_size=input_size)
        model.load_from_checkpoint(train_ckpt_path)
        model.freeze_encoder()

        logger = MLFlowLogger(run_id=run_id)

        checkpoint_callback = ModelCheckpoint(
            dirpath=finetune_ckpt_dir,
            filename="finetuned_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback],
            max_epochs=epochs,
            num_workers=num_workers,
            fast_dev_run=fast_dev_run,
        )

        trainer.fit(
            model=model,
            datamodule=data_module,
        )
