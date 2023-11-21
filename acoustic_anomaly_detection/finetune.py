import os
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model


def finetune(run_id: str):
    params = yaml.safe_load(open("params.yaml"))

    epochs = params["train"]["epochs"]
    num_workers = params["train"]["num_workers"]
    data_sources = params["data"]["data_sources"]
    run_dir = params["train"]["run_dir"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints"
        )
        train_ckpt_path = os.path.join(ckpt_dir, "train", "best.ckpt")
        finetune_ckpt_dir = os.path.join(ckpt_dir, "finetune")

        logger = MLFlowLogger(run_id=run_id)

        data_module = AudioDataModule()
        file_lists_iter = get_file_list(stage="finetune")

        for i, (machine_type, file_list) in enumerate(file_lists_iter):
            print(
                f"Finetuning on machine type {machine_type} ({i+1}/{len(data_sources)})"
            )
            data_module.setup(file_list=file_list)
            input_size = data_module.compute_input_size()

            model = get_model(input_size=input_size)
            model.load_from_checkpoint(train_ckpt_path)
            model.freeze_encoder()

            checkpoint_callback = ModelCheckpoint(
                dirpath=finetune_ckpt_dir,
                filename=f"{machine_type}_best",
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
