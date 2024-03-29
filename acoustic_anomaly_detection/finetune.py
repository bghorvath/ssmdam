import os
from tqdm import tqdm
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from acoustic_anomaly_detection.dataset import AudioDataModule, get_file_list
from acoustic_anomaly_detection.model import get_model
from acoustic_anomaly_detection.utils import load_params


def finetune(run_id: str):
    params = load_params()

    run_dir = params["log"]["run_dir"]
    epochs = params["finetune"]["epochs"]
    model_name = params["model"]["name"]
    fast_dev_run = params["data"]["fast_dev_run"]

    with mlflow.start_run(run_id=run_id) as mlrun:
        experiment_id = mlrun.info.experiment_id
        ckpt_dir = os.path.join(
            run_dir, experiment_id, run_id, "artifacts", "checkpoints"
        )
        train_ckpt_path = os.path.join(ckpt_dir, "train", "best.ckpt")
        finetune_ckpt_root_dir = os.path.join(ckpt_dir, "finetune")

        logger = MLFlowLogger(run_id=run_id)

        file_lists_iter = get_file_list("finetune")

        for machine_type, file_list in tqdm(file_lists_iter):
            data_module = AudioDataModule(file_list=file_list)
            input_size = data_module.calculate_input_size()
            finetune_ckpt_dir = os.path.join(finetune_ckpt_root_dir, machine_type)

            model = get_model(model=model_name, stage="finetune", input_size=input_size)
            if not os.path.exists(finetune_ckpt_dir):
                model.load_state_dict(torch.load(train_ckpt_path)["state_dict"])
            model.freeze_encoder()

            checkpoint_callback = ModelCheckpoint(
                dirpath=finetune_ckpt_dir,
                monitor="val_loss",
                mode="min",
                filename="best",
                save_top_k=1,
                save_last=True,
                verbose=True,
            )

            trainer = Trainer(
                log_every_n_steps=1,
                logger=logger,
                max_epochs=epochs,
                callbacks=[checkpoint_callback],
            )

            trainer.fit(
                model=model,
                datamodule=data_module,
                ckpt_path="last" if os.path.exists(finetune_ckpt_dir) else None,
            )

            if fast_dev_run:
                break
