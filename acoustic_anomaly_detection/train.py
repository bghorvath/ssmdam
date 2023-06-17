import yaml
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from dataset import AudioDataset
from model import LitAutoEncoder

params = yaml.safe_load(open("params.yaml"))

def train():
    dataset = AudioDataset(fast_dev_run=True)
    train_split = params["data"]["train_split"]
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    seed = params["train"]["seed"]
    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    log_dir = params["misc"]["log_dir"]
    num_workers = params["misc"]["num_workers"]
    ckpt_path = params["misc"]["ckpt_path"]
    ckpt_dir_path = "/".join(ckpt_path.split("/")[:-1])
    ckpt_filename = ckpt_path.split("/")[-1].split(".")[0]

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    input_size = train_dataset[0][0].shape[1:].numel()

    model = {
        "autoencoder": LitAutoEncoder,
    }[params["model"]["type"]]
    model_params = {k:v for k, v in params["model"]["params"].items() if k in model.__init__.__code__.co_varnames}
    model = model(input_size=input_size, **model_params)

    with Live(dir=log_dir) as live:
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir_path,
            monitor='val_loss',
            filename=ckpt_filename,
        )
        trainer = Trainer(
            logger=DVCLiveLogger(experiment=live),
            max_epochs=epochs,
            callbacks=checkpoint,
        )
        trainer.fit(model, train_loader, val_loader)
        # live.log_artifact(
        #     checkpoint.best_model_path,
        #     type="model",
        #     name="best"
        # )

if __name__ == "__main__":
    train()
