import yaml
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dvclive import Live
from dvclive.lightning import DVCLiveLogger

from acoustic_anomaly_detection.dataset import AudioDataset
from acoustic_anomaly_detection.model import LitAutoEncoder

params = yaml.safe_load(open("params.yaml"))

def main():
    dataset = AudioDataset()
    train_split = params["data"]["train_split"]
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=params["train"]["batch_size"], num_workers=params["train"]["num_workers"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["train"]["batch_size"], num_workers=params["train"]["num_workers"], shuffle=False)
    model = LitAutoEncoder(input_size=params["transform"]["params"]["n_mels"] * params["transform"]["params"]["n_fft"] // 2 + 1, hidden_size=params["model"]["hidden_size"], latent_size=params["model"]["latent_size"])
    with Live(save_dvc_exp=True) as live:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='models',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        trainer = Trainer(
            logger=DVCLiveLogger(),
            max_epochs=params["max_epochs"],
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, val_loader)
