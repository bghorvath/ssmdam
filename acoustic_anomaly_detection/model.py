import yaml
import torch
from torch import nn
import lightning.pytorch as pl
from torcheval.metrics.functional import binary_auroc, binary_auprc

params = yaml.safe_load(open("params.yaml"))

def get_model(input_size: int) -> pl.LightningModule:
    model = {
        "simple_ae": SimpleAE,
        "baseline_ae": BaselineAE,
    }[params["model"]["type"]]
    return model(input_size=input_size, layers=params["model"]["layers"])

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        x_hat = self(x)
        error_score = torch.mean(torch.square(x_hat - x), dim=tuple(range(1, x_hat.ndim)))
        self.error_score = torch.cat([self.error_score, error_score])
        self.y = torch.cat([self.y, y])

    def on_test_epoch_start(self) -> None:
        self.error_score = torch.tensor([])
        self.y = torch.tensor([])

    def on_test_epoch_end(self) -> None:
        auroc = binary_auroc(self.error_score, self.y)
        auprc = binary_auprc(self.error_score, self.y)
        self.log('auroc_epoch', auroc, prog_bar=True, logger=True)
        self.log('auprc_epoch', auprc, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=params["train"]["lr"])

class SimpleAE(Model):
    def __init__(self, input_size: int, layers: dict) -> None:
        super().__init__()
        hidden_size = layers["hidden_size"]
        latent_size = layers["latent_size"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensors = self.flatten(x)
        tensors = self.encoder(tensors)
        tensors = self.decoder(tensors)
        return tensors.view(x.shape)

class BaselineAE(Model):
    """
    Baseline AE model
    Source: https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/networks/dcase2023t2_ae/network.py
    """
    def __init__(self, input_size: int, layers: dict) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = layers["hidden_size"]
        self.latent_size = layers["latent_size"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_size),
            nn.BatchNorm1d(self.latent_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flatten(x)
        z = self.encoder(z)
        z = self.decoder(z)
        return z.view(x.shape)
