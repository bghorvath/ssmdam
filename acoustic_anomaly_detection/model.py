import yaml
import torch
from torch import nn
import lightning.pytorch as pl
from torcheval.metrics.functional import binary_auroc, binary_auprc

params = yaml.safe_load(open("params.yaml"))


def get_model(model_name: str, input_size: int) -> pl.LightningModule:
    model = {
        "simple_ae": SimpleAE,
        "baseline_ae": BaselineAE,
    }[params["model"]["type"]]
    return model(
        model_name=model_name, input_size=input_size, layers=params["model"]["layers"]
    )


class Model(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            f"{self.model_name}_train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            f"{self.model_name}_val_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        error_score = torch.mean(torch.square(x_hat - x))
        self.error_score.append(error_score.item())
        self.y.append(y.item())

    def on_test_epoch_start(self) -> None:
        self.error_score = []
        self.y = []

    def on_test_epoch_end(self) -> None:
        error_score = torch.tensor(self.error_score)
        y = torch.tensor(self.y)
        auroc = binary_auroc(error_score, y)
        auprc = binary_auprc(error_score, y)
        self.log(f"{self.model_name}_auroc_epoch", auroc, prog_bar=True, logger=True)
        self.log(f"{self.model_name}_auprc_epoch", auprc, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=params["train"]["lr"])


class SimpleAE(Model):
    def __init__(self, model_name: str, input_size: int, layers: dict) -> None:
        super().__init__(model_name)
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
        z = nn.Flatten(1, 2)(x)
        z = self.encoder(z)
        z = self.decoder(z)
        return z.view(x.shape)


class BaselineAE(Model):
    """
    Baseline AE model
    Source: https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/networks/dcase2023t2_ae/network.py
    """

    def __init__(self, model_name: str, input_size: int, layers: dict) -> None:
        super().__init__(model_name)
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
            nn.Linear(self.hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.Flatten(-2, -1)(x)
        z = self.encoder(z)
        z = self.decoder(z)
        return z.view(x.shape)
