import yaml
import torch
from torch import nn
import lightning.pytorch as pl
from torcheval.metrics.functional import binary_auroc, binary_auprc
from transformers import ASTModel

from acoustic_anomaly_detection.utils import slice_signal, reconstruct_signal

params = yaml.safe_load(open("params.yaml"))


def get_model(input_size: int) -> pl.LightningModule:
    model_type = params["model"]["type"]
    model = {
        "simple_ae": SimpleAE,
        "baseline_ae": BaselineAE,
    }[model_type]
    return model(input_size=input_size)


class Model(pl.LightningModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.init_transformer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, attributes = batch
        machine_type = attributes["machine_type"]
        x = self.transform(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, attributes = batch
        machine_type = attributes["machine_type"]
        x = self.transform(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> None:
        x, attributes = batch
        label = attributes["label"]
        machine_type = attributes["machine_type"][0]
        x = self.transform(x)
        x_hat = self(x)
        error_score = torch.mean(torch.square(x_hat - x))

        if machine_type not in self.error_scores:
            self.error_scores[machine_type] = []
            self.ys[machine_type] = []

        self.error_scores[machine_type].append(error_score.item())
        y = 1 if label == "anomaly" else 0
        self.ys[machine_type].append(y)

    def on_test_epoch_start(self) -> None:
        self.error_scores = {}
        self.ys = {}

    def on_test_epoch_end(self) -> None:
        for machine_type, error_score in self.error_scores.items():
            error_score = torch.tensor(error_score)
            y = self.ys[machine_type]
            y = torch.tensor(y)

            auroc = binary_auroc(error_score, y).float()
            auprc = binary_auprc(error_score, y).float()

            self.log(f"{machine_type}_auroc_epoch", auroc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_auprc_epoch", auprc, prog_bar=True, logger=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=params["train"]["lr"])

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if params["transform"]["type"] == "ast":
            with torch.no_grad():
                return self.transformer(x).last_hidden_state
        return x

    def init_transformer(self) -> None:
        if params["transform"]["type"] == "ast":
            self.transformer = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.input_size = 3840


class SimpleAE(Model):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)
        self.encoder_layers = params["model"]["layers"]["encoder"]
        self.decoder_layers = params["model"]["layers"]["decoder"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.encoder_layers[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[0], self.encoder_layers[1]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_layers[0], self.decoder_layers[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[1], self.input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        z = slice_signal(x)
        z = nn.Flatten(-2, -1)(z)
        z = self.encoder(z)
        z = self.decoder(z)
        z = reconstruct_signal(z, batch_size)
        return z.view(x.shape)


class BaselineAE(Model):
    """
    Baseline AE model
    Source: https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/networks/dcase2023t2_ae/network.py
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)
        self.encoder_layers = params["model"]["layers"]["encoder"]
        self.decoder_layers = params["model"]["layers"]["decoder"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.encoder_layers[0]),
            nn.BatchNorm1d(self.encoder_layers[0], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[0], self.encoder_layers[1]),
            nn.BatchNorm1d(self.encoder_layers[1], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[1], self.encoder_layers[2]),
            nn.BatchNorm1d(self.encoder_layers[2], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[2], self.encoder_layers[3]),
            nn.BatchNorm1d(self.encoder_layers[3], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[3], self.encoder_layers[4]),
            nn.BatchNorm1d(self.encoder_layers[4], momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_layers[0], self.decoder_layers[1]),
            nn.BatchNorm1d(self.decoder_layers[1], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[1], self.decoder_layers[2]),
            nn.BatchNorm1d(self.decoder_layers[2], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[2], self.decoder_layers[3]),
            nn.BatchNorm1d(self.decoder_layers[3], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[3], self.decoder_layers[4]),
            nn.BatchNorm1d(self.decoder_layers[4], momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[4], self.input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # [batch_size, 313, 128]
        z = slice_signal(x)
        # [batch_size, 309, 5, 128]
        z = nn.Flatten(0, 1)(z)
        # [batch_size * 309, 5, 128]
        z = nn.Flatten(-2, -1)(z)
        # [batch_size * 309, 5 * 128]
        z = self.encoder(z)
        z = self.decoder(z)
        z = reconstruct_signal(z, batch_size)
        return z
