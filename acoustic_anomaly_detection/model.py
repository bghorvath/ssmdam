import yaml
import torch
from torch import nn
import lightning.pytorch as pl
from torcheval.metrics.functional import binary_auroc, binary_auprc
from transformers import AutoProcessor, ASTModel

params = yaml.safe_load(open("params.yaml"))


def get_model(model_name: str, input_size: int) -> pl.LightningModule:
    model = {
        "simple_ae": SimpleAE,
        "baseline_ae": BaselineAE,
    }[params["model"]["type"]]
    return model(model_name=model_name, input_size=input_size)


class Model(pl.LightningModule):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            f"{self.model_name}_train_loss",
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
        x, _ = batch
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log(
            f"{self.model_name}_val_loss",
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
        x = nn.Flatten(0, 1)(x)
        x_hat = self(x)
        error_score = torch.mean(torch.square(x_hat - x))
        self.error_score.append(error_score.item())
        y = 1 if attributes["label"] == "anomaly" else 0
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
    def __init__(self, model_name: str, input_size: int) -> None:
        super().__init__(model_name)
        self.encoder_layers = params["model"]["layers"]["encoder"]
        self.decoder_layers = params["model"]["layers"]["decoder"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.encoder_layers[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[0], self.encoder_layers[1]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_layers[0], self.decoder_layers[1]),
            nn.ReLU(),
            nn.Linear(self.decoder_layers[1], input_size),
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

    def __init__(self, model_name: str, input_size: int) -> None:
        super().__init__(model_name)
        self.input_size = input_size
        self.encoder_layers = params["model"]["layers"]["encoder"]
        self.decoder_layers = params["model"]["layers"]["decoder"]
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.encoder_layers[0]),
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
            nn.Linear(self.decoder_layers[4], input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.Flatten(-2, -1)(x)
        z = self.encoder(z)
        z = self.decoder(z)
        return z.view(x.shape)


class ASTAE(Model):
    """
    Audio Spectrogram Transformer AutoEncoder
    Source: https://huggingface.co/transformers/model_doc/audio-spectrogram-transformer.html
    """

    def __init__(self, model_name: str, input_size: int):
        super().__init__(model_name)
        self.input_size = input_size
        self.decoder_layers = params["model"]["layers"]["decoder"]
        self.save_hyperparameters()
        self.processor = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.encoder = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
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
            nn.Linear(self.decoder_layers[4], input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = nn.Flatten(-2, -1)(x)
        with torch.no_grad():
            z = self.processor(x, return_tensors="pt", padding=True)
            z = self.encoder(z)
        z = self.decoder(z)  # TODO shapes don't match for now
        return z.view(x.shape)
