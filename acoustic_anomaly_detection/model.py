from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
import lightning.pytorch as pl
from transformers import ASTModel
from acoustic_anomaly_detection.utils import (
    slice_signal,
    reconstruct_signal,
    load_params,
    calculate_metrics,
    min_max_scaler,
)


def get_model(model: str, input_size: int, lr: float = 1e-3) -> pl.LightningModule:
    model_cls = {
        "ae": AutoEncoder,
    }[model]
    return model_cls(input_size, lr)


class Model(pl.LightningModule):
    def __init__(self, input_size: int, lr: float) -> None:
        super().__init__()

        params = load_params()
        self.model = params["model"]["name"]
        self.layers = params["model"]["layers"]
        self.max_fpr = params["classification"]["max_fpr"]
        self.decision_threshold = 0.5  # params["classification"]["decision_threshold"]
        self.mix_machine_types = params["data"]["mix_machine_types"]
        self.transform_type = params["data"]["transform"]["name"]
        self.window_size = params["data"]["transform"]["window_size"]
        self.stride = params["data"]["transform"]["stride"]
        self.input_size = input_size
        self.lr = lr
        self.init_transformer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, attributes = batch
        machine_types = attributes["machine_type"]
        x = self.transform(x)
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat, attributes)

        if self.mix_machine_types:
            for i, machine_type in enumerate(machine_types):
                self.train_error_scores[machine_type].append(loss[i].item())
        else:
            self.train_error_scores[machine_types[0]].append(loss.mean().item())

        self.log(
            "train_loss",
            loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0],
        )
        return loss.mean()

    def on_train_epoch_start(self) -> None:
        self.train_error_scores = defaultdict(list)

        # Reshuffle the batches of the training dataloader
        if self.current_epoch > 0:
            self.trainer.datamodule.reshuffle_train_batches()

    def on_train_epoch_end(self) -> None:
        for machine_type, error_score in self.train_error_scores.items():
            error_score = torch.tensor(error_score)
            mean_error_score = torch.mean(error_score)

            self.log(
                f"{machine_type}_train_loss_epoch",
                mean_error_score,
                prog_bar=True,
                logger=True,
            )

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, attributes = batch
        machine_type = attributes["machine_type"][0]
        # label = attributes["label"][0]
        x = self.transform(x)
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat, attributes).mean()

        self.val_error_scores[machine_type].append(loss.item())

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:  # TODO: Remove or fix
        self.val_error_scores = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        for machine_type, error_score in self.val_error_scores.items():
            error_score = torch.tensor(error_score)
            mean_error_score = torch.mean(error_score)

            self.log(
                f"{machine_type}_val_loss_epoch",
                mean_error_score,
                prog_bar=True,
                logger=True,
            )

    def test_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> None:
        x, attributes = batch
        label = attributes["label"][0]
        machine_type = attributes["machine_type"][0]
        domain = attributes["domain"][0]
        y = 1 if label == "anomaly" else 0

        x = self.transform(x)
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat, attributes).mean()

        self.test_loss[machine_type].append(loss.item())
        self.test_y_true[machine_type].append(y)
        self.test_domain[machine_type].append(domain)

    def on_test_epoch_start(self) -> None:
        self.test_loss = defaultdict(list)
        self.test_y_true = defaultdict(list)
        self.test_domain = defaultdict(list)
        self.performance_metrics = {}

    def on_test_epoch_end(self) -> None:
        for machine_type, loss in self.test_loss.items():
            loss = torch.tensor(loss)
            y_true = self.test_y_true[machine_type]
            y_true = torch.tensor(y_true)
            domain_dict = {"source": 0, "target": 1}
            domain_true = torch.tensor(
                [domain_dict[domain] for domain in self.test_domain[machine_type]]
            )
            error_score = self.calculate_error_score(loss)
            # Calculate metrics for source and target domains combined
            auc, p_auc, prec, recall, f1 = calculate_metrics(
                error_score, y_true, self.max_fpr, self.decision_threshold
            )

            self.log(f"{machine_type}_auc_epoch", auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_p_auc_epoch", p_auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_prec_epoch", prec, prog_bar=True, logger=True)
            self.log(f"{machine_type}_recall_epoch", recall, prog_bar=True, logger=True)
            self.log(f"{machine_type}_f1_epoch", f1, prog_bar=True, logger=True)

            machine_metrics = [auc, p_auc, prec, recall, f1]

            # Calculate metrics for source and target domains separately
            for domain in ("source", "target"):
                y_true_domain_auc = y_true[
                    (domain_true == domain_dict[domain]) | (y_true == 1)
                ]
                y_pred_domain_auc = error_score[
                    (domain_true == domain_dict[domain]) | (y_true == 1)
                ]
                # y_true_domain = y_true[domain_true == domain_dict[domain]]
                # y_pred_domain = error_score[domain_true == domain_dict[domain]]

                auc, p_auc, prec, recall, f1 = calculate_metrics(
                    y_pred_domain_auc,
                    y_true_domain_auc,
                    self.max_fpr,
                    self.decision_threshold,
                )

                self.log(
                    f"{machine_type}_{domain}_auc_epoch",
                    auc,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{machine_type}_{domain}_p_auc_epoch",
                    p_auc,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{machine_type}_{domain}_prec_epoch",
                    prec,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{machine_type}_{domain}_recall_epoch",
                    recall,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"{machine_type}_{domain}_f1_epoch", f1, prog_bar=True, logger=True
                )

                machine_metrics += [auc, p_auc, prec, recall, f1]

            self.performance_metrics[machine_type] = np.array(machine_metrics)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def calculate_loss(
        self, x: torch.Tensor, y: torch.Tensor, attributes: dict
    ) -> torch.Tensor:
        if self.loss_fn == "mse":
            loss = mse_loss(x, y, reduction="none")
            return loss.mean(dim=(1, 2))
        elif self.loss_fn == "cross_entropy":
            return cross_entropy(y, attributes)

    @staticmethod
    def calculate_error_score(loss: torch.Tensor) -> torch.Tensor:
        return min_max_scaler(loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "ast":
            with torch.no_grad():
                return self.transformer(x).last_hidden_state
        return x

    def init_transformer(self) -> None:
        if self.transform_type == "ast":
            self.transformer = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.input_size = 3840


class AutoEncoder(Model):
    """
    Baseline AE model
    Source: https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/networks/dcase2023t2_ae/network.py
    """

    def __init__(self, input_size: int, lr: float) -> None:
        super().__init__(input_size, lr)
        self.loss_fn = "mse"
        self.init_encoder_decoder()

    def init_encoder_decoder(self):
        encoder_layers = self.layers["encoder"]
        decoder_layers = self.layers["decoder"]
        encoder_input_output = zip(
            [self.input_size] + encoder_layers[:-1], encoder_layers
        )
        decoder_input_output = zip(decoder_layers, decoder_layers[1:])
        self.encoder = nn.Sequential(
            *[
                self.get_single_module(input_size, output_size)
                for input_size, output_size in encoder_input_output
            ]
        )
        self.decoder = nn.Sequential(
            *[
                self.get_single_module(input_size, output_size)
                for input_size, output_size in decoder_input_output
            ]
            + [
                self.get_single_module(
                    decoder_layers[-1], self.input_size, last_layer=True
                )
            ]
        )

    def get_single_module(
        self, input_size: int, output_size: int, last_layer: bool = False
    ):
        if last_layer:
            return nn.Linear(input_size, output_size)
        else:
            return nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size, momentum=0.01, eps=1e-03),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # [batch_size, 313, 128]
        z = slice_signal(x, self.window_size, self.stride)
        # [batch_size, 309, 5, 128]
        z = nn.Flatten(0, 1)(z)
        # [batch_size * 309, 5, 128]
        z = nn.Flatten(-2, -1)(z)
        # [batch_size * 309, 5 * 128]
        z = self.encoder(z)
        z = self.decoder(z)
        z = reconstruct_signal(z, batch_size, self.window_size)
        return z
