from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_precision,
    binary_recall,
    binary_f1_score,
)
import lightning.pytorch as pl
from transformers import ASTModel
from acoustic_anomaly_detection.utils import (
    slice_signal,
    reconstruct_signal,
    get_params,
)


def get_model(model: str, input_size: int, lr: float = 1e-3) -> pl.LightningModule:
    model_cls = {
        "ae": AutoEncoder,
    }[model]
    return model_cls(input_size, lr)


class Model(pl.LightningModule):
    def __init__(self, input_size: int, lr: float) -> None:
        super().__init__()

        params = get_params()
        self.model = params["model"]["name"]
        self.layers = params["model"]["layers"]
        self.max_fpr = params["classification"]["max_fpr"]
        self.decision_threshold = params["classification"]["decision_threshold"]
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
        label = attributes["label"][0]
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
        error_score = self.calculate_error_score(x, x_hat, attributes)
        self.error_scores[machine_type].append(error_score.item())
        self.ys[machine_type].append(y)
        self.domains[machine_type].append(domain)

    def on_test_epoch_start(self) -> None:
        self.error_scores = defaultdict(list)
        self.ys = defaultdict(list)
        self.domains = defaultdict(list)
        self.performance_metrics = {}

    def on_test_epoch_end(self) -> None:
        for machine_type, error_score_list in self.error_scores.items():
            error_score_list = torch.tensor(error_score_list)
            y_list = self.ys[machine_type]
            y_list = torch.tensor(y_list)
            domain_dict = {"source": 0, "target": 1}
            domain_list = torch.tensor(
                [domain_dict[domain] for domain in self.domains[machine_type]]
            )

            # Calculate metrics for source and target domains combined
            auc, p_auc, prec, recall, f1 = self.calculate_metrics(
                error_score_list, y_list, self.max_fpr, self.decision_threshold
            )

            self.log(f"{machine_type}_auc_epoch", auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_p_auc_epoch", p_auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_prec_epoch", prec, prog_bar=True, logger=True)
            self.log(f"{machine_type}_recall_epoch", recall, prog_bar=True, logger=True)
            self.log(f"{machine_type}_f1_epoch", f1, prog_bar=True, logger=True)

            machine_metrics = [auc, p_auc, prec, recall, f1]

            # Calculate metrics for source and target domains separately
            for domain in ("source", "target"):
                y_true_auc = y_list[
                    (domain_list == domain_dict[domain]) | (y_list == 1)
                ]
                y_pred_auc = error_score_list[
                    (domain_list == domain_dict[domain]) | (y_list == 1)
                ]
                y_true = y_list[domain_list == domain_dict[domain]]
                y_pred = error_score_list[domain_list == domain_dict[domain]]

                auc, p_auc, prec, recall, f1 = self.calculate_metrics(
                    y_pred_auc, y_true_auc, self.max_fpr, self.decision_threshold
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

    @staticmethod
    def calculate_metrics(
        error_score_list: torch.Tensor,
        y_list: torch.Tensor,
        max_fpr: float,
        decision_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        auc = binary_auroc(error_score_list, y_list)
        p_auc = binary_auroc(error_score_list, y_list, max_fpr=max_fpr)
        prec = binary_precision(error_score_list, y_list, threshold=decision_threshold)
        recall = binary_recall(error_score_list, y_list, threshold=decision_threshold)
        f1 = binary_f1_score(error_score_list, y_list, threshold=decision_threshold)
        return auc, p_auc, prec, recall, f1

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

    def calculate_error_score(
        self, x: torch.Tensor, x_hat: torch.Tensor, attributes: dict
    ) -> torch.Tensor:
        return self.calculate_loss(x, x_hat, attributes).mean()

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
