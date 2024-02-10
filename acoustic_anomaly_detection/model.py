from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import lightning.pytorch as pl
from acoustic_anomaly_detection.utils import (
    slice_signal,
    reconstruct_signal,
    load_params,
    calculate_metrics,
    min_max_scaler,
)
from acoustic_anomaly_detection.loss import SCAdaCos


def get_model(model: str, stage: str, input_size: int) -> pl.LightningModule:
    model_cls = {
        "ae": AutoEncoder,
        "ssmdam": SSMDAM,
    }[model]
    return model_cls(stage, input_size)


class Model(pl.LightningModule):
    def __init__(self, stage: str, input_size: int) -> None:
        super().__init__()

        params = load_params()
        self.model = params["model"]["name"]
        self.layers = params["model"]["layers"]
        self.max_fpr = params["classification"]["max_fpr"]
        self.decision_threshold = params["classification"]["decision_threshold"]
        self.anomaly_score = params["classification"]["anomaly_score"]
        self.mix_machine_types = params["data"]["mix_machine_types"]
        self.window_size = params["data"]["transform"]["window_size"]
        self.stride = params["data"]["transform"]["stride"]
        self.input_size = input_size
        self.stage = stage
        if stage in ("train", "finetune"):
            self.lr = params[stage]["lr"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, attributes = batch
        machine_types = attributes["machine_type"]
        loss = self(batch)

        if self.mix_machine_types:
            for i, machine_type in enumerate(machine_types):
                self.train_error_scores[machine_type] += loss[i]
        else:
            self.train_error_scores[machine_types[0]] += loss.mean()

        return loss.mean()

    def on_train_epoch_start(self) -> None:
        self.train_error_scores = defaultdict(
            lambda: torch.tensor(0.0, device=self.device)
        )

        # Reshuffle the batches of the training dataloader
        if self.current_epoch > 0:
            self.trainer.datamodule.reshuffle_train_batches()

    def on_train_epoch_end(self) -> None:
        for machine_type, error_score in self.train_error_scores.items():
            mean_error_score = error_score.mean().item()

            self.log(
                f"{machine_type}_train_loss_epoch",
                mean_error_score,
                prog_bar=True,
                logger=True,
            )

        if self.stage == "train":
            train_loss_epoch = list(self.train_error_scores.values())
            train_loss_epoch = torch.stack(train_loss_epoch).mean().item()

            self.log(
                "train_loss_epoch",
                train_loss_epoch,
                prog_bar=True,
                logger=True,
            )

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, dict[str, str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, attributes = batch
        machine_type = attributes["machine_type"][0]
        # label = attributes["label"][0]
        loss = self(batch).mean()

        self.val_error_scores[machine_type] += loss

        return loss

    def on_validation_epoch_start(self) -> None:  # TODO: Remove or fix
        self.val_error_scores = defaultdict(
            lambda: torch.tensor(0.0, device=self.device)
        )

    def on_validation_epoch_end(self) -> None:
        for machine_type, error_score in self.val_error_scores.items():
            mean_error_score = error_score.mean().item()

            self.log(
                f"{machine_type}_val_loss_epoch",
                mean_error_score,
                prog_bar=True,
                logger=True,
            )
        if self.stage == "train":
            val_loss_epoch = list(self.val_error_scores.values())
            val_loss_epoch = torch.stack(val_loss_epoch).mean().item()

            self.log(
                "val_loss_epoch",
                val_loss_epoch,
                prog_bar=True,
                logger=True,
            )

    def test_step(
        self, batch: tuple[torch.Tensor, dict[str, str]], batch_idx: int
    ) -> None:
        x, y, attributes = batch
        label = attributes["label"][0]
        machine_type = attributes["machine_type"][0]
        domain = attributes["domain"][0]
        y = 1 if label == "anomaly" else 0

        loss = self(batch).mean()

        self.test_loss[machine_type].append(loss.item())
        self.test_y_true[machine_type].append(y)
        self.test_domain[machine_type].append(domain)

    def on_test_epoch_start(self) -> None:
        self.test_loss = defaultdict(list)
        self.test_y_true = defaultdict(list)
        self.test_domain = defaultdict(list)
        self.roc = defaultdict(dict)
        self.performance_metrics = {}

    def on_test_epoch_end(self) -> None:
        for machine_type, loss in self.test_loss.items():
            loss = torch.tensor(loss)
            loss = min_max_scaler(loss)
            y_true = self.test_y_true[machine_type]
            y_true = torch.tensor(y_true)
            domain_dict = {"source": 0, "target": 1}
            domain_true = torch.tensor(
                [domain_dict[domain] for domain in self.test_domain[machine_type]]
            )
            # Calculate metrics for source and target domains combined
            auc, p_auc, prec, recall, f1, fpr, tpr = calculate_metrics(
                loss, y_true, self.max_fpr, self.decision_threshold
            )

            self.log(f"{machine_type}_auc_epoch", auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_p_auc_epoch", p_auc, prog_bar=True, logger=True)
            self.log(f"{machine_type}_prec_epoch", prec, prog_bar=True, logger=True)
            self.log(f"{machine_type}_recall_epoch", recall, prog_bar=True, logger=True)
            self.log(f"{machine_type}_f1_epoch", f1, prog_bar=True, logger=True)

            self.roc[machine_type]["total"] = (fpr, tpr)
            machine_metrics = [auc, p_auc, prec, recall, f1]

            # Calculate metrics for source and target domains separately
            for domain in ("source", "target"):
                y_true_domain_auc = y_true[
                    (domain_true == domain_dict[domain]) | (y_true == 1)
                ]
                y_pred_domain_auc = loss[
                    (domain_true == domain_dict[domain]) | (y_true == 1)
                ]
                # y_true_domain = y_true[domain_true == domain_dict[domain]]
                # y_pred_domain = error_score[domain_true == domain_dict[domain]]

                auc, p_auc, prec, recall, f1, fpr, tpr = calculate_metrics(
                    y_pred_domain_auc,
                    y_true_domain_auc,
                    self.max_fpr,
                    self.decision_threshold,
                )

                self.roc[machine_type][domain] = (fpr, tpr)

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )


class AutoEncoder(Model):
    """
    Baseline AE model
    Source: https://github.com/nttcslab/dcase2023_task2_baseline_ae/blob/main/networks/dcase2023t2_ae/network.py
    """

    def __init__(self, stage: str, input_size: int) -> None:
        super().__init__(stage, input_size)
        self.transform_func = MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0,
            f_max=8000,
            n_mels=128,
        )
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

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform_func(x)
        x = x.squeeze(0)
        x = AmplitudeToDB(stype="power")(x)
        x = x.transpose(0, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        batch_size = x.shape[0]
        # [batch_size, 313, 128]
        x_hat = slice_signal(x, self.window_size, self.stride)
        # [batch_size, 309, 5, 128]
        x_hat = nn.Flatten(0, 1)(x_hat)
        # [batch_size * 309, 5, 128]
        x_hat = nn.Flatten(-2, -1)(x_hat)
        # [batch_size * 309, 5 * 128]
        x_hat = self.encoder(x_hat)
        x_hat = self.decoder(x_hat)
        x_hat = reconstruct_signal(x_hat, batch_size, self.window_size)
        loss = mse_loss(x, x_hat, reduction="none").mean(dim=(1, 2))
        return loss


class SpectrogramResNet(nn.Module):
    def __init__(self, use_bias):
        super(SpectrogramResNet, self).__init__()
        self.use_bias = use_bias
        self.layer0 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=self.use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
            SqueezeExcitation(16, 16, dimension=2),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
            SqueezeExcitation(16, 16, dimension=2),
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.layer3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(32, 32, dimension=2),
        )
        self.layer3_1 = nn.Sequential(
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(16, 32, 1, padding=0, bias=self.use_bias),
        )
        self.bn3 = nn.BatchNorm2d(32)
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(32, 32, dimension=2),
        )
        self.bn4 = nn.BatchNorm2d(32)
        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(64, 64, dimension=2),
        )
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 1, padding=0, bias=self.use_bias),
        )
        self.bn5 = nn.BatchNorm2d(64)
        self.layer6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(64, 64, dimension=2),
        )
        self.bn6 = nn.BatchNorm2d(64)
        self.layer7 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(128, 128, dimension=2),
        )
        self.layer7_1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 1, padding=0, bias=self.use_bias),
        )
        self.bn7 = nn.BatchNorm2d(128)
        self.layer8 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=self.use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=self.use_bias),
            SqueezeExcitation(128, 128, dimension=2),
        )
        self.layer9 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 1, 0)),
            nn.MaxPool2d((18, 1), padding=0),
            nn.Flatten(),
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 128),
        )

    def forward(self, x):
        x = self.layer0(x)
        xr = self.layer1(x)
        x = x + xr
        x = self.bn1(x)
        xr = self.layer2(x)
        x = x + xr
        x = self.bn2(x)
        xr = self.layer3(x)
        x = self.layer3_1(x)
        x = x + xr
        x = self.bn3(x)
        xr = self.layer4(x)
        x = x + xr
        x = self.bn4(x)
        xr = self.layer5(x)
        x = self.layer5_1(x)
        x = x + xr
        x = self.bn5(x)
        xr = self.layer6(x)
        x = x + xr
        x = self.bn6(x)
        xr = self.layer7(x)
        x = self.layer7_1(x)
        x = x + xr
        x = self.bn7(x)
        xr = self.layer8(x)
        x = x + xr
        x = self.layer9(x)
        return x


class SpectraResNet(nn.Module):
    def __init__(self, use_bias):
        self.use_bias = use_bias
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 256, stride=64, padding=128, bias=self.use_bias)
        self.relu1 = nn.ReLU()
        self.se1 = SqueezeExcitation(128, 128)
        self.conv2 = nn.Conv1d(128, 128, 64, stride=32, padding=32, bias=self.use_bias)
        self.relu2 = nn.ReLU()
        self.se2 = SqueezeExcitation(128, 128)
        self.conv3 = nn.Conv1d(128, 128, 16, stride=4, padding=8, bias=self.use_bias)
        self.relu3 = nn.ReLU()
        self.se3 = SqueezeExcitation(128, 128)
        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(128 * 11, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.dense2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dense3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu6 = nn.ReLU()
        self.dense4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu7 = nn.ReLU()
        self.dense5 = nn.Linear(128, 128)

    def forward(self, x):
        x = torch.fft.fft(x)
        x = torch.sqrt(x.real**2 + x.imag**2)
        # Keep only the first half if the input is real
        x = x[..., : x.shape[-1] // 2]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.se3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu4(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu5(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.relu6(x)
        x = self.dense4(x)
        x = self.bn4(x)
        x = self.relu7(x)
        x = self.dense5(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, planes, ratio=16, dimension=1):
        super().__init__()
        self.dimension = dimension
        if self.dimension == 1:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        elif self.dimension == 2:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("dimension must be 1 or 2")
        self.fc1 = nn.Linear(inplanes, inplanes // ratio)
        self.fc2 = nn.Linear(inplanes // ratio, inplanes)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = torch.flatten(scale, 1)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        if self.dimension == 1:
            scale = scale.unsqueeze(-1)
        elif self.dimension == 2:
            scale = scale.unsqueeze(-1).unsqueeze(-1)
        return scale * input


class StatEx(nn.Module):
    def __init__(self, prob):
        super(StatEx, self).__init__()
        self.prob = prob

    def forward(self, inputs):
        x, y = inputs
        # mixup data
        original_data = x
        reversed_data = torch.flip(x, dims=[0])

        # mixup labels
        original_labels = torch.cat(
            [y, torch.zeros_like(y), torch.zeros_like(y)], dim=1
        )
        exchanged_labels = torch.cat(
            [torch.zeros_like(y), 0.5 * y, 0.5 * torch.flip(y, dims=[0])], dim=1
        )

        # statistics exchange data
        time_exchanged_data = (
            original_data - torch.mean(original_data, dim=2, keepdim=True)
        ) / (torch.std(original_data, dim=2, keepdim=True) + 1e-16) * torch.std(
            reversed_data, dim=2, keepdim=True
        ) + torch.mean(reversed_data, dim=2, keepdim=True)
        feature_exchanged_data = (
            original_data - torch.mean(original_data, dim=1, keepdim=True)
        ) / (torch.std(original_data, dim=1, keepdim=True) + 1e-16) * torch.std(
            reversed_data, dim=1, keepdim=True
        ) + torch.mean(reversed_data, dim=1, keepdim=True)

        # randomly decide on which statistics exchange axis to use
        decision = (torch.rand(size=[x.shape[0]], device=x.device) < 0).float()
        decision_reshaped = decision.view(-1, *([1] * (len(x.shape) - 1)))
        exchanged_data = (
            decision_reshaped * feature_exchanged_data
            + (1 - decision_reshaped) * time_exchanged_data
        )

        # apply mixup or not
        decision = (torch.rand(size=[x.shape[0]], device=x.device) < self.prob).float()
        decision_reshaped = decision.view(-1, *([1] * (len(x.shape) - 1)))
        output_data = (
            decision_reshaped * original_data + (1 - decision_reshaped) * exchanged_data
        )
        decision_reshaped_labels = decision.view(
            -1, *([1] * (len(original_labels.shape) - 1))
        )
        output_labels = (
            decision_reshaped_labels * original_labels
            + (1 - decision_reshaped_labels) * exchanged_labels
        )

        # pick output corresponding to training phase
        return (output_data, output_labels) if self.training else (x, original_labels)


class FeatEx(nn.Module):
    def __init__(self, prob):
        super(FeatEx, self).__init__()
        self.prob = prob

    def forward(self, inputs):
        sgram_input, spectra_input, label_input = inputs
        # mixup data
        original_data = sgram_input
        reversed_data = torch.flip(sgram_input, dims=[0])

        # mixup labels
        original_labels = torch.cat(
            [label_input, torch.zeros_like(label_input), torch.zeros_like(label_input)],
            dim=1,
        )
        mixed_labels = torch.cat(
            [
                torch.zeros_like(label_input),
                0.5 * label_input,
                0.5 * torch.flip(label_input, dims=[0]),
            ],
            dim=1,
        )

        # apply mixup or not
        mixup_decision = (
            torch.rand(size=[original_data.shape[0]], device=original_data.device)
            < self.prob
        ).float()
        mixup_decision_data = mixup_decision.view(
            [-1] + [1] * (len(original_data.shape) - 1)
        )
        mixed_data = (
            mixup_decision_data * original_data
            + (1 - mixup_decision_data) * reversed_data
        )
        mixup_decision_labels = mixup_decision.view(
            [-1] + [1] * (len(original_labels.shape) - 1)
        )
        mixed_labels = (
            mixup_decision_labels * original_labels
            + (1 - mixup_decision_labels) * mixed_labels
        )

        # pick output corresponding to training phase
        return (
            (mixed_data, spectra_input, mixed_labels)
            if self.training
            else (sgram_input, spectra_input, original_labels)
        )


class Mixup(nn.Module):
    def __init__(self, prob):
        super(Mixup, self).__init__()
        self.prob = prob

    def forward(self, inputs):
        x, y = inputs
        # get mixup weights
        mixup_weights = torch.rand(size=[x.shape[0]], device=x.device)
        l_data = mixup_weights.view([-1] + [1] * (len(x.shape) - 1))
        l_labels = mixup_weights.view([-1] + [1] * (len(y.shape) - 1))

        # mixup data
        original_data = x
        reversed_data = torch.flip(x, dims=[0])
        mixed_data = original_data * l_data + reversed_data * (1 - l_data)

        # mixup labels
        original_labels = y
        reversed_labels = torch.flip(y, dims=[0])
        mixed_labels = original_labels * l_labels + reversed_labels * (1 - l_labels)

        # apply mixup or not
        decision = (torch.rand(size=[x.shape[0]], device=x.device) < self.prob).float()
        decision_data = decision.view([-1] + [1] * (len(x.shape) - 1))
        output_data = decision_data * mixed_data + (1 - decision_data) * original_data
        decision_labels = decision.view([-1] + [1] * (len(y.shape) - 1))
        output_labels = (
            decision_labels * mixed_labels + (1 - decision_labels) * original_labels
        )

        # pick output corresponding to training phase
        return (output_data, output_labels) if self.training else (x, y)


class SSMDAM(Model):
    def __init__(self, stage: str, input_size: int):
        super().__init__(stage, input_size)
        self.n_classes = 135
        self.n_subclusters = 16
        self.use_bias = False
        self.mixup = Mixup(prob=0.5)
        self.statex = StatEx(prob=0.5)
        self.featex = FeatEx(prob=0.5)
        self.spectra_resnet = SpectraResNet(self.use_bias)
        self.spectrogram_resnet = SpectrogramResNet(self.use_bias)
        # self.mdam = MDAM()
        self.scadacos = SCAdaCos(
            n_classes=self.n_classes, n_subclusters=self.n_subclusters, trainable=False
        )
        self.scadacos_ssl = SCAdaCos(
            n_classes=self.n_classes * 3,
            n_subclusters=self.n_subclusters,
            trainable=True,
        )
        self.scadacos_ssl2 = SCAdaCos(
            n_classes=self.n_classes * 9,
            n_subclusters=self.n_subclusters,
            trainable=True,
        )

    def forward(self, inputs):
        x, y, attributes = inputs
        label_input = y
        x_mix, y_mix = self.mixup((x, y))
        emb_fft = self.spectra_resnet(x_mix)
        # x = self.spectrogram(x_mix)
        x = torch.stft(
            x_mix.squeeze(1), n_fft=1024, hop_length=512, return_complex=True
        )
        x = torch.sqrt(x.real**2 + x.imag**2)
        # x = x.unsqueeze(-1)
        x, y = self.statex((x, y_mix))
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x.unsqueeze(1)
        emb_mel = self.spectrogram_resnet(x)
        # emb_mel = self.mdam((emb_mel, y))
        emb_mel_ssl, emb_fft_ssl, y_ssl = self.featex((emb_mel, emb_fft, y))
        # x = torch.cat([emb_mel, emb_fft], dim=1)
        x_ssl = torch.cat([emb_mel_ssl, emb_fft_ssl], dim=1)
        loss_ssl2 = self.scadacos_ssl2((x_ssl, y_ssl, label_input))
        # loss = self.scadacos((x, y_mix, label_input))
        # loss_ssl = self.scadacos_ssl((x, y, label_input))
        return loss_ssl2
