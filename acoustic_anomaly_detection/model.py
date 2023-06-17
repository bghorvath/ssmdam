import torch
from torch import nn
import lightning.pytorch as pl
from torcheval.metrics.functional import binary_auroc, binary_auprc

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.save_hyperparameters()
        self.flatten = nn.Flatten()
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

    def forward(self, x):
        tensors = self.flatten(x)
        tensors = self.encoder(tensors)
        tensors = self.decoder(tensors)
        return tensors.view(x.shape)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_start(self) -> None:
        self.error_score = torch.tensor([])
        self.y = torch.tensor([])

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        error_score = torch.mean(torch.square(x_hat - x), dim=tuple(range(1, x_hat.ndim)))
        self.error_score = torch.cat([self.error_score, error_score])
        self.y = torch.cat([self.y, y])

    def on_test_epoch_end(self):
        auroc = binary_auroc(self.error_score, self.y)
        auprc = binary_auprc(self.error_score, self.y)
        self.log('auroc_epoch', auroc, prog_bar=True, logger=True)
        self.log('auprc_epoch', auprc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
