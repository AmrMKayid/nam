import pytorch_lightning as pl
import torch
from torch import nn

from nam.trainer.losses import penalized_loss
from nam.trainer.metrics import accuracy
from nam.trainer.metrics import mae
from nam.types import Config


class LitNAM(pl.LightningModule):

    def __init__(self, config: Config, model: nn.Module) -> None:
        super().__init__()
        self.config = Config(**vars(config))  #config
        self.model = model

        self.criterion = lambda inputs, targets, weights, fnns_out, model: penalized_loss(
            self.config, inputs, targets, weights, fnns_out, model)

        self.metrics = lambda logits, targets: mae(logits, targets) if config.regression else accuracy(logits, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"

        self.save_hyperparameters(vars(self.config))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, fnns_out = self.model(inputs)
        return logits, fnns_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.decay_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch
        # weights = weights.pop() if weights else torch.tensor(1)

        logits, fnns_out = self.model(features)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)

        self.log_dict(
            {
                'train_loss': loss,
                f"{self.metrics_name}_metric": metric
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss  #{'training_loss': loss}

    def validation_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch
        # weights = weights.pop() if weights else torch.tensor(1)

        logits, fnns_out = self.model(features)  #, weights)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)

        self.log_dict(
            {
                'val_loss': loss,
                f"{self.metrics_name}_metric": metric
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        features, targets = batch
        # inputs, targets, *weights = batch

        logits, fnns_out = self.model(features)
        loss = self.criterion(logits, targets, None, fnns_out, self.model)
        metric = self.metrics(logits, targets)

        self.log_dict(
            {
                'test_loss': loss,
                f"{self.metrics_name}_metric": metric
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {'test_loss': loss}
