import pytorch_lightning as pl
import torch
from torch import nn

from nam.engine.losses import penalized_loss
from nam.types import Config


class Engine(pl.LightningModule):

  def __init__(
      self,
      config: Config,
      model: nn.Module,
  ) -> None:
    super().__init__()
    self.config = config
    self.model = model

    self.criterion = lambda inputs, targets, fnns_out: penalized_loss(
        self.config, inputs, targets, fnns_out)

    self.save_hyperparameters(vars(self.config))

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    logits, fnns_out = self.model(inputs)
    return logits, fnns_out

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(),
                                  lr=self.config.lr,
                                  weight_decay=self.config.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                gamma=0.995,
                                                step_size=1)
    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx):
    inputs, targets, *weights = batch

    logits, fnns_out = self.model(inputs)
    loss = self.criterion(logits, targets, fnns_out)

    self.log('train_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return loss  #{'training_loss': loss}

  def validation_step(self, batch, batch_idx):
    inputs, targets, *weights = batch

    logits, fnns_out = self.model(inputs)
    loss = self.criterion(logits, targets, fnns_out)

    self.log('val_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return {'val_loss': loss}

  def test_step(self, batch, batch_idx):
    inputs, targets, *weights = batch

    logits, fnns_out = self.model(inputs)
    loss = self.criterion(logits, targets, fnns_out)

    self.log('test_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return {'test_loss': loss}
