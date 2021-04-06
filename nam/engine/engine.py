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

    self.criterion = lambda inputs, targets, weights, fnns_out: penalized_loss(
        self.config, inputs, targets, weights, fnns_out)

    self.metrics = pl.metrics.MeanAbsoluteError() if config.regression \
                    else pl.metrics.Accuracy()

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
    weights = weights.pop() if weights else torch.tensor(1)

    logits, fnns_out = self.model(inputs, weights)
    loss = self.criterion(logits, targets, weights, fnns_out)

    self.log('train_loss', loss, on_step=True, \
              on_epoch=True, prog_bar=True, logger=True)

    self.log('train_acc_step',
             self.metrics(logits, targets),
             on_step=True,
             prog_bar=True,
             logger=True,
             on_epoch=False)

    return loss  #{'training_loss': loss}

  def training_epoch_end(self, outs):
    # log epoch metric
    self.log('train_acc_epoch', self.metrics.compute(), prog_bar=True)

  def validation_step(self, batch, batch_idx):
    inputs, targets, *weights = batch
    weights = weights.pop() if weights else torch.tensor(1)

    logits, fnns_out = self.model(inputs, weights)
    loss = self.criterion(logits, targets, weights, fnns_out)

    self.log('val_loss', loss, on_step=True, \
              on_epoch=True, prog_bar=True, logger=True)

    self.log('valid_acc',
             self.metrics(logits, targets),
             on_step=True,
             prog_bar=True,
             logger=True,
             on_epoch=True)

    return {'val_loss': loss}

  def test_step(self, batch, batch_idx):
    inputs, targets, *weights = batch

    logits, fnns_out = self.model(inputs)
    loss = self.criterion(logits, targets, weights, fnns_out)

    self.log('test_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    self.log('test_acc',
             self.metrics(logits, targets),
             on_step=True,
             prog_bar=True,
             logger=True,
             on_epoch=True)

    return {'test_loss': loss}
