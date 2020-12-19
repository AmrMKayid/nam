import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

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

    self.loss = lambda inputs, targets: penalized_loss(self.config, self.model,
                                                       inputs, targets)

    self.save_hyperparameters(vars(self.config))

  def forward(
      self,
      inputs: torch.Tensor,
  ) -> torch.Tensor:
    outputs = self.model(inputs)
    return outputs

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.config.lr,
        weight_decay=self.config.decay_rate,
    )
    return optimizer

  def training_step(self, batch, batch_idx):
    if len(batch) > 2:
      inputs, targets, weights = batch
    else:
      inputs, targets = batch

    inputs = inputs.view(inputs.size(0), -1)
    # predictions = self.model(inputs)
    loss = self.loss(inputs, targets.squeeze())

    self.log('train_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return {
        'loss': loss,
    }

  def validation_step(self, batch, batch_idx):
    if len(batch) > 2:
      inputs, targets, weights = batch
    else:
      inputs, targets = batch

    inputs = inputs.view(inputs.size(0), -1)
    # predictions = self.model(inputs)
    loss = self.loss(inputs, targets.squeeze())

    self.log('val_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return loss

  def test_step(self, batch, batch_idx):
    if len(batch) > 2:
      inputs, targets, weights = batch
    else:
      inputs, targets = batch

    inputs = inputs.view(inputs.size(0), -1)
    # predictions = self.model(inputs)
    loss = self.loss(inputs, targets.squeeze())

    self.log('test_loss',
             loss,
             on_step=True,
             on_epoch=True,
             prog_bar=True,
             logger=True)

    return loss
