import torch
import torch.nn as nn

from nam.models.base import Model


def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.kaiming_normal_(m.weight)
    m.bias.data.fill_(0.01)


class DNN(Model):

  def __init__(
      self,
      config,
      name: str = "DNNModel",
      *,
      input_shape: int = 1,
      output_shape: int = 1,
  ) -> None:
    super(DNN, self).__init__(config, name)

    self.layers = []
    self.dropout = nn.Dropout(p=self.config.dropout)

    self.layers.append(nn.Linear(input_shape, 100, bias=True))
    self.layers.append(nn.ReLU())
    self.layers.append(self.dropout)

    for _ in range(9):
      self.layers.append(nn.Linear(100, 100, bias=True))
      self.layers.append(nn.ReLU())
      self.layers.append(self.dropout)

    self.layers.append(nn.Linear(100, output_shape, bias=True))

    self.model = nn.Sequential(*self.layers)
    self.apply(init_weights)

  def forward(self, inputs) -> torch.Tensor:
    return self.model(inputs)
