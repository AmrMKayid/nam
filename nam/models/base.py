import torch


class Model(torch.nn.Module):

  def __init__(self, config, name):
    super(Model, self).__init__()
    self._config = config
    self._name = name

  def forward(self):
    raise NotImplementedError

  @property
  def config(self):
    return self._config
