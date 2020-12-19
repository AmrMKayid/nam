import torch
import torch.nn as nn

from nam.models.base import Model

from .activation import ExU


class FeatureNN(Model):
  """Neural Network model for each individual feature."""

  def __init__(
      self,
      config,
      name,
      *,
      input_shape: int,
      num_units: int,
      feature_num: int = 0,
  ) -> None:
    """Initializes FeatureNN hyperparameters.

    Args:
      num_units: Number of hidden units in first hidden layer.
      dropout: Coefficient for dropout regularization.
      feature_num: Feature Index used for naming the hidden layers.
    """
    super(FeatureNN, self).__init__(config, name)
    self._input_shape = input_shape
    self._num_units = num_units
    self._feature_num = feature_num
    self._activation = self.config.activation

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=self.config.dropout)

    layers = [ExU(in_features=self._input_shape, out_features=self._num_units)]

    ## shallow: If True, then a shallow network with a single hidden layer is created,
    ## otherwise, a network with 3 hidden layers is created.
    if not self.config.shallow:
      h1 = nn.Linear(self._num_units, 64)
      h2 = nn.Linear(64, 32)
      linear = nn.Linear(32, 1)
      layers += [h1, h2, linear]
    else:
      linear = nn.Linear(self._num_units, 1)
      layers += [linear]

    self.model = nn.Sequential(*layers)

  def forward(self, inputs) -> torch.Tensor:
    """Computes FeatureNN output with either evaluation or training mode."""
    for layer in self.model:
      inputs = self.dropout(layer(inputs))
    outputs = torch.squeeze(inputs, dim=1)
    return outputs
