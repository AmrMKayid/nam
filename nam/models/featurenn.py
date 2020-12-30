import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.models.base import Model

from .activation import ExU, LinReLU


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

    hidden_sizes = [self._input_shape, self._num_units
                   ] + self.config.hidden_sizes + [1]

    layers = []
    layer = ExU if self.config.activation.lower() == 'exu' else LinReLU
    for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
      layers.append(layer(in_features=in_features, out_features=out_features))

    self.model = nn.Sequential(*layers)

  def forward(self, inputs) -> torch.Tensor:
    """Computes FeatureNN output with either evaluation or training mode."""
    for layer in self.model:
      inputs = F.dropout(layer(inputs), p=self.config.dropout)
    # outputs = torch.squeeze(inputs, dim=1)
    outputs = inputs
    return outputs
