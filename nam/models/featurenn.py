import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.models.base import Model

from .activation import ExU
from .activation import LinReLU


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

    hidden_sizes = [self._num_units] + self.config.hidden_sizes

    layers = []

    ## First layer is ExU
    layers.append(ExU(in_features=input_shape, out_features=num_units))

    ## Hidden Layers
    for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
      layers.append(LinReLU(in_features, out_features))

    ## Last Linear Layer
    layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1))

    self.model = nn.ModuleList(layers)
    # self.apply(init_weights)

  def forward(self, inputs) -> torch.Tensor:
    """Computes FeatureNN output with either evaluation or training mode."""
    outputs = inputs
    for layer in self.model:
      outputs = F.dropout(layer(outputs), p=self.config.dropout)
    return outputs
