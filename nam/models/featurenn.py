import torch
import torch.nn as nn

from nam.models.base import Model

from .activation import ExU

## input : 1
## output: 1

## TODO(amr): naming layers accroding to feature_num


class FeatureNN(Model):
  """Neural Network model for each individual feature.

  Attributes:
    hidden_layers: A list containing hidden layers. The first layer is an
      `ActivationLayer` containing `num_units` neurons with specified
      `activation`. If `shallow` is False, then it additionally contains 2
      tf.keras.layers.Dense ReLU layers with 64, 32 hidden units respectively.
    linear: Fully connected layer.
  """

  def __init__(
      self,
      config,
      name,
      *,
      input_shape: int,
      num_units: int,
      dropout: float = 0.5,
      shallow: bool = True,
      feature_num: int = 0,
      activation: str = 'exu',
  ) -> None:
    """Initializes FeatureNN hyperparameters.

    Args:
      num_units: Number of hidden units in first hidden layer.
      dropout: Coefficient for dropout regularization.
      shallow: If True, then a shallow network with a single hidden layer is
        created, otherwise, a network with 3 hidden layers is created.
      feature_num: Feature Index used for naming the hidden layers.
      name_scope: TF name scope str for the model.
      activation: Activation and type of hidden unit(ExUs/Standard) used in the
        first hidden layer.
    """
    super(FeatureNN, self).__init__(config, name)
    self._num_units = num_units
    self._dropout = dropout
    self._feature_num = feature_num
    self._shallow = shallow
    self._activation = activation
    self._input_shape = input_shape

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=self._dropout)

    ## TODO(amr): self._input_shape[-1] or self._input_shape ??!
    self.layers = [ExU(in_dim=self._input_shape, out_dim=self._num_units)]
    if not self._shallow:
      self._h1 = nn.Linear(self._num_units, 64)
      self._h2 = nn.Linear(64, 32)
      self.layers += [self._h1, self._h2]
      self.linear = nn.Linear(32, 1)
    else:
      self.linear = nn.Linear(self._num_units, 1)

    self.hidden_layers = nn.Sequential(*self.layers)

  def forward(self, inputs) -> torch.Tensor:
    """Computes FeatureNN output with either evaluation or training mode."""
    for l in self.hidden_layers:
      inputs = self.dropout(l(inputs))
    inputs = torch.squeeze(self.linear(inputs), dim=1)
    return inputs
