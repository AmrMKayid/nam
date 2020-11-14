import torch
import torch.nn as nn

from nam.models.base import Model

from .activation import ActivationLayer


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
      num_units: int,
      input_shape: int,
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
    super().__init__(config, name)
    self._num_units = num_units
    self._dropout = dropout
    self._feature_num = feature_num
    self._shallow = shallow
    self._activation = activation
    self._input_shape = input_shape

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=self._dropout)

    self.hidden_layers = [
        ActivationLayer(
            num_units=self._num_units,
            input_shape=self._input_shape,
            activation=self._activation,
            name='activation_layer_{}'.format(self._feature_num),
        )
    ]
    if not self._shallow:
      self._h1 = nn.Linear(self._input_shape, 64)
      self._h2 = nn.Linear(64, 32)
      self.hidden_layers += [self._h1, self._h2]

    self.linear = nn.Linear(32, 1)

  def forward(self, inputs) -> torch.Tensor:
    """Computes FeatureNN output with either evaluation or training mode."""
    for l in self.hidden_layers:
      inputs = self.dropout(self.relu(l(inputs)))
    inputs = torch.squeeze(self.linear(inputs), dim=1)
    return inputs
