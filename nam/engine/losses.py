from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.types import Config


def cross_entropy_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:  # https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
  """Cross entropy loss for binary classification.

  Args:
    model: Neural network model (NAM/DNN).
    inputs: Input values to be fed into the model for computing predictions.
    targets: Binary class labels.

  Returns:
    Cross-entropy loss between model predictions and the targets.
  """
  predictions = F.sigmoid(model(inputs))
  logits = torch.stack([predictions, torch.zeros_like(predictions)], dim=1)
  labels = torch.stack([targets, 1 - targets], dim=1)
  loss = torch.sum(-labels * F.log_softmax(logits, -1), -1)
  return loss.mean()


def mse_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
  """Mean squared error loss for regression."""
  predicted = model(inputs)
  # print(predicted, targets)
  return F.mse_loss(predicted, targets)


def feature_output_regularization(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
  """Penalizes the L2 norm of the prediction of each feature net."""
  per_feature_outputs = model.calc_outputs(inputs)
  per_feature_norm = [  # L2 Regularization
      torch.square(outputs).mean() for outputs in per_feature_outputs
  ]
  return torch.sum(per_feature_norm) / len(per_feature_norm)


def weight_decay(
    model: nn.Module,
    num_networks: int = 1,
) -> torch.Tensor:
  """Penalizes the L2 norm of weights in each feature net."""
  l2_losses = [F.mse_loss(x) for x in model.parameters()]
  return torch.sum(l2_losses) / num_networks


def penalized_loss(
    config: Config,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
  """Computes penalized loss with L2 regularization and output penalty.

  Args:
    config: Global config.
    model: Neural network model.
    inputs: Input values to be fed into the model for computing predictions.
    targets: Target values containing either real values or binary labels.

  Returns:
    The penalized loss.
  """
  loss_func = mse_loss if config.regression else cross_entropy_loss
  loss = loss_func(model, inputs, targets)

  reg_loss = 0.0
  if config.output_regularization > 0:
    reg_loss += config.output_regularization * \
                feature_output_regularization(model, inputs)

  if config.l2_regularization > 0:
    num_networks = 1 if config.use_dnn else len(model.feature_nns)
    reg_loss += config.l2_regularization * \
                  weight_decay(model,num_networks=num_networks)

  return loss + reg_loss
