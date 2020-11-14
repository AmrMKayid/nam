import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable


def cross_entropy_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:  #https://gist.github.com/tejaskhot/cf3d087ce4708c422e68b3b747494b9f
  """Cross entropy loss for binary classification.

  Args:
    model: Neural network model (NAM/DNN).
    inputs: Input values to be fed into the model for computing predictions.
    targets: Binary class labels.

  Returns:
    Cross-entropy loss between model predictions and the targets.
  """
  predictions = model(inputs)
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
    loss_func: Callable,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_regularization: float,
    l2_regularization: float = 0.0,
    use_dnn: bool = False,
) -> torch.Tensor:
  """Computes penalized loss with L2 regularization and output penalty.

  Args:
    loss_func: Loss function.
    model: Neural network model.
    inputs: Input values to be fed into the model for computing predictions.
    targets: Target values containing either real values or binary labels.
    output_regularization: Coefficient for feature output penalty.
    l2_regularization: Coefficient for L2 regularization.
    use_dnn: Whether using DNN or not when computing L2 regularization.

  Returns:
    The penalized loss.
  """
  loss = loss_func(model, inputs, targets)
  reg_loss = 0.0
  if output_regularization > 0:
    reg_loss += output_regularization * feature_output_regularization(
        model, inputs)
  if l2_regularization > 0:
    num_networks = 1 if use_dnn else len(model.feature_nns)
    reg_loss += l2_regularization * weight_decay(
        model,
        num_networks=num_networks,
    )
  return loss + reg_loss


def penalized_cross_entropy_loss(model: nn.Module,
                                 inputs: torch.Tensor,
                                 targets: torch.Tensor,
                                 output_regularization: float,
                                 l2_regularization: float = 0.0,
                                 use_dnn: bool = False) -> torch.Tensor:
  """Cross entropy loss with L2 regularization and output penalty."""
  return penalized_loss(cross_entropy_loss, model, inputs, targets,
                        output_regularization, l2_regularization, use_dnn)


def penalized_mse_loss(model: nn.Module,
                       inputs: torch.Tensor,
                       targets: torch.Tensor,
                       output_regularization: float,
                       l2_regularization: float = 0.0,
                       use_dnn: bool = False) -> torch.Tensor:
  """Mean Squared Error with L2 regularization and output penalty."""
  return penalized_loss(mse_loss, model, inputs, targets, output_regularization,
                        l2_regularization, use_dnn)
