import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.types import Config


def ce_loss(probs: torch.Tensor, targets: torch.Tensor,
            weights: torch.tensor) -> torch.Tensor:
  """Cross entropy loss for binary classification.

  Args:
    logits: NAM model outputs
    targets: Binary class labels.

  Returns:
    Binary Cross-entropy loss between model predictions and the targets.
  """
  # print(probs)
  # print(weights)
  # weighted_probs = probs * weights
  return F.cross_entropy(probs, targets.view(-1))


def mse_loss(logits: torch.Tensor, targets: torch.Tensor,
             weights: torch.tensor) -> torch.Tensor:
  """Mean squared error loss for regression."""
  return F.mse_loss(logits.view(-1), targets.view(-1))


def weight_decay(
    model: nn.Module,
    num_networks: int = 1,
) -> torch.Tensor:
  """Penalizes the L2 norm of weights in each feature net."""
  l2_losses = [F.mse_loss(x) for x in model.parameters()]
  return torch.sum(l2_losses) / num_networks


def penalized_loss(config: Config, logits: torch.Tensor, targets: torch.Tensor,
                   weights: torch.tensor,
                   fnn_out: torch.Tensor) -> torch.Tensor:
  """Computes penalized loss with L2 regularization and output penalty.

  Args:
    config: Global config.
    model: Neural network model.
    inputs: Input values to be fed into the model for computing predictions.
    targets: Target values containing either real values or binary labels.

  Returns:
    The penalized loss.
  """

  def features_loss(per_feature_outputs: torch.Tensor) -> torch.Tensor:
    """Penalizes the L2 norm of the prediction of each feature net."""
    return (config.output_regularization * (per_feature_outputs**2).sum() /
            per_feature_outputs.shape[1])

  loss_func = mse_loss if config.regression else ce_loss
  loss = loss_func(logits, targets, weights)

  return loss + features_loss(fnn_out)


# reg_loss = 0.0
# if config.output_regularization > 0:
#   reg_loss += config.output_regularization * \
#               features_loss(model, inputs)

# if config.l2_regularization > 0:
#   num_networks = 1 if config.use_dnn else len(model.feature_nns)
#   reg_loss += config.l2_regularization * \
#                 weight_decay(model,num_networks=num_networks)
