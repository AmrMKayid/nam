import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.types import Config


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    """Cross entropy loss for binary classification.

    Args:
      logits: NAM model outputs
      targets: Binary class labels.

    Returns:
      Binary Cross-entropy loss between model predictions and the targets.
    """
    return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))


def mse_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    """Mean squared error loss for regression."""
    return F.mse_loss(logits.view(-1), targets.view(-1))


def penalized_loss(config: Config, logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor,
                   fnn_out: torch.Tensor, model: nn.Module) -> torch.Tensor:
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
        return (config.output_regularization * (per_feature_outputs**2).sum() / len(per_feature_outputs))

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = 1 if config.use_dnn else len(model.feature_nns)
        l2_losses = torch.as_tensor([((x**2).sum() / 2) for x in model.parameters()])
        return torch.sum(l2_losses) / num_networks

    loss_func = mse_loss if config.regression else bce_loss
    loss = loss_func(logits, targets, weights)

    reg_loss = 0.0
    if config.output_regularization > 0:
        reg_loss += features_loss(fnn_out)

    if config.l2_regularization > 0:
        reg_loss += config.l2_regularization * weight_decay(model)

    return loss + reg_loss
