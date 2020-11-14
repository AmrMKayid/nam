import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable
import numpy as np
from sklearn import metrics as sk_metrics


def accuracy(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
  """Accuracy for a binary classification model."""
  pred = model(inputs)
  binary_pred = pred > 0
  correct = binary_pred == (targets > 0.5)
  return correct.mean().float()


def sigmoid(x):
  """Sigmoid function."""
  if isinstance(x, list):
    x = np.array(x)
  return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  """Root mean squared error between true and predicted values."""
  return float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))


def calculate_metric(y_true: torch.Tensor,
                     predictions: torch.Tensor,
                     regression: bool = True):
  """Calculates the evaluation metric."""
  if regression:
    return rmse(y_true, predictions)
  else:
    return sk_metrics.roc_auc_score(y_true, sigmoid(predictions))


def roc_auc_score(sess, y_true, pred_tensor, dataset_init_op):
  # TODO:
  pass
  """Calculates the ROC AUC score."""
  # Assumes that pred_tensor already applies the sigmoid transformation
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return sk_metrics.roc_auc_score(y_true, y_pred)


def rmse_loss(sess, y_true, pred_tensor, dataset_init_op):
  """Calculates the RMSE error."""
  # TODO:
  pass
  y_pred = generate_predictions(pred_tensor, dataset_init_op, sess)
  return rmse(y_true, y_pred)


def generate_predictions(pred_tensor, dataset_init_op, sess):
  # TODO:
  pass
