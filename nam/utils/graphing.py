from typing import Sequence
from typing import Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_unique_features(X_train: torch.Tensor) -> Tuple[torch.Tensor, ...]:
  unique_features = []
  features = []
  for feature_i in range(len(X_train[0])):
    features.append(X_train[:, feature_i])
    unique_features.append(torch.unique(features[-1]))
  return unique_features, features


def get_model_outputs(
    unique_features: torch.Tensor,
    model: torch.nn.Module,
) -> Sequence[torch.Tensor]:

  feature_contributions = []
  for feature_i in range(len(unique_features)):
    feature_contributions.append(
        dict(model.feature_nns.named_children())[f"FeatureNN_{feature_i}"](
            torch.unsqueeze(unique_features[feature_i], 1)))
  return feature_contributions


def shade_by_density_blocks(
    ax: matplotlib.axes.SubplotBase,
    single_feature_data: np.ndarray,
    unique_x_data: np.ndarray,
    n_blocks: int = 50,
    color: list = [0.9, 0.5, 0.5],
) -> None:
  min_y, max_y = ax.get_ylim()
  min_x = np.min(single_feature_data)
  max_x = np.max(single_feature_data)
  x_n_blocks = min(n_blocks, len(unique_x_data))
  segments = (max_x - min_x) / x_n_blocks
  density = np.histogram(single_feature_data, bins=x_n_blocks)
  normed_density = density[0] / np.max(density[0])
  rect_params = []
  for p in range(x_n_blocks):
    start_x = min_x + segments * p
    end_x = min_x + segments * (p + 1)
    d = min(1.0, 0.01 + normed_density[p])
    rect_params.append((d, start_x, end_x))

  for param in rect_params:
    alpha, start_x, end_x = param
    rect = patches.Rectangle(
        (start_x, min_y - 1),
        end_x - start_x,
        max_y - min_y + 1,
        linewidth=0.01,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(rect)


def plot_line(
    ax: matplotlib.axes.SubplotBase,
    unique_features: np.ndarray,
    feature_contributions: np.ndarray,
    alpha: float = 0.5,
    color_base: list = [0.3, 0.4, 0.9, 0.2],
) -> None:
  feature_contributions = feature_contributions - np.mean(feature_contributions)
  if len(unique_features) < 10:
    unique_features = np.round(unique_features, decimals=1)
    if len(unique_features) <= 2:
      step_loc = "mid"
    else:
      step_loc = "post"
    ax.step(
        unique_features,
        feature_contributions,
        color=color_base,
        alpha=alpha,
        where=step_loc,
    )
  else:
    ax.plot(
        unique_features,
        feature_contributions,
        color=color_base,
        alpha=alpha,
    )


def nam_plot(
    dataset: torch.Tensor,
    models: Sequence[torch.nn.Module],
    n_columns: int = 3,
) -> None:
  raw_features = dataset.features
  n_rows = int(np.ceil(len(raw_features[0]) / n_columns))
  unique_features, features = get_unique_features(raw_features)
  unique_model_outputs = [
      get_model_outputs(unique_features, model) for model in models
  ]

  fig, axs = plt.subplots(
      n_columns,
      n_rows,
      figsize=(5 * n_rows, 5 * n_columns),
  )
  fig.tight_layout(pad=5.0)

  for i in range(n_columns * n_rows):
    ax = axs.reshape(-1)[i]
    if i < len(unique_features):
      for m in range(len(unique_model_outputs)):
        plot_line(
            ax,
            unique_features[i].detach().numpy(),
            unique_model_outputs[m][i].detach().numpy(),
            alpha=4 / len(models),
        )
      shade_by_density_blocks(
          ax,
          features[i].detach().numpy(),
          unique_features[i].detach().numpy(),
      )
      ax.set_xlabel(dataset.features_names[i])
      ax.set_ylabel(dataset.targets_column)
    else:
      ax.set_visible(False)
