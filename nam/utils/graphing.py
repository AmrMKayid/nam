from typing import Sequence
from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_feature_contributions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
) -> Sequence[torch.Tensor]:

    feature_contributions = []
    unique_features = dataset.unique_features
    for i, feature in enumerate(unique_features):
        feature = torch.tensor(feature).float().to(model.config.device)
        feat_contribution = model.feature_nns[i](feature).cpu().detach().numpy().squeeze()
        feature_contributions.append(feat_contribution)

    return feature_contributions


def calc_mean_prediction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
) -> Tuple[np.ndarray, dict]:
    #@title Calculate the mean prediction

    feature_contributions = get_feature_contributions(model, dataset)
    avg_hist_data = {col: contributions for col, contributions in zip(dataset.features_names, feature_contributions)}
    all_indices, mean_pred = {}, {}

    for i, col in enumerate(dataset.features_names):
        feature_i = dataset.features[:, i].cpu()
        all_indices[col] = np.searchsorted(dataset.unique_features[i][:, 0], feature_i, 'left')

    for col in dataset.features_names:
        mean_pred[col] = np.mean([avg_hist_data[col]])  #[i] for i in all_indices[col]]) TODO: check the error here

    return mean_pred, avg_hist_data


def plot_mean_feature_importance(model: torch.nn.Module, dataset: torch.utils.data.Dataset, width=0.5):

    mean_pred, avg_hist_data = calc_mean_prediction(model, dataset)

    def compute_mean_feature_importance(mean_pred, avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            try:
                mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
            except:
                continue
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    ## TODO: rename x1 and x2
    x1, x2 = compute_mean_feature_importance(mean_pred, avg_hist_data)

    cols = dataset.features_names
    fig = plt.figure(figsize=(5, 5))
    ind = np.arange(len(x1))
    x1_indices = np.argsort(x2)

    cols_here = [cols[i] for i in x1_indices]
    x2_here = [x2[i] for i in x1_indices]

    plt.bar(ind, x2_here, width, label='NAMs')
    plt.xticks(ind + width / 2, cols_here, rotation=90, fontsize='large')
    plt.ylabel('Mean Absolute Score', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Overall Importance', fontsize='x-large')
    plt.show()

    return fig


def plot_nams(model: torch.nn.Module,
              dataset: torch.utils.data.Dataset,
              num_cols: int = 2,
              n_blocks: int = 20,
              color: list = [0.4, 0.5, 0.9],
              linewidth: float = 7.0,
              alpha: float = 1.0,
              feature_to_use: list = None):

    unique_features, single_features = dataset.ufo, dataset.single_features
    mean_pred, feat_data_contrib = calc_mean_prediction(model, dataset)

    num_rows = len(dataset.features[0]) // num_cols

    fig = plt.figure(num=None, figsize=(num_cols * 10, num_rows * 10), facecolor='w', edgecolor='k')
    fig.tight_layout(pad=7.0)

    feat_data_contrib_pairs = list(feat_data_contrib.items())
    feat_data_contrib_pairs.sort(key=lambda x: x[0])

    mean_pred_pairs = list(mean_pred.items())
    mean_pred_pairs.sort(key=lambda x: x[0])

    if feature_to_use:
        feat_data_contrib_pairs = [v for v in feat_data_contrib_pairs if v[0] in feature_to_use]

    min_y = np.min([np.min(a[1]) for a in feat_data_contrib_pairs])
    max_y = np.max([np.max(a[1]) for a in feat_data_contrib_pairs])

    min_max_dif = max_y - min_y
    min_y = min_y - 0.1 * min_max_dif
    max_y = max_y + 0.1 * min_max_dif

    total_mean_bias = 0

    def shade_by_density_blocks(color: list = [0.9, 0.5, 0.5]):
        single_feature_data = single_features[name]
        x_n_blocks = min(n_blocks, len(unique_feat_data))

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

    for i, (name, feat_contrib) in enumerate(feat_data_contrib_pairs):
        mean_pred = mean_pred_pairs[i][1]
        total_mean_bias += mean_pred

        unique_feat_data = unique_features[name]
        ax = plt.subplot(num_rows, num_cols, i + 1)

        ## TODO: CATEGORICAL_NAMES if..else
        plt.plot(unique_feat_data, feat_contrib - mean_pred, color=color, linewidth=linewidth, alpha=alpha)

        plt.xticks(fontsize='x-large')

        plt.ylim(min_y, max_y)
        plt.yticks(fontsize='x-large')

        min_x = np.min(unique_feat_data)  # - 0.5  ## for categorical
        max_x = np.max(unique_feat_data)  # + 0.5
        plt.xlim(min_x, max_x)

        shade_by_density_blocks()

        if i % num_cols == 0:
            plt.ylabel('Features Contribution', fontsize='x-large')

        plt.xlabel(name, fontsize='x-large')

    plt.show()

    return fig
