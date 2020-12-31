import torch
import pytorch_lightning as pl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def get_unique_features(X_train):
    unique_features = []
    features = []
    for feature_i in range(len(X_train[0])):
        features.append(X_train[:, feature_i])
        unique_features.append(torch.unique(features[-1]))
    return unique_features, features


def get_model_outputs(unique_features, model):
    feature_contributions = []
    for feature_i in range(len(unique_features[0])):
        feature_contributions.append(
            dict(model.feature_nns.named_children())[f"FeatureNN_{feature_i}"](torch.unsqueeze(
                unique_features[feature_i], 1)))
    return unique_features


def shade_by_density_blocks(ax, single_feature_data, unique_x_data, n_blocks=50, color=[0.9, 0.5, 0.5]):
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
        rect = patches.Rectangle((start_x, min_y - 1),
                                 end_x - start_x,
                                 max_y - min_y + 1,
                                 linewidth=0.01,
                                 edgecolor=color,
                                 facecolor=color,
                                 alpha=alpha)
        ax.add_patch(rect)


def plot_line(ax, unique_features, feature_contributions, alpha=1.0, color_base=[0.3, 0.4, 0.9, 0.2]):
    if len(unique_features) < 10:
        unique_features = np.round(unique_features, decimals=1)
        if len(unique_features) <= 2:
            step_loc = "mid"
        else:
            step_loc = "post"
        ax.step(unique_features, feature_contributions, color=color_base, where=step_loc, alpha=alpha)
    else:
        ax.plot(unique_features, feature_contributions, alpha=alpha)


def nam_plot(X_train, models, n_columns=3):
    n_columns = 3
    n_rows = int(np.ceil(len(X_train[0]) / n_columns))
    unique_features, features = get_unique_features(X_train)
    unique_model_outputs = [get_model_outputs(unique_features, model) for model in models]
    fig, axs = plt.subplots(n_columns, n_rows, figsize=(3 * n_rows, 3 * n_columns))
    for i in range(n_columns * n_rows):
        ax = axs.reshape(-1)[i]
        if i < len(unique_features):
            for m in range(len(unique_model_outputs))
                plot_line(ax, unique_features[i].detach().numpy(), unique_model_outputs[m][i].detach().numpy())
            shade_by_density_blocks(ax, features[i].detach().numpy(), unique_features[i].detach().numpy())
        else:
            ax.set_visible(False)