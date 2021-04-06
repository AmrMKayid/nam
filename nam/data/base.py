from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from .utils import read_dataset
from .utils import transform_data


class NAMDataset(torch.utils.data.Dataset):

  @classmethod
  def from_df(cls,
              config,
              file_path: Union[str, pd.DataFrame],
              features_columns: list,
              targets_column: str,
              weights_column: str = None):

    ## Read and handle NANs in the dataframe
    data = read_dataset(file_path)

    features = data[features_columns].copy()
    targets = data[targets_column].copy()
    weights = None
    if weights_column is not None:
      wgts = data[weights_column].copy().to_numpy()
      weights = torch.tensor(wgts).float()
      weights = ((weights - torch.min(weights)) /
                 (torch.max(weights) - torch.min(weights)))

    features, features_names = transform_data(features)
    features = torch.tensor(features)

    if (not config.regression) and (not isinstance(targets, np.ndarray)):
      targets = pd.get_dummies(targets).values
      targets = torch.tensor(np.argmax(targets, axis=-1)).long()
    else:
      targets = torch.tensor(targets.to_numpy().astype('float32'))

    dataset = cls(config=config,
                  features=features,
                  targets=targets,
                  weights=weights)
    dataset.data = data
    dataset.features_names = features_names
    dataset.targets_column = targets_column
    dataset.weights_column = weights_column

    return dataset

  @classmethod
  def from_X_y(cls,
               config,
               X: Union[str, pd.DataFrame],
               y: list,
               wgts: Optional = None):

    features = torch.tensor(X).float()
    targets = torch.tensor(y)
    targets = targets.float() if config.regression else targets.long()
    weights = None
    if wgts:
      weights = torch.tensor(wgts)
      weights = ((weights - torch.min(weights)) /
                 (torch.max(weights) - torch.min(weights)))

    dataset = cls(config=config,
                  features=features,
                  targets=targets,
                  weights=weights)

    return dataset

  def __init__(self,
               *,
               config,
               features: torch.Tensor,
               targets: torch.Tensor,
               weights: Optional[torch.Tensor] = None) -> None:
    """Custom dataset for csv files.

    Args:
        config ([type]): [description]
        file_path (str): [description]
        features_columns (list): [description]
        targets_column (str): [description]
        weights_column (str, optional): [description]. Defaults to None.
        transforms (Callable, optional): [description]. Defaults to None.
    """
    self._config = config
    self.features = features
    self.targets = targets
    self.weights = weights

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
    if self.weights is not None:
      return self.features[idx], self.targets[idx], self.weights[idx]

    return self.features[idx], self.targets[idx]

  def get_dataloaders(self,
                      val_split: float = 0.1,
                      test_split: float = 0.2) -> Tuple[DataLoader, ...]:
    test_size = int(test_split * len(self))
    val_size = int(val_split * (len(self) - test_size))
    train_size = len(self) - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        self, [train_size, val_size, test_size])

    train_dl = DataLoader(train_subset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.num_workers,
                          pin_memory=False)

    val_dl = DataLoader(val_subset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        pin_memory=False)

    test_dl = DataLoader(test_subset,
                         batch_size=self.config.batch_size,
                         shuffle=False,
                         num_workers=self.config.num_workers,
                         pin_memory=False)

    return train_dl, val_dl, test_dl

  @property
  def config(self):
    return self._config

  def __repr__(self):
    if self.weights is not None:
      return (
          f'NAMDatasetSample(\n\tfeatures={self.features[np.random.randint(len(self))]}, '
          + f'\n\ttargets={self.targets[np.random.randint(len(self))]}, ' +
          f'\n\tweights={self.weights[np.random.randint(len(self))]}\n)')

    return (
        f'NAMDatasetSample(features={self.features[np.random.randint(len(self))]}, '
        + f'targets={self.targets[np.random.randint(len(self))]})')
