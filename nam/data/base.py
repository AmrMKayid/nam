from typing import Callable
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch

from .utils import read_dataset
from .utils import transform_data


class NAMDataset(torch.utils.data.Dataset):

  def __init__(
      self,
      *,
      config,
      file_path: Union[str, pd.DataFrame],
      features_columns: list,
      targets_column: str,
      weights_column: str = None,
      transforms: Callable = None,
  ) -> None:
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
    self.features_columns = features_columns
    self.targets_column = targets_column
    self.weights_column = weights_column

    ## Read and handle NANs in the dataframe
    self.data = read_dataset(file_path)

    self.features = self.data[features_columns].copy()
    self.targets = self.data[targets_column].copy()
    if weights_column is not None:
      self.weights = torch.tensor(
          self.data[weights_column].copy().to_numpy()).float()

    self.features, self.column_names = transform_data(self.features)
    self.features = torch.tensor(self.features)

    if (not config.regression) and (not isinstance(self.targets, np.ndarray)):
      self.targets = pd.get_dummies(self.targets).values
      self.targets = torch.tensor(
          np.argmax(self.targets, axis=-1).astype('float32'))
    else:
      self.targets = torch.tensor(self.targets.to_numpy().astype('float32'))

    self.transforms = transforms

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
    if self.weights_column is not None:
      return self.features[idx], self.targets[idx], self.weights[idx]

    return self.features[idx], self.targets[idx]

  @property
  def config(self):
    return self._config

  def __repr__(self):
    if self.weights_column is not None:
      return (
          f'NAMDatasetSample(\n\tfeatures={self.features[np.random.randint(len(self))]}, '
          + f'\n\ttargets={self.targets[np.random.randint(len(self))]}, ' +
          f'\n\tweights={self.weights[np.random.randint(len(self))]}\n)')

    return (
        f'NAMDatasetSample(features={self.features[np.random.randint(len(self))]}, '
        + f'targets={self.targets[np.random.randint(len(self))]})')
