from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from nam.types import DataType

## TODO(amr): Target columns, weight columns, features columns


def preprocess_df(data: pd.DataFrame) -> pd.DataFrame:
  x_train, y_train = [], []
  for index, row in data.iterrows():
    x_train.append(torch.Tensor(row.iloc[:-1]))
    y_train.append(row.iloc[-1])

  x_train = torch.stack(x_train)
  y_train = torch.LongTensor(y_train)

  data['x_train'], data['y_train'] = x_train, y_train
  return data


class NAMDataset(torch.utils.data.Dataset):

  def __init__(
      self,
      *,
      config,
      csv_file: str,
      features_columns: list,
      targets_column: str,
      weights_column: str = None,
      header: str = 'infer',
      names: list = None,
      delim_whitespace: bool = False,
      preprocess_fn: Callable = None,
      transforms: Callable = None,
  ) -> None:
    """Custom dataset for csv files.

    Args:
        config ([type]): [description]
        csv_file (str): [description]
        features_columns (list): [description]
        targets_column (str): [description]
        weights_column (str, optional): [description]. Defaults to None.
        header (str, optional): [description]. Defaults to 'infer'.
        names (list, optional): [description]. Defaults to None.
        delim_whitespace (bool, optional): [description]. Defaults to False.
        preprocess_fn (Callable, optional): [description]. Defaults to None.
        transforms (Callable, optional): [description]. Defaults to None.
    """
    self._config = config
    self.features_columns = features_columns
    self.targets_column = targets_column
    self.weights_column = weights_column

    if isinstance(csv_file, str):
      self.data = pd.read_csv(
          csv_file,
          header=header,
          names=names,
          delim_whitespace=delim_whitespace,
      )
    else:
      self.data = csv_file

    if preprocess_fn is not None:
      self.data = preprocess_fn(self.data)

    self.features = torch.tensor(self.data[features_columns].copy().to_numpy())
    self.targets = torch.tensor(self.data[targets_column].copy().to_numpy())
    if weights_column is not None:
      self.weights = torch.tensor(self.data[weights_column].copy().to_numpy())

    self.transforms = transforms

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx: int) -> DataType:
    if self.weights_column is not None:
      return self.features[idx], self.weights[idx], self.targets[idx]

    return self.features[idx], self.targets[idx]

  def data_loaders(
      self,
      n_splits: int = 5,
      batch_size: int = 32,
      shuffle: bool = True,
      stratified: bool = True,
      random_state: int = 42,
  ) -> Tuple[torch.utils.data.DataLoader, ...]:

    if stratified:
      kf = StratifiedKFold(
          n_splits=n_splits,
          shuffle=shuffle,
          random_state=random_state,
      )
    else:
      kf = KFold(
          n_splits=n_splits,
          shuffle=shuffle,
          random_state=random_state,
      )

    for i, (train_index,
            test_index) in enumerate(kf.split(self.features, self.targets)):

      train = torch.utils.data.Subset(self, train_index)
      test = torch.utils.data.Subset(self, test_index)

      trainloader = torch.utils.data.DataLoader(
          train,
          batch_size=self.config.batch_size,
          shuffle=shuffle,
          num_workers=0,
          pin_memory=False,
      )
      testloader = torch.utils.data.DataLoader(
          test,
          batch_size=self.config.batch_size,
          shuffle=shuffle,
          num_workers=0,
          pin_memory=False,
      )

      print(
          f'Fold({i + 1,}), train: {len(trainloader.dataset)}, test: {len(testloader.dataset)}'
      )

      yield trainloader, testloader

  @property
  def config(self):
    return self._config
