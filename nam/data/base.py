from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from nam.types import DataType

## Label work for features only

## TODO(amr): Target columns, weight columns, features columns


def preprocess_df(data: pd.DataFrame) -> pd.DataFrame:
  """One Hot Encoding.

  Args:
      data (pd.DataFrame): unprocessed dataframe

  Returns:
      pd.DataFrame: processed dataframe with one hot encoded columns
  """
  ## Save label encoder (Mapping -> str to int)
  return data.apply(LabelEncoder().fit_transform)


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
      preprocess_fn: Callable = preprocess_df,
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

    self.train_subset, self.test_subset = self.get_train_test_fold()

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx: int) -> DataType:
    if self.weights_column is not None:
      return self.features[idx], self.weights[idx], self.targets[idx]

    return self.features[idx], self.targets[idx]

  def get_train_test_fold(
      self,
      fold_num: int = 1,
      num_folds: int = 5,
      shuffle: bool = True,
      stratified: bool = True,
      random_state: int = 42,
  ) -> Tuple[torch.utils.data.Subset, ...]:
    if stratified:
      kf = StratifiedKFold(
          n_splits=num_folds,
          shuffle=shuffle,
          random_state=random_state,
      )
    else:
      kf = KFold(
          n_splits=num_folds,
          shuffle=shuffle,
          random_state=random_state,
      )
    assert fold_num <= num_folds and fold_num > 0, 'Pass a valid fold number.'
    for train_index, test_index in kf.split(self.features, self.targets):
      if fold_num == 1:
        train = torch.utils.data.Subset(self, train_index)
        test = torch.utils.data.Subset(self, test_index)
        return train, test
      else:
        fold_num -= 1

  def data_loaders(
      self,
      n_splits: int = 5,
      batch_size: int = 32,
      test_size: int = 0.125,
      shuffle: bool = True,
      stratified: bool = True,
      random_state: int = 42,
  ) -> Tuple[torch.utils.data.DataLoader, ...]:

    if stratified:
      shuffle_split = StratifiedShuffleSplit(
          n_splits=n_splits,
          test_size=test_size,
          random_state=random_state,
      )
    else:
      shuffle_split = ShuffleSplit(
          n_splits=n_splits,
          test_size=test_size,
          random_state=random_state,
      )

    for i, (train_index, validation_index) in enumerate(
        shuffle_split.split(self.features[self.train_subset.indices],
                            self.targets[self.train_subset.indices])):

      train = torch.utils.data.Subset(self, train_index)
      val = torch.utils.data.Subset(self, validation_index)

      trainloader = torch.utils.data.DataLoader(
          train,
          batch_size=self.config.batch_size,
          shuffle=shuffle,
          num_workers=0,
          pin_memory=False,
      )
      valloader = torch.utils.data.DataLoader(
          val,
          batch_size=self.config.batch_size,
          shuffle=shuffle,
          num_workers=0,
          pin_memory=False,
      )

      print(
          f'Fold({i + 1,}), train: {len(trainloader.dataset)}, test: {len(valloader.dataset)}'
      )

      yield trainloader, valloader

  @property
  def config(self):
    return self._config
