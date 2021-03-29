from typing import Callable
from typing import Tuple
from typing import Union

import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from .base import NAMDataset


class FoldedDataset(NAMDataset):

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
    super(FoldedDataset, self).__init__(
        config=config,
        file_path=file_path,
        features_columns=features_columns,
        targets_column=targets_column,
        weights_column=weights_column,
        transforms=transforms,
    )

    self.train_subset, self.test_subset = self.get_folds()

  def get_folds(
      self,
      fold_num: int = 1,
      num_folds: int = 5,
      shuffle: bool = True,
      random_state: int = 42,
  ) -> Tuple[torch.utils.data.Subset, ...]:
    stratified = not self.config.regression

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

  def train_dataloaders(
      self,
      n_splits: int = 5,
      batch_size: int = 32,
      test_size: int = 0.125,
      shuffle: bool = True,
      random_state: int = 42,
  ) -> Tuple[torch.utils.data.DataLoader, ...]:
    stratified = not self.config.regression
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
          num_workers=self.config.num_workers,
          pin_memory=False,
      )
      valloader = torch.utils.data.DataLoader(
          val,
          batch_size=self.config.batch_size,
          shuffle=False,
          num_workers=self.config.num_workers,
          pin_memory=False,
      )

      logger.info(
          f'Fold[{i + 1}]: train: {len(trainloader.dataset)}, val: {len(valloader.dataset)}'
      )

      yield trainloader, valloader

  def test_dataloaders(self) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        self.test_subset,
        batch_size=self.config.batch_size,
        shuffle=False,
        num_workers=self.config.num_workers,
        pin_memory=False,
    )
