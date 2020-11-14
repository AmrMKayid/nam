from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from nam.types import DataType


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
      config,
      *,
      csv_file: str,
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
        header (str, optional): [description]. Defaults to 'infer'.
        names (list, optional): [description]. Defaults to None.
        delim_whitespace (bool, optional): [description]. Defaults to False.
        preprocess_fn (Callable, optional): [description]. Defaults to None.
        transform (Callable, optional): [description]. Defaults to None.
    """
    self._config = config
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

    self.x_train = self.data.x_train
    self.y_train = self.data.y_train
    self.transforms = transforms

  def __len__(self):
    return len(self.y_train)

  def __getitem__(self, idx: int) -> DataType:
    if self.transforms is not None:
      data = self.transforms(self.data)
    return self.x_train[idx], self.y_train[idx]

  @property
  def config(self):
    return self._config
