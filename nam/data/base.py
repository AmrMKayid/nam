from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder


def preprocess_df(data: pd.DataFrame) -> pd.DataFrame:
  """One Hot Encoding.

  Args:
      data (pd.DataFrame): unprocessed dataframe

  Returns:
      pd.DataFrame: processed dataframe with one hot encoded columns
  """
  ## Save label encoder (Mapping -> str to int)
  # label_encoder, oh_encoder = LabelEncoder(), OneHotEncoder()
  # return data.apply(label_encoder.fit_transform)
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
      one_hot: bool = False,
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

    self.features = torch.tensor(
        self.data[features_columns].copy().to_numpy()).float()

    if torch.isnan(self.features).any():
      raise InterruptedError('Dataset features columns contains NAN values')

    if one_hot:
      self.oh_encoder = OneHotEncoder()
      self.targets = torch.tensor(
          self.oh_encoder.fit_transform(
              self.data[targets_column].copy()).toarray())
    else:
      self.targets = torch.tensor(
          self.data[targets_column].copy().to_numpy()).float()

    if weights_column is not None:
      self.weights = torch.tensor(
          self.data[weights_column].copy().to_numpy()).float()

    self.transforms = transforms

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
    ##TODO(amr): discuss weights columns with Nick and how can we use it
    # if self.weights_column is not None:
    #   return self.features[idx], self.weights[idx], self.targets[idx]

    return self.features[idx], self.targets[idx]

  @property
  def config(self):
    return self._config
