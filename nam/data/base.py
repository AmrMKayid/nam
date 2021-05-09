from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from .utils import read_dataset
from .utils import transform_data


class CSVDataset(torch.utils.data.Dataset):

    def __init__(self,
                 config,
                 data_path: Union[str, pd.DataFrame],
                 features_columns: list,
                 targets_column: str,
                 weights_column: str = None):
        """Custom dataset for csv files.

        Args:
            config ([type]): [description]
            data_path (str): [description]
            features_columns (list): [description]
            targets_column (str): [description]
            weights_column (str, optional): [description]. Defaults to None.
        """

        self._config = config
        self.data_path = data_path
        self.features_columns = features_columns
        self.targets_column = targets_column
        self.weights_column = weights_column

        self.data = read_dataset(data_path)

        self.raw_X = self.data[features_columns].copy()
        self.raw_y = self.data[targets_column].copy()

        self.X, self.y = self.raw_X.to_numpy(), self.raw_y.to_numpy()
        if weights_column is not None:
            self.wgts = self.data[weights_column].copy().to_numpy()
        else:
            self.wgts = np.ones_like(self.raw_y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.X[idx], self.y[idx], self.wgts[idx]

    @property
    def config(self):
        return self._config


class NAMDataset(CSVDataset):

    def __init__(self,
                 config,
                 data_path: Union[str, pd.DataFrame],
                 features_columns: list,
                 targets_column: str,
                 weights_column: str = None) -> None:
        super().__init__(config=config,
                         data_path=data_path,
                         features_columns=features_columns,
                         targets_column=targets_column,
                         weights_column=weights_column)

        self.col_min_max = self.get_col_min_max()

        self.features, self.features_names = transform_data(self.raw_X)
        self.compute_features()

        if (not config.regression) and (not isinstance(self.raw_y, np.ndarray)):
            targets = pd.get_dummies(self.raw_y).values
            targets = np.array(np.argmax(targets, axis=-1))
        else:
            targets = self.y

        self.features = torch.from_numpy(self.features).float().to(config.device)
        self.targets = torch.from_numpy(targets).view(-1, 1).float().to(config.device)
        self.wgts = torch.from_numpy(self.wgts).to(config.device)

        self.setup_dataloaders()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx]  #, self.wgts[idx]

    def get_col_min_max(self):
        col_min_max = {}
        for col in self.raw_X:
            unique_vals = self.raw_X[col].unique()
            col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

        return col_min_max

    def compute_features(self):
        single_features = np.split(np.array(self.features), self.features.shape[1], axis=1)
        self.unique_features = [np.unique(f, axis=0) for f in single_features]

        self.single_features = {col: sorted(self.raw_X[col].to_numpy()) for col in self.raw_X}
        self.ufo = {col: sorted(self.raw_X[col].unique()) for col in self.raw_X}

    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])

        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)

        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)

        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)

    def train_dataloaders(self) -> Tuple[DataLoader, ...]:
        yield self.train_dl, self.val_dl

    def test_dataloaders(self) -> DataLoader:
        return self.test_dl
