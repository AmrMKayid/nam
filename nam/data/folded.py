from typing import Tuple
from typing import Union

import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .base import NAMDataset


class FoldedDataset(NAMDataset):

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

        self.train_subset, self.test_subset = self.get_folds()

    def get_folds(self, shuffle: bool = True, random_state: int = 42) -> Tuple[Subset, ...]:
        fold_num = self.config.fold_num
        n_splits = self.config.num_splits
        stratified = not self.config.regression

        if stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        assert fold_num <= n_splits and fold_num > 0, 'Pass a valid fold number.'
        for train_index, test_index in kf.split(self.features, self.targets):
            if fold_num == 1:
                train = Subset(self, train_index)
                test = Subset(self, test_index)
                return train, test
            else:
                fold_num -= 1

    def train_dataloaders(self,
                          test_size: int = 0.125,
                          shuffle: bool = True,
                          random_state: int = 42) -> Tuple[DataLoader, ...]:

        num_folds = self.config.num_folds
        stratified = not self.config.regression
        if stratified:
            shuffle_split = StratifiedShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=random_state)
        else:
            shuffle_split = ShuffleSplit(n_splits=num_folds, test_size=test_size, random_state=random_state)

        for i, (train_index, validation_index) in enumerate(
                shuffle_split.split(self.features[self.train_subset.indices], self.targets[self.train_subset.indices])):

            train = Subset(self.train_subset, train_index)
            val = Subset(self.train_subset, validation_index)

            trainloader = DataLoader(
                train,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
            )
            valloader = DataLoader(
                val,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            logger.info(f'Fold[{i + 1}]: train: {len(trainloader.dataset)}, val: {len(valloader.dataset)}')

            yield trainloader, valloader

    def test_dataloaders(self) -> DataLoader:
        return DataLoader(self.test_subset, batch_size=self.config.batch_size, shuffle=False)
