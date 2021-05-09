from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer

from nam.config import defaults
from nam.data.base import NAMDataset
from nam.data.folded import FoldedDataset

cfg = defaults()


def load_breast_data(config=cfg) -> Dict:
    breast_cancer = load_breast_cancer()
    dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    dataset['target'] = breast_cancer.target

    config.regression = False

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=dataset,
                             features_columns=dataset.columns[:-1],
                             targets_column=dataset.columns[-1])
    else:
        return NAMDataset(config,
                          data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])


def load_sklearn_housing_data(config=cfg) -> Dict:
    housing = sklearn.datasets.fetch_california_housing()

    dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    dataset['target'] = housing.target

    config.regression = True

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=dataset,
                             features_columns=dataset.columns[:-1],
                             targets_column=dataset.columns[-1])
    else:
        return NAMDataset(config,
                          data_path=dataset,
                          features_columns=dataset.columns[:-1],
                          targets_column=dataset.columns[-1])


def load_housing_data(config=cfg,
                      data_path: str = 'data/housing.csv',
                      features_columns: list = [
                          'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                          'households', 'median_income'
                      ],
                      targets_column: str = 'median_house_value') -> Dict:

    config.regression = True
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'population', 'AveOccup', 'Latitude', 'Longitude']

    data = pd.read_csv(data_path)

    data['MedInc'] = data['median_income']
    data['HouseAge'] = data['housing_median_age']
    data['Latitude'] = data['latitude']
    data['Longitude'] = data['longitude']

    # avg rooms = total rooms / households
    data['AveRooms'] = data['total_rooms'] / data["households"]

    # avg bed rooms = total bed rooms / households
    data['AveBedrms'] = data['total_bedrooms'] / data["households"]

    # avg occupancy = population / households
    data['AveOccup'] = data['population'] / data['households']

    data[targets_column] = data[targets_column] / 100000.0

    if config.cross_val:
        return FoldedDataset(config, data_path=data, features_columns=feature_names, targets_column=targets_column)
    else:
        return NAMDataset(config, data_path=data, features_columns=feature_names, targets_column=targets_column)


def load_gallup_data(config=cfg,
                     data_path: str = 'data/GALLUP.csv',
                     features_columns: list = ["country", "income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"],
                     targets_column: str = "WP16",
                     weights_column: str = "wgt") -> Dict:

    config.regression = False
    data = pd.read_csv(data_path)
    data["WP16"] = np.where(data["WP16"] < 7, 0, 1)
    # data = data.sample(frac=0.1)

    if config.cross_val:
        return FoldedDataset(config,
                             data_path=data,
                             features_columns=features_columns,
                             targets_column=targets_column,
                             weights_column=weights_column)
    else:
        return NAMDataset(config,
                          data_path=data,
                          features_columns=features_columns,
                          targets_column=targets_column,
                          weights_column=weights_column)
