from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer

from nam.config import defaults
from nam.data.folded import FoldedDataset

cfg = defaults()


def load_breast_data(config=cfg) -> Dict:
  breast_cancer = load_breast_cancer()
  dataset = pd.DataFrame(data=breast_cancer.data,
                         columns=breast_cancer.feature_names)
  dataset['target'] = breast_cancer.target

  config.regression = False

  return FoldedDataset(
      config=config,
      file_path=dataset,
      features_columns=dataset.columns[:-1],
      targets_column=dataset.columns[-1],
  )


def load_sklearn_housing_data(config=cfg) -> Dict:
  housing = sklearn.datasets.fetch_california_housing()
  dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
  dataset['target'] = housing.target

  config.regression = True

  return FoldedDataset(
      config=config,
      file_path=dataset,
      features_columns=dataset.columns[:-1],
      targets_column=dataset.columns[-1],
  )


def load_housing_data(
    config=cfg,
    housing_path: str = 'data/housing.csv',
    features_columns: list = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ],
    targets_column: str = 'median_house_value',
) -> Dict:

  config.regression = True

  return FoldedDataset(
      config=config,
      file_path=housing_path,
      features_columns=features_columns,
      targets_column=targets_column,
  )


def load_gallup_data(
    config=cfg,
    gallup_path: str = 'data/GALLUP.csv',
    features_columns: list = [
        "country", "income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"
    ],
    targets_column: str = "WP16",
    weights_column: str = "wgt",
) -> Dict:

  ## TODO: multi-classification
  config.regression = False
  data = pd.read_csv(gallup_path)
  data["WP16"] = np.where(data["WP16"] < 6, 0, 1)

  return FoldedDataset(
      config=config,
      file_path=data,
      features_columns=features_columns,
      targets_column=targets_column,
      weights_column=weights_column,
  )
