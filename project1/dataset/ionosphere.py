"""
Load, process and clean the data for the ionosphere dataset.

https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from dataset import DATASET_DIR


class IonosphereDataset:

    feature_columns = [
        'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',
        'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29',
        'f30', 'f31', 'f32', 'f33'
    ]
    label_column = 'good'


    def __init__(self):
        self.path = DATASET_DIR / 'ionosphere.data'

        self.data = self._load_dataset(self.path)

        # Split the dataset into train and test
        self.train_data, self.test_data = self._train_test_split()
        self._set_standardization_values()

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        print(f'Loading following dataset: {path.resolve()}.')
        names = np.append(self.feature_columns, [self.label_column])
        df = pd.read_csv(path, header=None, names=names)

        # Remap binary features
        df['good'] = df['good'].map({'b': 0, 'g': 1})

        return df

    def _train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Shuffle the dataset
        df = self.data.copy(deep=True)
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and test, 75:25
        test = df[0:int(df.shape[0] / 4)]
        train = df[int(df.shape[0] / 4):]

        return train, test

    def get_continuous_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are continuous features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        return df[self.feature_columns].copy(deep=True), df[self.label_column].copy(deep=True)

    def _set_standardization_values(self):
        df = self.get_continuous_data()[0]
        self.means = df.mean()
        self.stds = df.std()

    def get_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features into design matrix, standardize and add bias value."""
        df = self.test_data if test else self.train_data

        x = df[self.feature_columns].copy(deep=True)
        y = df[self.label_column].copy(deep=True)

        # Standardize the feature values
        # Note: we calculate the standardization parameters on the train split and apply the same transformation to the test split
        x = (x - self.means) / self.stds

        # Remove features containing nan values
        mask = x.notna().all()
        mask = mask.index[mask]
        x = x[mask]

        # Add bias feature of 1 to front of dataframe
        x.insert(0, 'bias', 1)

        return x.to_numpy(), y.to_numpy().flatten()