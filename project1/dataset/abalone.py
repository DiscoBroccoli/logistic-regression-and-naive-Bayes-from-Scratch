"""
Load, process and clean the data for the abalone dataset.

https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from dataset import DATASET_DIR


class AbaloneDataset:

    feature_columns = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'
    ]
    categorical_features = ['Sex']
    continuous_features = ['Length', 'Diameter', 'Height', 'Whole weight',
                           'Shucked weight', 'Viscera weight', 'Shell weight']
    label_column = 'Rings'

    def __init__(self):
        path = DATASET_DIR / 'abalone.data'

        self.data, self.missing = self._load_dataset(path)

        # We get the one hot and add it to data
        self.one_hot = pd.get_dummies(self.data[self.categorical_features], drop_first=True)
        self.one_hot_features = self.one_hot.columns
        self.data = pd.concat([self.data, self.one_hot], axis=1)

        # Split the dataset into train and test
        self.train_data, self.test_data = self._train_test_split()

        self._set_standardization_values()

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        print(f'Loading following dataset: {path.resolve()}.')
        names = np.append(self.feature_columns, [self.label_column])
        print(names)

        df = pd.read_csv(path, header=None, names=names, skipinitialspace=True, na_values=['?'])

        # Map labels to a binary class
        # We define <= 9 for class 0 and > 9 for class 1
        # 9 is chosen because it is the median, providing a good split of the data
        df['Rings'] = df['Rings'].apply(lambda x: 0 if x <= 9 else 1)

        # Keep track of rows with missing features
        df_missing = df[(df == '?').any(axis=1)]
        df = df.dropna()

        return df, df_missing

    def _train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Shuffle the dataset
        df = self.data.copy(deep=True)
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and test, 75:25
        test = df[0:int(df.shape[0] / 4)]
        train = df[int(df.shape[0] / 4):]

        return train, test

    def get_categorical_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are categorical features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        return df[self.categorical_features].copy(deep=True), df[self.label_column].copy(deep=True)

    def get_continuous_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are continuous features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        return df[self.continuous_features].copy(deep=True), df[self.label_column].copy(deep=True)

    def get_one_hot_data(self, test: bool=False):
        df = self.test_data if test else self.train_data

        return df[self.one_hot_features].copy(deep=True), df[self.label_column].copy(deep=True)

    def _set_standardization_values(self):
        df = self.get_continuous_data()[0]
        self.means = df.mean()
        self.stds = df.std()

    # return one hot encoding features as design matrix
    def get_data(self, test: bool = False):
        # get one hot features
        categorical = self.get_one_hot_data(test)[0].to_numpy()

        # get continuous features and standardize
        continuous, labels = self.get_continuous_data(test)
        labels = labels.to_numpy().flatten()
        continuous = (continuous - self.means) / self.stds

        features = np.hstack((np.array([np.ones(continuous.shape[0])]).T, continuous, categorical))
        return features, labels
