"""
Load, process and clean the data for the adult dataset.

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from dataset import DATASET_DIR


class AdultDataset:

    feature_columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
    ]
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'native_country']
    continuous_features = ['age', 'education_num', 'hours_per_week']
    binary_features = ['sex']
    drop_features = ['fnlwgt', 'capital_gain', 'capital_loss', 'relationship']
    label_column = 'rich'

    def __init__(self):
        path = DATASET_DIR / 'adult.data'
        #self.test_path = DATASET_DIR / 'adult.test'

        self.data, self.missing = self._load_dataset(path)

        # We get the one hot and add it to data using pandas method
        self.one_hot = pd.get_dummies(self.data[self.categorical_features], drop_first=True)
        self.one_hot_features = self.one_hot.columns
        self.data = pd.concat([self.data, self.one_hot], axis=1)

        # Split the dataset into train and test
        self.train_data, self.test_data = self._train_test_split()

        self._set_standardization_values()

    def _load_dataset(self, path: Path) -> pd.DataFrame:
        print(f'Loading following dataset: {path.resolve()}.')
        names = np.append(self.feature_columns, [self.label_column])

        df = pd.read_csv(path, header=None, names=names, skipinitialspace=True, na_values=['?'])


        # Remap binary features
        df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
        df['rich'] = df['rich'].map({'<=50K': 0, '>50K': 1})

        df.drop(self.drop_features, axis=1)

        # Keep track of rows with missing features
        # Please don't delete this, we need to keep track of the data with missing features for the report
        df_missing = df[(df == '?').any(axis=1)]
        print(df_missing)
        df = df.dropna()

        return df, df_missing

    def _train_test_split(self, one_hot: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Shuffle the dataset
        df = self.one_hot.copy(deep=True) if one_hot else self.data.copy(deep=True)
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and test, 75:25
        test = df[0:int(df.shape[0] / 4)]
        train = df[int(df.shape[0] / 4):]

        return train, test

    def get_binary_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are binary features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        features = ['sex']
        labels = 'rich'

        return df[features].copy(deep=True), df[labels].copy(deep=True)

    def get_categorical_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are categorical features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
        labels = 'rich'

        return df[features].copy(deep=True), df[labels].copy(deep=True)

    def get_continuous_data(self, test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns a deep copy of a DataFrame containing only the columns
        that are continuous features, and another deep copy containing
        the label columns.

        :params test: Flag to get the test data
        :returns: The feature DataFrame and the label DataFrame
        """
        df = self.test_data if test else self.train_data
        features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        labels = 'rich'

        return df[features].copy(deep=True), df[labels].copy(deep=True)

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

        binary = self.get_binary_data(test)[0].to_numpy()

        features = np.hstack((np.array([np.ones(continuous.shape[0])]).T, continuous.to_numpy(), binary, categorical))
        return features, labels