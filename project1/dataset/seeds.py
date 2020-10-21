"""
Load, process and clean the data for the seeds dataset.

https://archive.ics.uci.edu/ml/machine-learning-databases/seeds/
"""
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from dataset import DATASET_DIR

class SeedsDataset:

    feature_columns = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove length"]
    label_column = 'canadian'

    def __init__(self):
        self.train_path = DATASET_DIR / 'seeds_dataset.data'

        self.data, self.missing = self._load_dataset(self.train_path)

        # Split the dataset into train and test
        self.train_data, self.test_data = self._train_test_split()

        self._set_standardization_values()


    def _load_dataset(self, path: Path) -> pd.DataFrame:
        print(f'Loading following dataset: {path.resolve()}.')
        names = np.append(self.feature_columns, [self.label_column])
        print(names)

        df = pd.read_csv(path, header=None, names=names, skipinitialspace=True, na_values=['?'])

        # Map labels to a binary class
        # We define 1 as the class of seed being canadian and 0 as the class of seed not being canadian
        # Originally, 1=Kama, 2=Rosa, 3=Canadian
        df['canadian'] = df['canadian'].apply(lambda x: 1 if x == 3 else 0)

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