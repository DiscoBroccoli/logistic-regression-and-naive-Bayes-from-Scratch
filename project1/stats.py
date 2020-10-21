import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from dataset.ionosphere import IonosphereDataset
from dataset.abalone import AbaloneDataset
from dataset.adult import AdultDataset
from dataset.seeds import SeedsDataset


# Plot the distributions of labels and features
def distributions(df, label, categorical_features=[], continuous_features=[], binary_features=[], max_col=4):
    num_plots = 1 + len(categorical_features) + len(continuous_features) + len(binary_features)

    f, axes = plt.subplots(math.ceil(num_plots / max_col), max_col)

    axes = axes.flatten()
    index = 0

    # Plot labels
    sns.distplot(a=df[label], ax=axes[index], kde=False)
    index += 1

    # Plot categorical
    for feature in (categorical_features + binary_features):
        sns.countplot(x=feature, data=df, ax=axes[index], palette="Blues_d")
        axes[index].set_ylabel('')
        axes[index].set_xticklabels('')

        #axes[index].set_xticklabels(axes[index].get_xticklabels(), rotation=90)
        index += 1

    # Plot continuous
    for feature in continuous_features:
        sns.distplot(a=df[feature], ax=axes[index], kde=False)
        index += 1

    for i in range(index, len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.show()

# Calculate statistics on the continuous features and labels
def normal_stats(df, features, label):
    columns = np.append(features, [label])
    df = df[columns]
    means = df.mean()
    std = df.std()
    max = df.max()
    min = df.min()

    df_stats = pd.DataFrame({"Mean": means, "Standard Deviation": std, "Max": max, "Min": min})
    return df_stats

abalone_dataset = AbaloneDataset()
# 3. Abalone
#distributions(abalone_dataset.data, abalone_dataset.label_column,
#                    categorical_features=abalone_dataset.categorical_features,
#                    continuous_features=abalone_dataset.continuous_features)
print("Statistics for Abalone:\n", normal_stats(abalone_dataset.data, abalone_dataset.continuous_features, abalone_dataset.label_column))

'''
ionosphere_dataset = IonosphereDataset()
abalone_dataset = AbaloneDataset()
adult_dataset = AdultDataset()
seeds_dataset = SeedsDataset()

# Task 1. Perform statistics on each of the datasets
# 1. Ionosphere
#distributions(ionosphere_dataset.data, ionosphere_dataset.label_column,
#                    continuous_features=ionosphere_dataset.feature_columns,
#                    max_col=7)
print("Statistics for Ionosphere:\n", normal_stats(ionosphere_dataset.data, ionosphere_dataset.feature_columns, ionosphere_dataset.label_column))

# 2. Adult
distributions(adult_dataset.data, adult_dataset.label_column,
                    categorical_features=adult_dataset.categorical_features)
                    #continuous_features=adult_dataset.continuous_features,
                    #binary_features=adult_dataset.binary_features)
print("Statistics for Adult:\n", normal_stats(adult_dataset.data, adult_dataset.continuous_features, adult_dataset.label_column))

# 3. Abalone
#distributions(abalone_dataset.data, abalone_dataset.label_column,
#                    categorical_features=abalone_dataset.categorical_features,
#                    continuous_features=abalone_dataset.continuous_features)
print("Statistics for Abalone:\n", normal_stats(abalone_dataset.data, abalone_dataset.continuous_features, abalone_dataset.label_column))


# 4. Seeds
#distributions(seeds_dataset.data, seeds_dataset.label_column, continuous_features=seeds_dataset.feature_columns)
print("Statistics for Seeds:\n", normal_stats(seeds_dataset.data, seeds_dataset.feature_columns, seeds_dataset.label_column))
'''
