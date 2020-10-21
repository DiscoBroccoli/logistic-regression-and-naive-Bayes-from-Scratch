import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation
from dataset.ionosphere import IonosphereDataset
from dataset.abalone import AbaloneDataset
from dataset.adult import AdultDataset
from dataset.seeds import SeedsDataset
from models.logistic_regression import LogisticRegression

ionosphere_dataset = IonosphereDataset()
abalone_dataset = AbaloneDataset()
adult_dataset = AdultDataset()
seeds_dataset = SeedsDataset()

# Train and test data for each dataset
ionosphere_train_features, ionosphere_train_labels = ionosphere_dataset.get_data()
ionosphere_test_features, ionosphere_test_labels = ionosphere_dataset.get_data(test=True)

adult_train_features, adult_train_labels = adult_dataset.get_data()
adult_test_features, adult_test_labels = adult_dataset.get_data(test=True)

abalone_train_features, abalone_train_labels = abalone_dataset.get_data()
abalone_test_features, abalone_test_labels = abalone_dataset.get_data(test=True)

seeds_train_features, seeds_train_labels = seeds_dataset.get_data()
seeds_test_features, seeds_test_labels = seeds_dataset.get_data(test=True)

# Grid search hyperparameters
lrs = [0.05, 0.1, 0.5, 1]
eps = [0.01, 0.05, 0.1, 0.5]

# Task 3. Experiments


# 2. Plot accuracy on train/validation set as a function of iterations of gradient descent

def plot_data(acc):
    train_acc = acc[0]
    val_acc = acc[1]
    index = np.tile(np.arange(train_acc.shape[1]), 10)
    event = np.repeat(np.array(['train', 'val']), train_acc.shape[1] * 5)
    accuracies = np.stack((train_acc.flatten(), val_acc.flatten())).flatten()
    df = pd.DataFrame({'Accuracy': accuracies, 'Iteration': index, 'event': event})
    return df

def task_2(x, y):
    x, y, = evaluation.shuffle(x, y)

    # Similar to cross validation here but we get an accuracy prediction after every iteration
    df_1 = plot_data(evaluation.cv_task_2(5, x, y, model=LogisticRegression, iterations=200, lr=0.05))
    df_2 = plot_data(evaluation.cv_task_2(5, x, y, model=LogisticRegression, iterations=200, lr=0.1))
    df_3 = plot_data(evaluation.cv_task_2(5, x, y, model=LogisticRegression, iterations=200, lr=0.5))
    df_4 = plot_data(evaluation.cv_task_2(5, x, y, model=LogisticRegression, iterations=200, lr=1))

    fig, axes = plt.subplots(2, 2)

    sns.lineplot(x='Iteration', y='Accuracy', data=df_1, hue='event', ax=axes[0,0])
    axes[0,0].set_title("lr = 0.05")
    sns.lineplot(x='Iteration', y='Accuracy', data=df_2, hue='event', ax=axes[0,1])
    axes[0,1].set_title("lr = 0.1")

    sns.lineplot(x='Iteration', y='Accuracy', data=df_3, hue='event', ax=axes[1,0])
    axes[1,0].set_title("lr = 0.5")

    sns.lineplot(x='Iteration', y='Accuracy', data=df_4, hue='event', ax=axes[1,1])
    axes[1,1].set_title("lr = 1")


    for ax in axes.flat:
        ax.label_outer()


    plt.tight_layout()
    plt.show()

#task_2(ionosphere_train_features, ionosphere_train_labels)
#task_2(seeds_train_features, seeds_train_labels)

#task_2(abalone_train_features, abalone_train_labels)
#task_2(adult_train_features, adult_train_labels)