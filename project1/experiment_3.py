import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import evaluation
from dataset.ionosphere import IonosphereDataset
from dataset.abalone import AbaloneDataset
from dataset.adult import AdultDataset
from dataset.seeds import SeedsDataset
from models.logistic_regression import LogisticRegression
from models.Naive_Bayes import NaiveBayes



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


# 3. Plot accuracy as function of dataset size for logistic and naive bayes

# for logistic
def task_3_logistic(x, y, x_test, y_test, args):
    accuracies = []
    sizes = np.linspace(10, 200, num=20)
    N = y.shape[0]
    for size in sizes:
        acc = 0
        for i in range(50):

            rand = np.random.randint(int(N), size=int(size))
            m = LogisticRegression(x[rand], y[rand])
            m.fit(lr=args[0], eps=args[1], regularization=args[2])
            pred = m.predict(x_test)
            cm = evaluation.confusion_matrix(y_test, pred)
            acc += evaluation.accuracy(cm)

        accuracies.append(acc/50)

    return accuracies, sizes

def task_3_naive(df, test_df, label, cont=[], cat=[], bin=[]):
    accuracies = []
    sizes = np.linspace(10, 200, num=20)
    N = df.shape[0]
    for size in sizes:
        acc = 0
        for i in range(25):
            print(size, i)
            rand = np.random.randint(int(N), size=int(size))
            m = NaiveBayes(df.loc[rand], label, continuous=cont, categorical=cat, binary=bin)
            pred = test_df.apply(m.predict, axis=1)

            cm = evaluation.confusion_matrix(test_df[label].to_numpy(), pred.to_numpy())
            acc += evaluation.accuracy(cm)

        accuracies.append(acc/25)



    return accuracies, sizes

def task_3(args, data, train_features, train_labels, label, cont=[], cat=[], bin=[]):
    # Get the size of data, select 10% at random and set aside
    df = data.copy(deep=True).reset_index(drop=True)
    N = df.shape[0]

    # 10% of indices
    indices = np.random.randint(int(N), size=int(40))

    test_df = df.loc[indices]
    train_df = df.drop(indices).reset_index(drop=True)

    x_test = train_features[indices]
    x_train = np.delete(train_features, indices, axis=0)

    y_test = train_labels[indices]
    y_train = np.delete(train_labels, indices, axis=0)

    acc_1, size_1 = task_3_logistic(x_train, y_train, x_test, y_test, args)
    acc_2, size_2 = task_3_naive(train_df, test_df, label, cont, cat, bin)
    sns.lineplot(x=size_1, y=acc_1, color="blue",legend=False)

    sns.lineplot(x=size_2, y=acc_2, color="orange", legend=False)
    plt.legend(title='Model', loc='lower right', labels=['Logistic Regression', 'Naive Bayes'])
    plt.xlabel("Training set size")
    plt.ylabel("Test accuracy")

    plt.show()

#task_3([0.05, 0.05, 0.1],ionosphere_dataset.train_data, ionosphere_train_features, ionosphere_train_labels, ionosphere_dataset.label_column, ionosphere_dataset.feature_columns)
#task_3([0.1, 0.01, 0], adult_dataset.train_data, adult_train_features, adult_train_labels, adult_dataset.label_column, adult_dataset.continuous_features, adult_dataset.categorical_features, adult_dataset.binary_features)
#task_3([0.5, 0.01, 0],abalone_dataset.train_data, abalone_train_features, abalone_train_labels, abalone_dataset.label_column, abalone_dataset.continuous_features, abalone_dataset.categorical_features)
task_3([0.05, 0.01, 0], seeds_dataset.train_data, seeds_train_features, seeds_train_labels, seeds_dataset.label_column, seeds_dataset.feature_columns)
