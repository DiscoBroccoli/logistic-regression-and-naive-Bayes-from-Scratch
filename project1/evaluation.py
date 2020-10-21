import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# [0,0] = TN
# [1,1] = TP
# [0,1] = FP
# [1,0] = FN

# cm is a confusion matrix

# Accuracy: (TP + TN) / Total
def accuracy(cm: pd.DataFrame) -> float:
    return (cm[0,0] + cm[1,1]) / cm.sum()

# Precision: TP / (TP + FP)
def precision(cm: pd.DataFrame) -> float:
    return cm[1,1] / (cm[1,1] + cm[0,1])

# False positive rate: FP / N = FP / (FP + TN)
def false_positive(cm: pd.DataFrame) -> float:
    return cm[0,1] / (cm[0,0] + cm[0,1])

# True positive rate: TP / P = TP / (TP + FN)
# Equivalent to sensitivity/recall
def true_positive(cm: pd.DataFrame) -> float:
    return cm[1,1] / (cm[1,0] + cm[1,1])

# F1 score:  2 * precision * recall / (precision + recall)
def f_score(cm: pd.DataFrame) -> float:
    return 2 * precision(cm) * true_positive(cm) / (precision(cm) + true_positive(cm))


# Returns a confusion matrix for labels and predictions
# [[TN, FP],
#  [FN, TP]]
def confusion_matrix(y, y_hat):
    cm = np.zeros((2, 2))
    np.add.at(cm, [y.astype(int), y_hat.astype(int)], 1)
    return cm


def visualize_cm(cm):
    df_cm = pd.DataFrame(cm, columns=['0', '1'], index=['0', '1'])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(5, 3))
    sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size': 16}, fmt='g')


# Function to return two shuffled arrays, is a deep copy
def shuffle(x, y):
    x_copy = x.copy()
    y_copy = y.copy()
    rand = np.random.randint(0, 10000)
    np.random.seed(rand)
    np.random.shuffle(x_copy)
    np.random.seed(rand)
    np.random.shuffle(y_copy)
    return x_copy, y_copy

# Shuffles and splits data into two sets
# test split will be 1/size of the data
def split(x, y, size):
    x1, y1, = shuffle(x, y)
    x1_test = x1[0:int(x1.shape[0] / size)]
    x1_train = x1[int(x1.shape[0] / size):]
    y1_test = y1[0:int(y1.shape[0] / size)]
    y1_train = y1[int(y1.shape[0] / size):]
    return x1_train, x1_test, y1_train, y1_test


def cross_validation(k, X, Y, model, lr=0.5, regularization=0, eps=1e-2, verbose=True):
    # randomize X and Y by shuffling
    x, y = shuffle(X, Y)

    # split into k folds
    x_folds = np.array_split(x, k)
    y_folds = np.array_split(y, k)

    acc = 0
    f1 = 0
    prec = 0
    rec = 0
    cms = []

    for i in range(k):
        validation_features = x_folds[i]
        validation_labels = np.squeeze(y_folds[i])

        train_features = np.delete(x_folds, i, axis=0)
        train_features = np.concatenate(train_features)
        train_labels = np.delete(y_folds, i, axis=0)
        train_labels = np.concatenate(train_labels)

        m = model(train_features, train_labels)
        m.fit(lr, verbose=False, regularization=regularization, eps=eps)

        predicted_labels = m.predict(validation_features)
        cm = confusion_matrix(validation_labels, predicted_labels)

        acc += accuracy(cm)
        f1 += f_score(cm)
        prec += precision(cm)
        rec += true_positive(cm)
        cms.append(cm)

    if verbose:
        print("Accuracy:", acc/k, "Precision:", prec/k, "Recall:", rec/k, "F1:", f1/k)
    # Return the accuracy and array of confusion matrices
    return acc/k, np.array(cms)

# assume 5 fold for now
def cross_validation_naive(k, df, model, label, cont=[], cat=[], bin=[]):
    df = df.copy(deep=True)
    np.random.shuffle(df.values)
    df = df.reset_index(drop=True)

    indices = np.arange(df.shape[0])
    indices = np.array_split(indices, k)

    acc = 0
    f1 = 0
    prec = 0
    rec = 0
    cms = []

    for i in range(k):
        val = df.loc[indices[i]]

        train = df.loc[np.concatenate(np.delete(indices, i, axis=0))]

        m = model(train, label, cont, cat, bin)
        pred = val.apply(m.predict, axis=1)

        cm = confusion_matrix(val[label], pred)

        acc += accuracy(cm)
        f1 += f_score(cm)
        prec += precision(cm)
        rec += true_positive(cm)
        cms.append(cm)

    print("Accuracy:", acc / k, "Precision:", prec / k, "Recall:", rec / k, "F1:", f1 / k)

        # Return the accuracy and array of confusion matrices
    return acc / k, np.array(cms)



def cv_task_2(k, X, Y, model, lr = 0.5, regularization=0, eps = 1e-2, iterations=200):
    # randomize X and Y by shuffling
    x, y = shuffle(X, Y)

    # split into k folds
    x_folds = np.array_split(x, k)
    y_folds = np.array_split(y, k)

    train_acc_history = np.empty([k, iterations])
    val_acc_history = np.empty([k, iterations])


    for i in range(k):
        val_features = x_folds[i]
        val_labels = np.squeeze(y_folds[i])

        train_features = np.delete(x_folds, i)
        train_features = np.concatenate(train_features)
        train_labels = np.delete(y_folds, i, axis=0)
        train_labels = np.concatenate(train_labels)

        m = model(train_features, train_labels)

        costs = []
        train_accuracies = []
        val_accuracies = []

        # Keep on training until difference reached threshold
        for j in range(iterations):

            # fit model for 1 iteration
            cost = m.fit(lr=lr, verbose=False, regularization=regularization, eps=None, epochs=1)
            costs.append(cost)

            # predict the labels and eval accuracy for train and val split
            val_pred_labels = m.predict(val_features)
            train_pred_labels = m.predict(train_features)
            cm_val = confusion_matrix(val_labels, val_pred_labels)
            cm_train = confusion_matrix(train_labels, train_pred_labels)

            val_accuracies.append(accuracy(cm_val))
            train_accuracies.append(accuracy(cm_train))

        # store the costs and accuracies
        train_acc_history[i] = np.array(train_accuracies)
        val_acc_history[i] = np.array(val_accuracies)

    return train_acc_history, val_acc_history


def grid_search(learning_rates, epsilons, lambdas, x, y, model):
    max_acc = 0
    arg_max = [0,0,0]

    for lr in learning_rates:
        for eps in epsilons:
            for regularization in lambdas:
                #print(lr, eps, regularization)
                acc, cm = cross_validation(5, x, y, lr=lr, eps=eps, regularization=regularization, model=model, verbose=False)
                if acc > max_acc:
                    max_acc = acc
                    arg_max = [lr, eps, regularization]
                    max_cm = cm
    f1 = []
    prec = []
    rec = []
    for cm in max_cm:
        f1.append(f_score(cm))
        prec.append(precision(cm))
        rec.append(true_positive(cm))

    f1 = np.mean(f1)
    prec = np.mean(prec)
    rec = np.mean(rec)

    print(arg_max)
    print("Accuracy:", max_acc, "Precision:", prec, "Recall:", rec, "F1:", f1)


    return max_acc, arg_max