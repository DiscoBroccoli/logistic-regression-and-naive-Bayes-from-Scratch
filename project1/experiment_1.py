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

# Grid search hyperparameters
lamdas = [0, 0.01, 0.05, 0.1, 0.5, 1]
lrs = [0.05, 0.1, 0.5, 1]
eps = [0.01, 0.05, 0.1, 0.5]

# Task 3. Experiments
# 1. Compare accuracy of naive bayes and logistic regression

# Get cross validation accuracy for 5-fold cv
print("Ionosphere validation accuracy (default parameters):")
evaluation.cross_validation(5, ionosphere_train_features, ionosphere_train_labels, model=LogisticRegression)

# Grid search for optimal hyperparameters
print("Ionosphere grid search hyperparameters:")
ionosphere_max_val_acc, ionosphere_arg_max = evaluation.grid_search(learning_rates=lrs, epsilons=eps, lambdas=lamdas, x=ionosphere_train_features, y=ionosphere_train_labels, model=LogisticRegression)

# Accuracy on test split - train with best hyperparameters
print("Ionosphere test accuracy:")
logistic_ionosphere = LogisticRegression(ionosphere_train_features, ionosphere_train_labels)
logistic_ionosphere.fit(lr=ionosphere_arg_max[0], eps=ionosphere_arg_max[1], regularization=ionosphere_arg_max[2])
ionosphere_prediction = logistic_ionosphere.predict(ionosphere_test_features)
cm_ionosphere = evaluation.confusion_matrix(ionosphere_test_labels, ionosphere_prediction)
print("Accuracy:", evaluation.accuracy(cm_ionosphere), "Precision:", evaluation.precision(cm_ionosphere), "Recall:", evaluation.true_positive(cm_ionosphere), "F1:", evaluation.f_score(cm_ionosphere))

# 5-fold CV for naive bayes
print("Ionosphere validation accuracy (naive bayes):")
evaluation.cross_validation_naive(5, ionosphere_dataset.train_data, NaiveBayes, ionosphere_dataset.label_column, ionosphere_dataset.feature_columns)

naive_ionosphere = NaiveBayes(ionosphere_dataset.train_data, ionosphere_dataset.label_column, continuous=ionosphere_dataset.feature_columns)

print("Ionosphere test accuracy (naive bayes):")

ionosphere_pred_naive = ionosphere_dataset.test_data.apply(naive_ionosphere.predict, axis=1)
cm_ionosphere_naive = evaluation.confusion_matrix(ionosphere_test_labels, ionosphere_pred_naive.to_numpy())
print("Accuracy:", evaluation.accuracy(cm_ionosphere_naive), "Precision:", evaluation.precision(cm_ionosphere_naive), "Recall:", evaluation.true_positive(cm_ionosphere_naive), "F1:", evaluation.f_score(cm_ionosphere_naive))


# Abalone -----


# Get cross validation accuracy for 5-fold cv
print("Abalone validation accuracy (default parameters):")
evaluation.cross_validation(5, abalone_train_features, abalone_train_labels, model=LogisticRegression)[0]

# Grid search for optimal hyperparameters
print("Abalone grid search hyperparameters:")
abalone_max_val_acc, abalone_arg_max = evaluation.grid_search(learning_rates=lrs, epsilons=eps, lambdas=lamdas,
                                                              x=abalone_train_features, y=abalone_train_labels,
                                                              model=LogisticRegression)

# Accuracy on test split - train with best hyperparameters
logistic_adult = LogisticRegression(abalone_train_features, abalone_train_labels)
logistic_adult.fit(lr=abalone_arg_max[0], eps=abalone_arg_max[1], regularization=abalone_arg_max[2])
abalone_prediction = logistic_adult.predict(abalone_test_features)
cm_abalone = evaluation.confusion_matrix(abalone_test_labels, abalone_prediction)
print("Abalone test accuracy:")

print("Accuracy:", evaluation.accuracy(cm_abalone), "Precision:", evaluation.precision(cm_abalone), "Recall:", evaluation.true_positive(cm_abalone), "F1:", evaluation.f_score(cm_abalone))

# 5-fold CV for naive bayes
print("Abalone validation accuracy (naive bayes):")
evaluation.cross_validation_naive(5, abalone_dataset.train_data, NaiveBayes, abalone_dataset.label_column,
                                                  abalone_dataset.continuous_features,
                                                  abalone_dataset.categorical_features)[0]

naive_abalone = NaiveBayes(abalone_dataset.train_data, abalone_dataset.label_column, continuous=abalone_dataset.continuous_features,
                         categorical=abalone_dataset.categorical_features)

abalone_pred_naive = abalone_dataset.test_data.apply(naive_abalone.predict, axis=1)
cm_abalone_naive = evaluation.confusion_matrix(abalone_test_labels, abalone_pred_naive.to_numpy())
print("Abalone test accuracy (Naive):")

print("Accuracy:", evaluation.accuracy(cm_abalone_naive), "Precision:", evaluation.precision(cm_abalone_naive), "Recall:", evaluation.true_positive(cm_abalone_naive), "F1:", evaluation.f_score(cm_abalone_naive))


# Seed ----


# Get cross validation accuracy for 5-fold cv
print("seeds validation accuracy (default parameters):")
evaluation.cross_validation(5, seeds_train_features, seeds_train_labels, model=LogisticRegression)[0]

# Grid search for optimal hyperparameters
print("seeds grid search hyperparameters:")
seeds_max_val_acc, seeds_arg_max = evaluation.grid_search(learning_rates=lrs, epsilons=eps, lambdas=lamdas, x=seeds_train_features, y=seeds_train_labels, model=LogisticRegression)

# Accuracy on test split - train with best hyperparameters
logistic_adult = LogisticRegression(seeds_train_features, seeds_train_labels)
logistic_adult.fit(lr=seeds_arg_max[0], eps=seeds_arg_max[1], regularization=seeds_arg_max[2])
seeds_prediction = logistic_adult.predict(seeds_test_features)
cm_seeds = evaluation.confusion_matrix(seeds_test_labels, seeds_prediction)
print("seeds test accuracy:")

print("Accuracy:", evaluation.accuracy(cm_seeds), "Precision:", evaluation.precision(cm_seeds), "Recall:", evaluation.true_positive(cm_seeds), "F1:", evaluation.f_score(cm_seeds))

# 5-fold CV for naive bayes
print("seeds validation accuracy (naive bayes):")
evaluation.cross_validation_naive(5, seeds_dataset.train_data, NaiveBayes, seeds_dataset.label_column,
                                                  seeds_dataset.feature_columns)

naive_seeds = NaiveBayes(seeds_dataset.train_data, seeds_dataset.label_column, continuous=seeds_dataset.feature_columns)

seeds_pred_naive = seeds_dataset.test_data.apply(naive_seeds.predict, axis=1)
cm_seeds_naive = evaluation.confusion_matrix(seeds_test_labels, seeds_pred_naive.to_numpy())
print("seeds test accuracy (Naive):")
print("Accuracy:", evaluation.accuracy(cm_seeds_naive), "Precision:", evaluation.precision(cm_seeds_naive), "Recall:", evaluation.true_positive(cm_seeds_naive), "F1:", evaluation.f_score(cm_seeds_naive))



# Adult -------

# Get cross validation accuracy for 5-fold cv
print("Adult validation accuracy (default parameters):")
evaluation.cross_validation(5, adult_train_features, adult_train_labels, model=LogisticRegression)[0]

# Grid search for optimal hyperparameters
print("Adult grid search hyperparameters:")
adult_max_val_acc, adult_arg_max = evaluation.grid_search(learning_rates=lrs, epsilons=eps, lambdas=lamdas, x=adult_train_features, y=adult_train_labels, model=LogisticRegression)

# Accuracy on test split - train with best hyperparameters
logistic_adult = LogisticRegression(adult_train_features, adult_train_labels)
logistic_adult.fit(lr=adult_arg_max[0], eps=adult_arg_max[1], regularization=adult_arg_max[2])
adult_prediction = logistic_adult.predict(adult_test_features)
cm_adult = evaluation.confusion_matrix(adult_test_labels, adult_prediction)
print("Adult test accuracy:")
print("Accuracy:", evaluation.accuracy(cm_adult), "Precision:", evaluation.precision(cm_adult), "Recall:", evaluation.true_positive(cm_adult), "F1:", evaluation.f_score(cm_adult))

# 5-fold CV for naive bayes
print("Adult validation accuracy (naive bayes):")
evaluation.cross_validation_naive(5, adult_dataset.train_data, NaiveBayes, adult_dataset.label_column,
                                                  adult_dataset.continuous_features,
                                                  adult_dataset.categorical_features,
                                                  adult_dataset.binary_features)[0]


naive_adult = NaiveBayes(adult_dataset.train_data, adult_dataset.label_column, continuous=adult_dataset.continuous_features,
                         categorical=adult_dataset.categorical_features,
                         binary=adult_dataset.binary_features)

adult_pred_naive = adult_dataset.test_data.apply(naive_adult.predict, axis=1)
cm_adult_naive = evaluation.confusion_matrix(adult_test_labels, adult_pred_naive.to_numpy())
print("Adult test accuracy (Naive):")
print("Accuracy:", evaluation.accuracy(cm_adult_naive), "Precision:", evaluation.precision(cm_adult_naive), "Recall:", evaluation.true_positive(cm_adult_naive), "F1:", evaluation.f_score(cm_adult_naive))
