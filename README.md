# Logistic-regression-and-naive-Bayes-from-Scratch
Explored 4 dataset - Ionosphere (Sigillito et al., 1989), Adult (Kohavi, 1996), Abalone (Nash et al., 1994) and Seeds (Charytanowicz et al., 2010)

Completed for COMP 551 - Applied Machine Learning

## Data Exploration Analysis

In this phase the data are cleaned. One-hot encoding is applied using pandas get_dummies method, the continuous data are standardized, other features in adult dataset has been remaped into binary ('rich' higher or lower than 50K salary, 'sex' {'Male': 1, 'Female': 0}), and finally the training and test are split. The files are located in dataset folder.

### Histograms of Binary and Continuous Features and Labels in the Adult Data Set

![adult dist binary cont](https://user-images.githubusercontent.com/57273222/96785151-638e9480-13bc-11eb-8d76-c46b2ab6cf41.png)

### Histograms of Categorical Features in the Adult Data Set

![adult dist categorical](https://user-images.githubusercontent.com/57273222/96785278-93d63300-13bc-11eb-8ee6-992d92e39ecd.png)

### Histograms of Features and Labels in the Abalone Data Set 

![abalone dist](https://user-images.githubusercontent.com/57273222/96785457-d26bed80-13bc-11eb-9897-da35fa20bab4.png)

### Histograms of Features and Labels in the Seeds Data Set

![seeds dist](https://user-images.githubusercontent.com/57273222/96785532-eadc0800-13bc-11eb-93fc-6cc197bdfb3c.png)

## Grid Search

A grid search has been performed to find the best hyperparameters. It is located in evaluation.py

The naive Bayes and Logistic model will use the hyperparameters: 
```
lamdas = [0, 0.01, 0.05, 0.1, 0.5, 1]
lrs = [0.05, 0.1, 0.5, 1]
eps = [0.01, 0.05, 0.1, 0.5]
```

lamdas is the L2 regularization, lrs is the learning rate, and eps is the termination condition for the norm gradient.

## Experiment Results
### Ionosphere - Accuracy over the number of iterations of gradient descent with different
learning rates
![exp2](https://user-images.githubusercontent.com/57273222/96787907-9044ab00-13c0-11eb-80ef-48f877e79f84.png)

### Adult - Accuracy over the number of iterations of gradient descent with different learning
rates
![exp2 adult](https://user-images.githubusercontent.com/57273222/96787934-9dfa3080-13c0-11eb-970d-3157b1212383.png)

### Abalone - Accuracy over the number of iterations of gradient descent with different
learning rates
![exp2 abalone](https://user-images.githubusercontent.com/57273222/96787977-ad797980-13c0-11eb-9898-faebf6c2bbd1.png)

### Seeds - Accuracy over the number of iterations of gradient descent with different learning
rates
![exp2 seeds](https://user-images.githubusercontent.com/57273222/96788004-b8340e80-13c0-11eb-9719-01976c6620bd.png)

### Accuracy of both models on all data set 
![exp3_all](https://user-images.githubusercontent.com/57273222/96789288-e1ee3500-13c2-11eb-8d94-0d42eeabe2e4.PNG)

