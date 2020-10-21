import numpy as np
import pandas as pd


class NaiveBayes:
    # Initialize with features and labels
    def __init__(self, df, label, continuous=[], categorical=[], binary=[]):
        self.label = label,
        self.continuous = continuous
        self.categorical = categorical
        self.binary = binary

        self.continuous_df = df[continuous].copy(deep=True)
        self.categorical_df = df[categorical].copy(deep=True)
        self.binary_df = df[binary].copy(deep=True)
        self.label_df=df[label].copy(deep=True)

        num_samples = self.label_df.shape[0]

        # only 2 classes, 0 and 1
        num_classes = 2
        self._classes = [0, 1]

        self.mean = np.zeros((num_classes, self.continuous_df.shape[1]))
        self.var = np.zeros((num_classes, self.continuous_df.shape[1]))
        self.priors = np.zeros(num_classes)
        self.likelihoods = [0, 1]



        for s in self._classes:

            rows = self.label_df == s
            class_count = rows[rows].index.values.shape[0]
            # Get the features only of this label

            cont = self.continuous_df.loc[rows]
            cat = self.categorical_df.loc[rows]
            bin = self.binary_df.loc[rows]

            # Calculate the means, var, prior and probabilities
            self.mean[s, :] = cont.mean()
            self.var[s, :] = cont.var()
            self.likelihoods[s] = {}

            # Prob of categorical # occurences/total
            for col in cat.columns:
                unique = cat[col].value_counts()
                unique = unique.apply(lambda x: x/class_count)
                self.likelihoods[s][col] = unique

            # Prob of binary
            for col in bin.columns:
                unique = bin[col].value_counts()
                unique = unique.apply(lambda x: x / class_count)
                self.likelihoods[s][col] = unique



            self.priors[s] = class_count / float(num_samples)


    def max_posterior(self, cont=pd.Series(), cat=pd.Series(), bin=pd.Series()):
        posteriors = []
        # probability for each class
        for i in self._classes:
            prior = np.log(self.priors[i])
            # sum of all the n feature probabilities
            cont_likelihood = np.log(self.normal(i, cont))
            cont_likelihood = np.sum(cont_likelihood[~np.isnan(cont_likelihood)])
            cat_likelihood = np.sum(np.log(self.categorical_likelihood(i, cat)))
            bin_likelihood = np.sum(np.log(self.categorical_likelihood(i, bin)))
            likelihood = cont_likelihood + cat_likelihood + bin_likelihood
            posterior = prior + likelihood
            # posteriors contain the probabilities of the two labels for each instance
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def normal(self, label_idx, x):
        x = x.to_numpy(dtype=float)
        # Choosing the statistics according to the corresponding label
        mean = self.mean[label_idx]
        var = self.var[label_idx]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def categorical_likelihood(self, label_idx, x):

        prob = []
        for col in x.index:
            df = self.likelihoods[label_idx][col]
            if x[col] in df:
                prob.append(df[x[col]])
            else:
                prob.append(0)
        return prob

    def predict(self, x):
        # Split dataframe into cont, cat, bin
        cont = x[self.continuous]
        cat = x[self.categorical]
        bin = x[self.binary]
        return self.max_posterior(cont, cat, bin)
