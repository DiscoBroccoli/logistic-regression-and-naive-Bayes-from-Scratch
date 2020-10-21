import numpy as np


class LogisticRegression:

    # Initialize with features and labels
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.w = np.ones(X.shape[1])
        self.N = X.shape[0]

    # average cost function
    # w is the weights
    # X is the design matrix
    # y is the labels
    def cost(self):
        z = np.dot(self.X, self.w)
        J = np.mean(self.Y * np.log1p(np.exp(-z)) + (1 - self.Y) * np.log1p(np.exp(z)))
        return J

    # logistic function on x
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    # gradient of logistic regression
    def gradient(self, regularization):

        # Calculate y_hat based on logistic func
        y_hat = self.logistic(np.dot(self.X, self.w))

        # Gradient of logistic descent
        gradient = np.dot(self.X.T, y_hat - self.Y) / self.N

        # L2 regularization
        gradient[1:] += regularization * self.w[1:]

        return gradient

    # lr is the learning rate
    # epochs is the number of iterations
    # w is the weights to be learned, default value is 0 array
    # eps is the termination condition, when the norm of the gradient is less than the value
    def fit(self, lr=0.5, epochs=10000, regularization=0, verbose=False, eps=1e-2):

        # Iterate epochs times, performing gradient descent
        for epoch in range(epochs):
            gradient = self.gradient(regularization)

            # go down the gradient with given learning rate
            self.w -= lr * gradient

            # print details about the training if needed
            if verbose:
                # every 1000 epochs, print the cost
                if epoch % 1000 == 0:
                    print('iter: ', epoch, ' cost: ', self.cost())

            if eps != None and np.linalg.norm(gradient) < eps:
                #print('Reached min cost.')
                break

        return self.cost()

    # X is a feature instance, default decision boundary = 0.5
    def predict(self, X, boundary=0.5):

        return (self.logistic(np.dot(X, self.w)) > boundary).astype(int)
