import numpy as np

"""In logistics regression, we have the following components:

The LogisticRegression class, which takes three arguments: the learning rate (alpha), the number of iterations to run gradient descent for, and whether or not to fit an intercept term.
The add_intercept function, which adds an intercept term (a column of ones) to the input data X.
The sigmoid function, which computes the sigmoid function.
The score_function function, which computes the predicted probabilities for each data point in X using the sigmoid function and the current model parameters theta.
The cost_function function, which computes the binary cross-entropy loss function for the current model parameters theta and the input data X and labels y.
The gradient_descent function, which performs gradient descent to update the model parameters theta.
The fit function, which trains the model by running gradient descent on the input data X and labels y.
The predict function, which predicts the binary labels for new input data X using the trained model parameters theta."""

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def score_function(self, X, theta):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.sigmoid(np.dot(X, theta))

    def cost_function(self, X, y, theta):
        h = self.score_function(X, theta)
        m = X.shape[0]
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient_descent(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        m, n = X.shape
        theta = np.zeros(n)
        for i in range(self.num_iterations):
            h = self.score_function(X, theta)
            gradient = np.dot(X.T, (h - y)) / m
            theta -= self.learning_rate * gradient
        return theta

    def fit(self, X, y):
        self.theta = self.gradient_descent(X, y)

    def predict(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        return np.round(self.score_function(X, self.theta))

