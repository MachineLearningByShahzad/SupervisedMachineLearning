import numpy as np
from numpy.linalg import inv
from scipy import stats

"""The LinearRegression class has four methods:

1) fit(X, y): Fits the model to the training data X and y.
2) predict(X): Predicts the output for the input data X.
3) cost(X, y): Calculates the mean squared error cost function for the input data X and y.
4) gradient_descent(X, y, alpha, iterations): Performs gradient descent to optimize the coefficients for the input data X and y.

Note that the implementation uses the closed-form solution to calculate the coefficients, which is more computationally 
efficient than gradient descent for small datasets. However, the gradient_descent method is included for demonstration purposes."""


class LinearRegressionModel:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for intercept
        X = np.insert(X, 0, 1, axis=1)

        # Calculate coefficients using closed-form solution
        self.coefficients = inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        # Add a column of ones to X for intercept
        X = np.insert(X, 0, 1, axis=1)

        # Calculate predictions using coefficients
        return X.dot(self.coefficients)

    def cost(self, X, y):
        # Add a column of ones to X for intercept
        X = np.insert(X, 0, 1, axis=1)

        # Calculate mean squared error
        predictions = X.dot(self.coefficients)
        mse = np.mean((predictions - y) ** 2)

        return mse

    def gradient_descent(self, X, y, alpha=0.01, iterations=1000):
        # Add a column of ones to X for intercept
        X = np.insert(X, 0, 1, axis=1)

        # Initialize coefficients to zeros
        self.coefficients = np.zeros(X.shape[1])

        # Loop through iterations and update coefficients
        for i in range(iterations):
            predictions = X.dot(self.coefficients)
            errors = predictions - y
            gradient = X.T.dot(errors) / len(X)
            self.coefficients -= alpha * gradient

    def score(self, y_true, y_pred):
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        # Compute the R-squared value
        r2 = 1 - (ss_res / ss_tot)

        return r2

