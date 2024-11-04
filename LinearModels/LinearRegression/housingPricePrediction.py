import pandas as pd
import numpy as np

# Using closed form solution to calculate ideal regression with lowest 
def getClosedFormLeastSquareError():
    # Data from https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    raw_housing_data = pd.read_csv('LinearRegression/boston.csv', header=None)
    data_rows = np.reshape(raw_housing_data.to_numpy(), (506, 14)) # 14 columns => 13 features, 1 target (housing price)
    data_features = data_rows[:,:13]
    housing_prices = data_rows[:,13]

    #Normalize features to zero mean and unit-variance
    meanFeatures = np.mean(data_features, axis=0) # Find mean of the columns/features
    stdFeatures = np.std(data_features, axis=0) # Find standard deviation of the columns/features
    normalizedFeaturesData = (data_features - meanFeatures)/stdFeatures

    print(normalizedFeaturesData.shape)
    print(housing_prices.shape)

    # Add column of '1' to accomodate the bias and treat affine function as linear
    columnOfOnes = np.ones((normalizedFeaturesData.shape[0], 1), dtype=normalizedFeaturesData.dtype)
    data_wb = np.hstack((normalizedFeaturesData, columnOfOnes))

    """
    Calculate model matrix w with closed-form solution for linear regression with least square error as objective funciton.
    Closed-form solution: w* = (X.T @ X)^(-1) @ X.T @ y
    """
    w = np.linalg.inv(data_wb.T @ data_wb) @ data_wb.T @ housing_prices

    # Use the weight vector (linear regression model) w on the data_wb and output the prediction vectors
    prediction = data_wb @ w

    # Compare the prediction to the real housing values using least square error formula
    LSE = (np.linalg.norm(prediction - housing_prices) ** 2)/data_rows.shape[0]

    print(f'The lease square error of this model is: {LSE}')

    return LSE

# Solve Linear regression using stochastic gradient descent
# Optimizer: Class to hold the gradient descent training parameters
class Optimizer():
    def __init__(self, lr, annealing_rate, batch_size, max_epochs):
        self.lr = lr
        self.annealing_rate = annealing_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

def linear_regression_gd(X, y, op):
    """
    Run linear regression training algorithm using stochastic gradient descent.

    Parameters
    ----------
    X : ndarray
        Input features of shape (N, d), where N is the number of samples and d is the number of features.
    y : ndarray
        Target output vector of shape (N,), where N is the number of samples.
    op : dict
        Hyper-parameters for the optimizer (e.g., learning rate, number of iterations).

    Returns
    -------
    tuple
        A tuple containing optimized parameters and loss history.
    """

    n = X.shape[0] # Number of samples
    w = np.zeros(X.shape[1]) # Initial empty lienar regression model

