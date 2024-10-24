import pandas as pd
import numpy as np

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

print(f'The least square error of this model is: {LSE}')
