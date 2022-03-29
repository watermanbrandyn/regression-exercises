# os functions
import os

# local files
from env import host, user, password
import wrangle as w

# df manipulations
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# stats and math
from scipy import stats
from math import sqrt

# splitting, scaling, and imputing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Regression metric calculations
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)


def plot_residuals(y, yhat):
    '''
    This function takes in y and yhat (predictions) variables to create a residual plots for baseline and model. 
    '''
    # Establish the baseline target value
    baseline = y.mean()
    # Compute the baseline residual
    baseline_residual = baseline - y
    # Calculate yhat (predictions) residuals
    yhat_residual = yhat - y

    # Creation of scatterplot using 
    plt.figure(figsize=(13,7))

    # rows, cols, placement
    # one row, two columns, first plot
    # Baseline Graph
    plt.subplot(121)
    plt.scatter(y, baseline_residual)
    plt.axhline(y=0, ls=':')
    plt.xlabel('Tip')
    plt.ylabel('Residual')
    plt.title('Baseline Residuals')
    # Model Graph
    plt.subplot(122)
    plt.scatter(y, yhat_residual)
    plt.axhline(y=0, ls=':')
    plt.xlabel('Tip')
    plt.ylabel('Residual')
    plt.title('OLS model Residuals')


def regression_errors(y, yhat):
    '''
    This function takes in y and yhat (predictions) variables and returns the sum of squared errors (SSE), explained sum of squares (ESS),
    total sum of squares (TSS), mean squared error (MSE), and root mean squared error (RMSE)
    '''
    # Calculate yhat (predictions) residuals
    yhat_residual = yhat - y

    # Sum of Squared Errors
    SSE = sum(yhat_residual**2)
    # Explained sum of squares
    ESS = sum((yhat - y.mean())**2)
    # Total sum of squares
    TSS = ESS + SSE
    # Mean Squared Error
    MSE = mean_squared_error(y, yhat)
    # Root mean squared error
    RMSE = sqrt(MSE)
    # print(SSE)
    # print(ESS)
    # print(TSS)
    # print(MSE)
    # print(RMSE)
    # Return SSE, ESS, TSS, MSE, RMSE
    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    '''
    This function takes in y variables and returns SSE, MSE, and RMSE for the baseline
    '''
    # Establish the baseline target value
    baseline = np.full_like(y, y.mean())
    # Compute the baseline residual
    baseline_residual = baseline - y

    # Sum of Squared Errors
    SSE_base = sum(baseline_residual**2)
    # Mean Squared Error
    MSE_base = mean_squared_error(y, baseline)
    # Root mean squared error
    RMSE_base = sqrt(MSE_base)

    # Return SSE, MSE, and RMSE
    return SSE_base, MSE_base, RMSE_base 


def better_than_baseline(y, yhat):
    '''
    This function takes in y and yhat (predictions) variables and returns assessment of model performance compared to baseline.
    Using the SSE metric.
    '''
    sse_base = ((y - y.mean()) ** 2).sum()
    sse_model = ((y - yhat) ** 2).sum()
    return sse_model < sse_base
    



