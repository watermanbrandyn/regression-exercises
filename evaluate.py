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
%matplotlib inline
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
    This function takes in y and yhat (predictions) variables to create residual plots for the model and baseline regression fits.
    '''
    


def regression_errors(y, yhat):
    '''
    This function takes in y and yhat (predictions) variables and returns the sum of squared errors (SSE), explained sum of squares (ESS),
    total sum of squares (TSS), mean squared error (MSE), and root mean squared error (RMSE)
    '''


def baseline_mean_errors(y):
    '''
    This function takes in y variables and returns SSE, MSE, and RMSE for the baseline
    '''


def better_than_baseline(y, yhat):
    '''
    This function takes in y and yhat (predictions) variables and returns assessment of model performance compared to baseline.
    '''



