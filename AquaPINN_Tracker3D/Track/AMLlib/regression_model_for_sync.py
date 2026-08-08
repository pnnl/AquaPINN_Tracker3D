from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import warnings

import numpy as np

def sync_polyfit(soa1at0, soa0, deg=2):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coef = np.polyfit(soa1at0, soa0, deg)
        fit = np.poly1d(coef)
    return fit  

def sync_polyfit_w_random_weights(soa1at0, soa0, deg=2):
    weights = np.random.rand(len(soa1at0))  # Example: Random weights, replace with your actual weights

    coef = np.polyfit(soa1at0, soa0, deg, w=weights)
    fit = np.poly1d(coef)
    return fit  


def sync_ridge(soa1at0, soa0):
    x = np.array(soa1at0).reshape(-1, 1)  # Reshape x to be a 2D array
    y = np.array(soa0)
    # Fit a Ridge regression model
    model = Ridge(alpha=10)
    model.fit(x, y)

    return model  

def sync_lasso(soa1at0, soa0):
    x = np.array(soa1at0).reshape(-1, 1)  # Reshape x to be a 2D array
    y = np.array(soa0)
    # Fit a Lasso regression model
    model = Lasso(alpha=1.0)
    model.fit(x, y)

    return model

def sync_ransac(soa1at0, soa0):
    x = np.array(soa1at0).reshape(-1, 1)  # Reshape x to be a 2D array
    y = np.array(soa0)
    # Fit a Ridge regression model
    ransac = RANSACRegressor()
    ransac.fit(x, y)
    return ransac  

def sync_Huber(soa1at0, soa0):
    x = np.array(soa1at0).reshape(-1, 1)  # Reshape x to be a 2D array
    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)  # Reshape if x is a 1D array

    y = np.array(soa0)
    # Assuming x and y are your data
    huber = HuberRegressor()
    huber.fit(x_scaled, y)
    return huber, scaler 
