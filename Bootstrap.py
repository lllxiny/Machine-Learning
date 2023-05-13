#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression import linear_model
from linearmodels.iv import IV2SLS

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, explained_variance_score, accuracy_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsIC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import seaborn as sns
import plotnine as p



# Use bootstrap to estimate the standard error of the treatment effects measured in (2).


# times of bootstrap
n_resamples = 1000

# create the matrix to store the TE estimates
treatment_effects = np.zeros((n_resamples,confounder_control_2.params.shape[0]-1))

# Use bootstrapping to estimate std error of TE
i = 0

while i < n_resamples:
    resample_index = np.random.choice(hotelData.index, size = hotelData.index.size, replace = True)
    resample = hotelData.iloc[resample_index]
    x_resample = x.iloc[resample_index]
    y_resample = y.iloc[resample_index]
    model1 = sm.Logit(y_resample, x_resample).fit()
    y_hat = np.array(model1.predict(x_resample)).reshape(len(hotelData['predicted_different_room']), 1)
    x_new = np.hstack((x_resample, y_hat))
    model2 = sm.Logit(y_resample, x_new).fit()
    treatment_effects[i, :] = confounder_control_2.params[:-1]
    i += 1


treatment_effects


# Calculate the standard error of the treatment effects
treatment_effects_se = treatment_effects.std(axis=0)

# Print the standard errors of the treatment effect estimates
print('Standard errors of the treatment effects:')
print(treatment_effects_se)