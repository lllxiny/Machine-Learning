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


# # Logistic Regression

# 1. Using hotel_cancellation.csv Download hotel_cancellation.csv, write code to estimate the treatment effects if a ‘different room is assigned’ as the treatment indicator and interpret its effect on the room being ‘canceled’. Use all the other columns as the covariates. Write your observations for the results.

# View the data
hotelData = pd.read_csv('hotel_cancellation.csv')
hotelData.head()


# Change the T/F columns to numeric values
hotelData['const'] = 1
hotelData['different_room_assigned'] = (hotelData['different_room_assigned'] == True).astype(int)
hotelData['is_canceled'] = (hotelData['is_canceled'] == True).astype(int)

# Check the new values
hotelData.head()



# Create a logistic regression model with 'is_canceled' as dependent variable, 'different_room_assigned' as treatment indicator, and all other variables as covariates
logit = LogisticRegression(solver='liblinear',random_state=0)
y = hotelData['is_canceled']
x = hotelData[['const','different_room_assigned','lead_time','arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'days_in_waiting_list']]

one_stage = sm.Logit(y,x).fit()
print(one_stage.summary())



print(one_stage.params)


#  

#  

# The logistic regression model has 'is_canceled' as the dependent variable, 'different_room_assigned' as the treatment indicator, and the other 5 variables as covaraites.  Customers with different rooms assigned are of the treatment group, and of the control group if otherwise.
#     
# The model has a pseudo R^2 of 0.1053. All variables except 'arrival_date_day_of_month' are significant at 95% confidence level.   
#   
#     
#     
# Observation about the treatment indicator:
#     
# The treatment indicator 'different_room_assigned' is significant and has a coefficient of -2.55. It shows that having a different room assigned decreases the cancellation rate.
#     
#     
#     
#     
# Other observations:
#     
# Besides, according to the regression result, 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month' all have negative coefficients. This shows that arrival time might also have some effect on cancellation.
#     
# 'lead_time' and 'days_in_waiting_list' both have positive coefficients. It means that longer waiting time in general increases cancellation, which is reasonable.

#  

# 2. For hotel_cancellation.csv Download hotel_cancellation.csv, now use double logistic regression to measure the effect of ‘different room is assigned’ on the room being ‘canceled’.


# Regress all other variables on different_room_assigned, d=d(x)^+nu
logit = LogisticRegression(solver='liblinear',random_state=0)
y = hotelData['different_room_assigned']
x = hotelData[['const','lead_time','arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'days_in_waiting_list']]
               
confounder_control_1 = sm.Logit(y,x).fit()
print(confounder_control_1.summary())


print(confounder_control_1.params)


#   
#  Model's pseudo R^2 is 0.02975, which is at a reasonable level.  
#     
#  All variables are significant at 95% confidence level.  
#  
#  Variables with negative coefficients: lead_time, arrival_date_year, arrival_date_day_of_month. 
#  
#  Variables with postivie coefficients: arrival_date_week_number and days_in_waiting_list.  
#   


# Predict d(x)^ with the confounder_control_1 model
hotelData['predicted_different_room'] = confounder_control_1.predict()
hotelData


# regress 'is_canceled' on 'different_room_assigned' (d), 'predicted_different_room'(d(x)^), and all of the other covariates

y = hotelData['is_canceled']
x = hotelData[['const','different_room_assigned','predicted_different_room','lead_time','arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'days_in_waiting_list']]

confounder_control_2 = sm.Logit(y,x).fit()
print(confounder_control_2.summary())
print(confounder_control_2.params)


#  

#  

#  In the model, we try to find whether a 'different_room_assigned' influences whether there is a cancellation or not. 
#  
#  According to the model, after controlling for confounding effects of the covariates, the variable of 'different_room_assigned' is significant and has a negative coefficient. The value of the coefficient decreases slightly from -2.55 to -2.54.
#  
#  
#  We can keep the previous conclusion that being assigned to a different room (treated) causes the cancellation rate to decrease.
