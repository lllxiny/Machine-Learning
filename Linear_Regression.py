#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd
pd.options.mode.chained_assignment = None

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression import linear_model

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, explained_variance_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns


# # Part 1 Linear Regression Model for prediction

# Using the sales.csv, write code to show effects of interactions, if any, on the linear regression model to predict the total_sales for a new area using given sales from three areas.


# View the data
q1Data = pd.read_csv('sales.csv')
q1Data.head()


# Describe the data 
q1Data.describe()


# ## Fit a no-interaction model for reference

# Define x and y
x = q1Data.iloc[:, 1:-1]
y = q1Data.iloc[:, -1]

# Add a constant to x1_train
x1 = sm.add_constant(x)

# Split data to train and test
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y) 


# Fit the model
q1_no_interaction_model = linear_model.OLS(y1_train,x1_train).fit()
print(q1_no_interaction_model.summary())



# Predict the test data. Similar to in sample R²
q1_no_interaction_model_predicted = q1_no_interaction_model.predict(x1_test)
print(explained_variance_score(y1_test, q1_no_interaction_model_predicted,force_finite=True))


# 

#  

# ## Fit a model with interactions between each two areas


# Build interaction dataframe

x_interaction = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(x)
interaction_df1 = pd.DataFrame(x_interaction, columns=['area1_sales','area2_sales','area3_sales','area1_sales:area2_sales','area1_sales:area3_sales','area2_sales:area3_sales'])


# Add a constant to inateraction_df1
interaction_df1 = sm.add_constant(interaction_df1)

# Split data to train and test
x1_int_train, x1_int_test, y1_int_train, y1_int_test = train_test_split(interaction_df1, y) 


# Fit the model
q1_interaction_model = linear_model.OLS(y1_int_train,x1_int_train).fit()
print(q1_interaction_model.summary())


# Predict the test data
q1_interaction_model_predicted = q1_interaction_model.predict(x1_int_test)
print(explained_variance_score(y1_int_test, q1_interaction_model_predicted,force_finite=True))


#  

# Compared to the model without interaction, including interaction terms improved both the in-sample R2 and explained variance score when used to predict y_test.

#   

#  

#  

# # Part 2

# Develop a full Logistic Regression Model using customer.csv to predict whether the customer will purchase the product. Also train trimmed logistic regression models (Trimmed over features in the data). Compute the "in-sample R2" (pseudo) for the models you train and compare the models based on this metric.



# View the data
q2Data = pd.read_csv('customer.csv')
q2DataDF = pd.DataFrame(q2Data)
q2DataDF.head()



# Create dummy for Gender variable

q2DataDF["GenderNum"] = np.where(q2DataDF["Gender"]=="Male",1,0)
q2DataDF.describe()


# ## Full model, without interaction, without spliting the data


# Full model including all three variables

logit = LogisticRegression(solver='liblinear',random_state=0)
x2 = q2DataDF[["GenderNum","Age","EstimatedSalary"]]
y2 = q2DataDF[["Purchased"]]



q2Model1 = sm.Logit(y2.values.ravel(),sm.add_constant(x2)).fit()
print(q2Model1.summary())


#   

# Pseudo R-squ.: 0.4711. Variable 'Gender' not significant

#  

#  

#  

# ## Trimmed model, without interaction, without spliting the data

# Trim the model by dropping the 'Gender' variable which was not significant



x2_1 = q2DataDF[["Age","EstimatedSalary"]]


q2Model1_trimmed = sm.Logit(y2.values.ravel(),sm.add_constant(x2_1)).fit()
print(q2Model1_trimmed.summary())


# Pseudo R-squ.: 0.4688. Not much difference from the model with all three variables. All variables in the model are significant.

# Based on psuedo R2 , full model with all 3 variables has higher pseudo R2, so is slightly better than the trimmed model.

# >But the difference is trivial, showing that adding Gender dummy does not improve the model a lot.

#  

#  

#  

# ## Full model with interaction and without spliting train/test data



# Build interaction dataframe
x2_interaction = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(x2)
interaction_df2 = pd.DataFrame(x2_interaction, columns = ['GenderNum','Age','EstimatedSalary','GenderNum:Age','GenderNum:EstimatedSalary','Age:EstimatedSalary'])


q2_Model_interaction_1 = sm.Logit(y2.values.ravel(),sm.add_constant(interaction_df2)).fit()
print(q2_Model_interaction_1.summary())


# Pseudo R-squ.: 0.5968. Improved from both full and trimmed model without interaction. 

# Gender variable and interaction terms with Gender are all not significant.

#  

#  

#  

#  

# ## Trimmed model with interaction and without spliting train/test data


# Build interaction dataframe with Gender trimmed
x2_interaction_1 = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(x2_1)
interaction_df2_1 = pd.DataFrame(x2_interaction_1, columns = ['Age','EstimatedSalary','Age:EstimatedSalary'])



# Fit the model
q2_interaction_model_1_1 = sm.Logit(y2,sm.add_constant(interaction_df2_1)).fit()
print(q2_interaction_model_1_1.summary())


# Pseudo R-squ.:0.5905, very small decrease from the full model with interaction.

#  

#  

#  

# # Part 3

# For the Logistic Regression models trained above, pick the best model wrt to the in-sample R2 and give your interpretation of the model’s coefficients (For example, what effect does a positive or negative coefficient have on the model and so on).

# In-sample pseudo R-square of 4 models trained:

# Full model without interaction terms: 0.4711  
# Trimmed model without interaction terms: 0.4688  
# Full model with interaction terms: 0.5968  
# Trimmed model with interaction terms: 0.5905

# Based on in-sample R2, either with or without interaction, we should choose the full model.  
# The model with all three variables and all three interaction terms has the highest pseudo R2.

#  

#  

#  

# ## Interpreting the model with Gender, Age, Estimated Salary


q2Model1.summary()


# Full model without interaction terms:  
# logit(purchased) = -12.7836 + 0.3338 * Gender + 0.237 * Age + 3.644e-05 * Estimated Salary

#  



# Calculate odds ratios with coefficients
odds_q1_1 = np.exp(q2Model1.params)
odds_q1_1


# Interpreting the odds ratios:  
# for 'Gender' variable, men are 1.4 times more likely to purchase than women;  
# for 'Age' variable, with each unit increase in age, the person is 1.27 times more likely to purchase;  
# for 'Estimated Salary' variable, an odds ratio of 1 shows that there estimated salary does not influence purchase decision very much.

#  

#  

#  

# ## Interpreting the model with Gender, Age, Estimated Salary, and their interactions

# Full model without interaction terms:  
# logit(purchased) = -37.037282 - 0.603125 * Gender + 0.787453 * Age + 0.000310 * Estimated Salary + 0.026411 * Gender:Age + 0.000001 * Gender:EstimatedSalary - 0.000006 * Age:EstimatedSalary

# 


q2_Model_interaction_1.summary()


# Full model with interaction terms:  
# logit(purchased) = -37.0373 - 0.6031 * Gender + 0.7875 * Age + 0.0003 * Estimated Salary + 0.0264 * (GenderNum:Age) + 1.443e-06 * (GenderNum:EstimatedSalary) - 6.337e-06 * (Age:EstimatedSalary)

# 


odds_q1_2 = np.exp(q2_Model_interaction_1.params)
odds_q1_2


# Interpreting the odds ratios:  
# for 'Gender' variable, men are 0.55 times more likely to purchase than women;  
# for 'Age' variable, with each unit increase in age, the person is 2.2 times more likely to purchase;  
# for 'Estimated Salary' variable and all of the interaction terms, an odds ratio of 1 shows that there estimated salary does not influence purchase decision very much.  
#       
#      
#     
# Adding the interaction terms does make a difference to the odds ratio of the other three variables, but the interactions themselves do not make much difference to the odds ratio.

#  

#  

#  

# # Part 4

# Is accuracy a good metric to judge the above model? Give reasons and alternatives to support your answer.

# ## Train and test the full no interaction models



x2_const = sm.add_constant(x2)
y2 = q2DataDF[["Purchased"]]
x2_noint_train, x2_noint_test, y2_noint_train, y2_noint_test = train_test_split(x2_const, y2) 

# Fit the model
q2_no_interaction_model1 = sm.Logit(y2_noint_train,x2_noint_train).fit()
print(q2_no_interaction_model1.summary())


y_predicted_no_1 = q2_no_interaction_model1.predict(x2_noint_test)

# Accuracy
accuracy_score(y2_noint_test, list(map(round, y_predicted_no_1)))



# Classification reprt
print(classification_report(y2_noint_test, list(map(round, y_predicted_no_1))))



# Confusion matrix
confusion_matrix(y2_noint_test, list(map(round, y_predicted_no_1)))


#  

# ## Train/test the full interaction models



# Add a constant to inateraction_df2
interaction_df2 = sm.add_constant(interaction_df2)

# Split data to train and test
x2_int_train, x2_int_test, y2_int_train, y2_int_test = train_test_split(interaction_df2, y2) 

# Fit the model
q2_interaction_model1 = sm.Logit(y2_int_train,x2_int_train).fit()
print(q2_interaction_model1.summary())



y_predicted_1 = q2_interaction_model1.predict(x2_int_test)

# Accuracy
accuracy_score(y2_int_test, list(map(round, y_predicted_1)))




# Classification reprt
print(classification_report(y2_int_test, list(map(round, y_predicted_1))))




# Confusion matrix
confusion_matrix(y2_int_test, list(map(round, y_predicted_1)))


#  

# Full model with interactions has higher accuracy than full model without interactions, which is in align with pseudo R2. With current data and models, accuracy works fine.
#       
# Accuracy might not be a good metric to use when the data is imbalanced, because it calculates on the proportion of True Positives out of all Positives. Imbalanced data might cause more False Negatives, but this is not shown by Accuracy.  
#     
# In such cases, classification reports and confusion matrix can help as they evaluate the models more comprehensively.

#  

# # Part 5

# Plot the interactions of the ‘Age’ and ‘Gender’ features with the ‘Purchased’ output. 


sns.pointplot(data=q2DataDF, x="Purchased", y="Age", hue="Gender")


# According to the plot, there is an interaction between Gender and 


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 

q2Model2 = sm.Logit(y_train.values.ravel(),sm.add_constant(x_train)).fit()
q2Model2.summary()

ro_scaler = RobustScaler()
x_train_scaled = ro_scaler.fit_transform(x_train)
X_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)


y_train_scaled = ro_scaler.fit_transform(y_train)
q2Model3 = sm.Logit(y_train_scaled,sm.add_constant(x_train_scaled)).fit()
q2Model3.summary()


q2Model4 = sm.Logit(y_train.values.ravel(),x_train).fit()
q2Model4.summary()