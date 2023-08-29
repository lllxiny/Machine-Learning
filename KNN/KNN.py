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


# Use the iris.csv for this question. How does the performance of k-nearest neighbors change as k takes on the following values: 1, 3, 5, 7? Which of these is the optimal value of k? Which distance/similarity metric did you choose to use and why

# View the data
irisData = pd.read_csv('iris.csv')
irisData.head()


y = irisData['variety']
x = irisData[['sepal.length','sepal.width','petal.length','petal.width']]

# Split train and test dataset to calculate model accuracy later
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=111)



# Choose to use Euclidean distance as the measurement

knn1 = KNeighborsClassifier(n_neighbors=1, metric = 'minkowski', p = 2)
knn3 = KNeighborsClassifier(n_neighbors=3, metric = 'minkowski', p = 2)
knn5 = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
knn7 = KNeighborsClassifier(n_neighbors=7, metric = 'minkowski', p = 2)



knn1.fit(x_train, y_train)
knn3.fit(x_train, y_train)
knn5.fit(x_train, y_train)
knn7.fit(x_train, y_train)

y_pred_1 = knn1.predict(x_test)
y_pred_3 = knn3.predict(x_test)
y_pred_5 = knn5.predict(x_test)
y_pred_7 = knn7.predict(x_test)



df1 = pd.DataFrame(y_test)
df1.columns =['Original']
df1['KNN1 Prediction'] = y_pred_1
df1['Accuracy']=np.where(df1['Original']== df1['KNN1 Prediction'],'Yes','No')
print(df1)
print(' \n')
print("K=1, Accuracy=", accuracy_score(y_test, y_pred_1)*100)




df2 = pd.DataFrame(y_test)
df2.columns =['Original']
df2['KNN3 Prediction'] = y_pred_3
df2['Accuracy']=np.where(df2['Original']== df2['KNN3 Prediction'],'Yes','No')
print(df2)
print(' \n')
print("K=3, Accuracy=", accuracy_score(y_test, y_pred_3)*100)




df3 = pd.DataFrame(y_test)
df3.columns =['Original']
df3['KNN5 Prediction'] = y_pred_5
df3['Accuracy']=np.where(df3['Original']== df3['KNN5 Prediction'],'Yes','No')
print(df3)
print(' \n')
print("K=5, Accuracy=", accuracy_score(y_test, y_pred_5)*100)




df4 = pd.DataFrame(y_test)
df4.columns =['Original']
df4['KNN7 Prediction'] = y_pred_7
df4['Accuracy']=np.where(df4['Original']== df4['KNN7 Prediction'],'Yes','No')
print(df4)
print(' \n')
print("K=7, Accuracy=", accuracy_score(y_test, y_pred_7)*100)



print("K=1, Accuracy=", accuracy_score(y_test, y_pred_1)*100)
print("K=3, Accuracy=", accuracy_score(y_test, y_pred_3)*100)
print("K=5, Accuracy=", accuracy_score(y_test, y_pred_5)*100)
print("K=7, Accuracy=", accuracy_score(y_test, y_pred_7)*100)


#  According to the prediction accuracy calculated from test set, k=7 has the highest prediction accuracy. When k=7, the KNN method came up with 100% accurate prediction.
#  For the choice of nearest k neighbors, I used Euclidean distance in the KNN classification, because it measures the straight line direction between two observations instead of the distance of a specific direction.
#  For the choice of value k, I chose out of sample prediction accuracy as the metrics, which is a better metrics compared to in-sample prediction accuracy. By using OOS accuracy, we can choose the model that performs the best with unseen data.
#  However, as the dataset is relative small, the difference between different k values is small. Given a larger dataset, the difference should be more significant.