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


# # Regression Discontinuity Design (RDD)

# Use the drinking.csv Download drinking.csvdataset for this question.  
# Keeping 21 as the threshold for age, explore the data with an RDD by writing very simple code (no package needed, just average to one side of the threshold minus average to the other side) to determine if alcohol increases the chances of death by accident, suicide and/or others (the three given columns) and comment on the question “Should the legal age for drinking be reduced from 21?” based on the results.   
# Plot graphs to show the discontinuity (if any) and to show results for the change in chances of death with all the three features (i.e., accident vs age, suicide vs age and others vs age). For this problem, choose the bandwidth to be 1 year (i.e., 21 +- 1). What might be the effect of choosing a smaller bandwidth?  What if we chose the maximum bandwidth?


# View the data
drinkingData = pd.read_csv('drinking.csv')
drinkingData.head()


# Create a tag column for grouping
drinkingData['legal_to_drink']=np.where(drinkingData['age']>=21 ,'Yes','No')
# Check the dataframe with the new tag column
drinkingData.head()


# ## With bandwidth of 1

# filter the data with bandwidth of 1
illegal = ((drinkingData[['age']] >= 20) & (drinkingData[['age']] < 21))['age']
legal = ((drinkingData[['age']] >= 21) & (drinkingData[['age']] <= 22))['age']
legal = drinkingData.loc[legal]
illegal = drinkingData.loc[illegal]


# Calculate the means by groups

i111 = legal['accident'].mean()
i112 = legal['suicide'].mean()
i113 = legal['others'].mean()

i211 = illegal['accident'].mean()
i212 = illegal['suicide'].mean()
i213 = illegal['others'].mean()

# Create a df to store results and differences

# initialize data of lists.
data1 = {'Yes': [i111,i112,i113],
        'No': [i211, i212, i213]}
  
# Create pandas df

diff_df1 = pd.DataFrame(data1, index=['Accident',
                                   'Suicide',
                                   'Others'])
diff_df1['Difference'] = diff_df1['Yes']-diff_df1['No']
diff_df1


#  With a bandwidth of 1:
#  According to the comaprison, passing the age of 21 and becoming legal to drink increases the death rate caused by all three reasons.   
#  The increase of 'Accident' is relatively small (+1.33), the increase in 'Suicide' is slightly larger (+1.7) and the increase in 'other' is the largest (+6.5).
#  Thus, based on the result, the legal age for drinking shoud NOT be reduced from 21.




p.ggplot(drinkingData, p.aes(x='age', y='suicide', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying of Suicide (Y1)")+ p.scales.xlim(20,22

p.ggplot(drinkingData, p.aes(x='age', y='accident', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying in Accident (Y1)") + p.scales.xlim(20,22)

p.ggplot(drinkingData, p.aes(x='age', y='others', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying of other reasons (Y1)")+ p.scales.xlim(20,22)

# ## With maximized bandwidth

# Check the avg age on both sides
drinkingData.groupby('legal_to_drink')['age'].mean()

# Calculate the means by groups

i11 = drinkingData[drinkingData['legal_to_drink'] == 'Yes'].groupby('legal_to_drink')['accident'].mean()
i12 = drinkingData[drinkingData['legal_to_drink'] == 'Yes'].groupby('legal_to_drink')['suicide'].mean()
i13 = drinkingData[drinkingData['legal_to_drink'] == 'Yes'].groupby('legal_to_drink')['others'].mean()

i21 = drinkingData[drinkingData['legal_to_drink'] == 'No'].groupby('legal_to_drink')['accident'].mean()
i22 = drinkingData[drinkingData['legal_to_drink'] == 'No'].groupby('legal_to_drink')['suicide'].mean()
i23 = drinkingData[drinkingData['legal_to_drink'] == 'No'].groupby('legal_to_drink')['others'].mean()

# Create a df to store results and differences

# initialize data of lists.
data = {'Yes': [i11[0],i12[0],i13[0]],
        'No': [i21[0], i22[0], i23[0]]}
  
# Create pandas df

diff_df = pd.DataFrame(data, index=['Accident',
                                   'Suicide',
                                   'Others'])
diff_df['Difference'] = diff_df['Yes']-diff_df['No']
diff_df


# According to the comaprison, passing the age of 21 and becoming legal to drink decreases the chance of dying of accident, but increases the chance of dying of suicide and other reasons. 
# As the decrease in chance of dying of accident caused by becoming legal to drink is relatively small (-1.7) compared to the increase caused by the reason in chance of suicide death (+1.4) and other deaths (+5.7).



p.ggplot(drinkingData, p.aes(x='age', y='accident', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying in Accident (Y1)")

p.ggplot(drinkingData, p.aes(x='age', y='suicide', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying of Suicide (Y1)")

p.ggplot(drinkingData, p.aes(x='age', y='others', color = 'legal_to_drink')) +    p.geom_point(alpha = 0.5) +    p.geom_vline(xintercept = 21, colour = "grey") +    p.stat_smooth(method = "lm", se = 'F') +    p.labs(x = "Age (X)", y = "Chance of dying of other reasons (Y1)")


#  As can be seen from the two sets of results and plots, when the bandwidth is smaller, the observations show up on both sides of the cutoff value can be regarded as being the same in all characteristics besides whether passing the age of 21 years old or not. The plot shows a clear treatment effect of becoming legal to drink on increasing chances of death. 
#  If the width is expanded, and observations located further from the cutoff are also included, the observations might start to differ in other characteristics. Some confounding characteristics might cause the treatment effect to be not as obvious as the previous three plots. 
