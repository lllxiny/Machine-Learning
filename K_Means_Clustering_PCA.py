#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression import linear_model
from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, explained_variance_score, accuracy_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLarsIC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as p


# # Pt 1

# ## What preprocessing techniques did you apply, if any?



# View the data
data = pd.read_csv('madelon.csv')
data.head()




# check if scaling is needed
data.describe()



# using Shapiro-Wilk test to check normality

from scipy.stats import shapiro

for col in data.columns:
    stat, p = shapiro(data[col])
    print(col, 'Shapiro-Wilk test statistic:', stat, 'p-value:', p)
    if p > 0.05:
        print('The data in column', col, 'are likely normally distributed.')
    else:
        print('The data in column', col, 'are likely not normally distributed.')





# use IQR to count the number of outliers

for col in data.columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_indices = (data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)
    outliers = data.loc[outlier_indices, col]
    print(col, 'has', len(outliers), 'outliers')


#  The data is very different in mean, std, range. Most of the columns have data that does not follow normal distribution according to Shapiro-Wilk normality test. 
#  Further checks of outliers using IQR method shows that most of the columns have less than 5% of the data as outliers, which is a relatively smaller proportion. So I choose to keep the outliers in the data. 
#  But scale matter in k-means clustering because it is based on the distance. Thus, the dataset should be scaled before doing k-means clustering.
#  As the dataset has some outliers, I choose RobustScaler to scale the data.


# scale the data
scale = RobustScaler()
data_scaled = pd.DataFrame(scale.fit_transform(data))

# checking the scaled data
data_scaled.describe()


# ## Describe qualitatively: how does the number of clusters affect the performance of the clustering algorithm on the dataset?


# ### Running the clustering with k=4


# run k means clusterings with k=4
kmeans4 = KMeans(init="random",n_clusters=4, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans4.n_iter_


# lowest sse
kmeans4.inertia_


# ### Running the clustering with k=8

# run k means clusterings with k=8
kmeans8 = KMeans(init="random",n_clusters=8, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans8.n_iter_


# lowest sse
kmeans8.inertia_


# ### Running the clustering with k=16


# run k means clusterings with k=16
kmeans16 = KMeans(init="random",n_clusters=16, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans16.n_iter_



# lowest sse
kmeans16.inertia_


# ### Running the clustering with k=32


# run k means clusterings with k=32
kmeans32 = KMeans(init="random",n_clusters=32, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans32.n_iter_


# lowest sse
kmeans32.inertia_


# ### Running the clustering with k=64


# run k means clusterings with k=64
kmeans64 = KMeans(init="random",n_clusters=64, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans64.n_iter_


# lowest sse
kmeans64.inertia_


# ### Qualitatively describe how the value of k affects the clustering algorith


d = {'Times of iteration before convergence': [kmeans4.n_iter_,kmeans8.n_iter_,kmeans16.n_iter_,kmeans32.n_iter_,kmeans64.n_iter_], 'Min SSE': [kmeans4.inertia_, kmeans8.inertia_, kmeans16.inertia_,kmeans32.inertia_,kmeans64.inertia_]}
df = pd.DataFrame(data=d)
df['Value of K']=['k=4','k=8','k=16','k=32','k=64']
df


# As the number of iteration increases, the times of iteration before convergence decreases, and the lowest SSE of the algorithms decreases. 
# In general, if the value of k is too small, the clustering might be too general and fail to capture the underlying grouping information of the data. This leads to a higher SSE of the clustering. 
# But if k value is too big, adding each cluster might not contribute much to a better grouping of the data. The decrease of SSE is going to slow as k value increases to larger values.


# ## Generate a plot of the number of clusters k (x-axis) versus the sum of squared distance (SSE) between data points and their assigned centroids (y-axis). What appears to be the optimal k from the list of values you used, and why?

# obtain sse values

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 0,
    }

sse = []
for k in [2,4,8,16,32,64]:
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

    
# plot the sse values

plt.style.use("fivethirtyeight")
plt.plot([2,4,8,16,32,64], sse, 'bx-')
plt.xticks([2,4,8,16,32,64])
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# According to the plot, after k=8, the SSE started to decrease in a linear fashion. It means that after k=8, the SSE started to decrease relatively slower with each cluter added. So, the optimal k = 8 for the current data.

#  

#  

#  

# ## For k = 8, how did you initialize the set of cluster centroids before running k-means? Rerun k-means again, this time ensuring that the centroids are all different. Does this – and if so, to what extent – affect the final clusters created?



# with k=8 and init="random"
kmeans81 = KMeans(init="random", n_clusters=8, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans81.n_iter_


# lowest sse
kmeans81.inertia_




# with k=8 and init="kmeans++"
kmeans82 = KMeans(init="k-means++", n_clusters=8, random_state=0, n_init="auto").fit(data_scaled)

# check the number of iterations before converging
kmeans82.n_iter_



# lowest sse
kmeans82.inertia_


#  By changing 'init' from 'random' to 'k-means++', a different set of cluster centroids are selected to speed up convergence and improve clustering performance.
#  Normally, using init='k-means++' should speed up convergence and decrease SSE. However, with the current data, the number of iteration before convergence and SSE both increase with 'k-means++'. 
#  One explanation might be 'k-means++' uses previous random clusterings to compare and improve centroid selection. If the 'random' selected centroids already perform well, 'k-means++' might cause the selection to diverge from the smaller-SSE centroid choices.



# ## What evaluation metrics can be used to assess the quality of the clusters produced?

#  Evaluation metrics such as SSE, which measures the sum of squared distances between each point in a cluster and the centroid of the cluster. The smaller the SSE, the better.
#  Another metrics could be 'silhouette score', which measures how well a point fits into its assigned cluster as opposed to other clusters.
#  Calinski-Harabasz Index can also be used to measure the ratio of the between-cluster variance to the within-cluster variance. Higher values indicate more compact and well-separated clusters.
#  We could also check the variance/homogeneity within each cluster to see if one clustering method is better than the others.




# # PCA

# ## Fit the standardized data with PCA. Then, create a cumulative variance plot – showing the number of components included (x-axis) versus the amount of variance captured (y-axis). Generally, we want to retain at least 75% of the variance. How many components would you decide to keep?

# start by including all 500 components to explain 100% of the variance
pca = PCA(n_components=500)
pca.fit(data_scaled)


print("Number of components:", pca.n_components_)
print("Variance explained:", sum(pca.explained_variance_ratio_))



# plot the varaince explained by each of component
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )

 
plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Explained Variance')
plt.show()



plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    label='Cumulative Explained Variance')
 
plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Cumulative Explained variance')
plt.title('Cumulative Explained variance')
 
plt.show()




#


# specify to keep 
pca = PCA(n_components=.75)
pca.fit(data_scaled)
print("Number of components:", pca.n_components_)
print("Variance explained:", sum(pca.explained_variance_ratio_))


# To retain at least 75% of the variance, 276 out of 500 components are kept



# ## Perform PCA with your selected principal components.

# ### Plot the transformed data on a graph with the first two principal components as the axes i.e. x = PC 1, y = PC 2


# Fit and transform data
pca_features = pca.fit_transform(data_scaled)
 
# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features)

# Create plot
plt.scatter(pca_features[:, 0], pca_features[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# ### Plot the original data on a graph with the two original variables that have the highest absolute combined loading for PC 1 and PC 2 i.e. maximizing |loading PC1| + |loading PC2|.


# get the loadings
loadings = np.abs(pca.components_[0]) + np.abs(pca.components_[1])


# Get components
components = pca.components_

# Calculate absolute combined loading for each variable
abs_combined_loading = np.sum(np.abs(components), axis=0)

# find the 2 components' original data
sorted_components = np.argsort(abs_combined_loading)[::-1]

orig_2 = data.iloc[:, sorted_components]

# plot the original data on a scatter plot with the two variables with the highest absolute combined loading for PC 1 and PC 2 as the axes
plt.scatter(orig_2.iloc[:, pc1_idx], orig_2.iloc[:, pc2_idx])
plt.xlabel(f'Variable {pc1_idx}')
plt.ylabel(f'Variable {pc2_idx}')

plt.legend()
plt.show()


# ## Examine the scatter plot of PC 1 (x-axis) versus PC 2 (y-axis) for all data points that you created in the previous part. Qualitatively, can you identify visible clusters? Why or why not might this be the case with this particular dataset?

#  According to the two plots, we can see that they show similar ranges of dipersion along the two axis. But the original data does not show an obvious difference. The PCA plot makes the difference a little bit more obvious.
# 
#  However, either of the plots shows a very visible clustering. This might be that the original data has 500 variables, and two variables or components might not suffice to explain the majority of the variances. 


# # Run k-means clustering on the transformed data from the previous problem

# ## Why is it a good idea to do this, especially for high-dimensional datasets? Name a number of reasons.

# Using PCA-transformed data for k-means clustering might be a good idea especially for high-dimensional datasets.
#
# One of the reasons is that PCA reduces the number of dimensions by a lot. This can make plotting and interpreting the variation in data easier.
#
# Also, PCA identifies the features (principle components) that have higher variances. This helps improve k means clustering to be more accurate.
#
# Besides, PCA centers the underlying data. This can make the clustering more stable and the converge faster. 

# ## Use the same k values again (4, 8, 16, 32, 64) to again generate an elbow plot.

# ### What is the optimal k? Is it different from the one you found in (1)


# transform the data
data_transformed = pd.DataFrame(pca.fit_transform(data_scaled))

# print the transformed data
data_transformed.head()


# run pca k means clusterings with k=4
kmeans4_1 = KMeans(init="random",n_clusters=4, random_state=0, n_init="auto").fit(data_transformed)

# check the number of iterations before converging
kmeans4_1.n_iter_


kmeans4_1.inertia_

# run pca k means clusterings with k=8
kmeans8_1 = KMeans(init="random",n_clusters=8, random_state=0, n_init="auto").fit(data_transformed)

# check the number of iterations before converging
kmeans8_1.n_iter_
kmeans8_1.inertia_




# run pca k means clusterings with k=16
kmeans16_1 = KMeans(init="random",n_clusters=16, random_state=0, n_init="auto").fit(data_transformed)

# check the number of iterations before converging
kmeans16_1.n_iter_
kmeans16_1.inertia_

# run pca k means clusterings with k=32
kmeans32_1 = KMeans(init="random",n_clusters=32, random_state=0, n_init="auto").fit(data_transformed)

# check the number of iterations before converging
kmeans32_1.n_iter_
kmeans32_1.inertia_


# run pca k means clusterings with k=64
kmeans64_1 = KMeans(init="random",n_clusters=64, random_state=0, n_init="auto").fit(data_transformed)

# check the number of iterations before converging
kmeans64_1.n_iter_
kmeans64_1.inertia_



# obtain sse values

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 0,
    }

sse = []
for k in [2,4,8,16,32,64]:
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_transformed)
    sse.append(kmeans.inertia_)

    
# plot the sse values

plt.style.use("fivethirtyeight")
plt.plot([2,4,8,16,32,64], sse, 'bx-')
plt.xticks([2,4,8,16,32,64])
plt.xlabel("Number of Clusters with Transformed Data")
plt.ylabel("SSE")
plt.title('After PCA')
plt.show()



# bring back the previous plot
sse = []
for k in [2,4,8,16,32,64]:
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

    
# plot the sse values

plt.style.use("fivethirtyeight")
plt.plot([2,4,8,16,32,64], sse, 'bx-')
plt.xticks([2,4,8,16,32,64])
plt.xlabel("Number of Clusters with Original Scaled Data")
plt.ylabel("SSE")
plt.title('Before PCA')
plt.show()


#  According the new plot with PCA-transformed data, the optimal k=16. After k=16, the SSE started to decrease relatively slower with each cluter added. This is different from the optimal k=8 in 1(c).

# ### Compare the SSE values plotted in this exercise to the previous plot you generated in (1c) before performing PCA.

# Compared to SSE in 1(c), the SSE of transformed data has decreased a lot. Also, the variances of the data seems to be easier to be clustered after being transformed with PCA. 

# ## Again, create a scatter plot of PC 1 (x-axis) versus PC 2 (y-axis) for all of the transformed data points. Label the cluster centers and color-code by cluster assignment for the first 5 iterations of k = 32. Can you see the algorithm begin to converge to optimal assignments?



# plot the 5 iterations
for i in range(5):
    kmeans = KMeans(n_clusters=32, init='random', n_init='auto', max_iter=1, random_state=i).fit(data_transformed)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.scatter(data_transformed.iloc[:, 0], data_transformed.iloc[:, 1], c=labels, cmap='autumn')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='green', marker='+')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Iteration = {}".format(i+1))
    plt.show()


#   According to the 5 plots, we can definitely see a trend of convergence. In the first plot, we can see that the data are mixed together without a clear classification. Some of the centroids are located further away from the majority of the data, which indicates higher SSE caused by outliers. 
#  As the number of iteration increases, we can see that data of similar groups start to move together. The centroids also move toward the center of each groups instead of being on the edge. 
#  By iteration 5, we can clearly see that the light yellow and the red dots lie on the upper left corner, the orange ones are in the middle, and the gold one are on the lower right corner. The clusterings become clearer. The centroids are also closer to the location of the center of each cluster without being influenced much by the outliers.
