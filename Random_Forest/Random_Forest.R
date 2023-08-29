# load the tree package
install.packages('rpart')
library(rpart)
install.packages('rpart.plot')
library(rpart.plot)
install.packages("randomForest")
library(randomForest)


# Trees are supervised algorithms that can be used for both regression and classification tasks. For the following trees, please explain how it is grown (i.e., how to select the variables to split on at each node)

## Classification Tree
###### Classification tree is a type of decision tree. Its purpose is to put each obervation into the class it belongs to. At each split, the purpose is to minimize the sum of square errors of the classification. All of the possible combinations are tested at each split, and the one with smallest deviance is kept. 
###### The spliting rules are deterministic, and the observations are always splitted in a binary way with homogeneous observations in the same group. The algorithm is recursive and greedy. This spliting stops when the minimum threshold is met (e.g. target number of observations at each leaf).
###### To measure the classification result, we can use SSE, deviance or Gini Deviance (all are the smaller the better).

###### Using the wine data as an example:
wine <- read.csv("wine.csv")
head(wine)
wine_cate <- rpart(color~residual.sugar+chlorides+density, data=na.omit(wine))
prp(wine_cate, col=8)

###### As shown in the above fitted classification tree, only classification is shown. No prediction is made on the value of each observation. We end up with a classification of each observation and a set of splitting rule. 


## Regression Tree
###### Regression tree is another type of decision tree. Its purpose is to predict the value of each node apart from prediciting the classficiation of each observation. The regression tree calculates the mean response of all observations in the same leaf to predict y value of observations selected into that node. Regression tress is more frequently used when the continuous variables are included in tree model. 

###### Using the wine data as an example:
wine_cate1 <- rpart(fixed.acidity~residual.sugar+chlorides+density, data=na.omit(wine))
prp(wine_cate1, extra = 1, branch = 0.5, varlen = 0, yesno = 2, under = TRUE)

###### As shown in the above fitted regression tree, both classification and average value calculated based on the observations in each leaf. We can get not only a classification of y variable but also a predicted value of it.




# Please explain how a tree is pruned?
###### Pruning trees includes pre-pruning and post-pruning. 
###### Pre-pruning means specifying a early stopping point for the tree to stop growing. 
###### Post-pruning means allowing the tree to grow to the fullest, deepest level. Then, starting from the bottom level, try to remove the node (and the following children) that contributes the less to reducing the deviance. After removing the least contributing node, with a whole set of candidate models, we can test them with out of sample data. Then, we can keep the model with the best out of sample performance. 




# Please explain why a Random Forest usually outperforms regular regression methods (such as linear regression, logistic regression, and lasso regression).
###### First, Random Forest is better at capturing non-linear relationships within variables. The regular regression methods are all based on linear regression, which can only capture non-linear relationships when specified with functional form and interaction terms. Random Forest, on the other hand, is based on CART and is better at measuring non-linear relationships, non-paramatric data, or interactions.
###### Second, Random Forest uses the bagging to make predictions. This means that several different models are fitted independently with different observations and variables, then an average is calculated as the prediction. This bagging way could grant better results as they fit many models instead of focusing on improving one. 
###### Besides, Random Forest is normally more robust to capture higher variance without overfitting unnecessarily and ending up with an unstable model. 








# Use the Trasaction.csv dataset to create payment default classifier ('payment_default' column) and explain your output using

## Classification Tree (CART)


# load the tree package
install.packages('rpart')
library(rpart)
install.packages('rpart.plot')
library(rpart.plot)



## load the data
data <- read.csv('Transaction.csv')
dim(data)



# factor the payment_default column
data$payment_default = as.factor(data$payment_default)

# split training and test set ()
set.seed(111)
train = sample(1:nrow(data), size=30000*0.8)
data_train=data[train,]
data_test=data[-train,]







# fit the first tree with cp=0.00111
cart_default <- rpart(payment_default~., cp=0.00111, data=data_train)
rpart.plot(cart_default)


cart_default



printcp(cart_default)

### Interpreting the CART result
###### The tree splits on 12 nodes and ended up with 12 leaves. There are 6 levels of this tree in total. 
###### 7 variables are used to fit the tree.



#### Obtain OOS accuracy

# predict with the CART
pred_tree = predict(cart_default, data_test, type = "class")
test_tree1 <- table(pred_tree,data_test$payment_default)
test_tree1



accuracy_tree1 <- (test_tree1[1]+test_tree1[4])/length(data_test$payment_default)
accuracy_tree1 


###### OOS classification accuracy is 0.815. 

### Pruning trees

# obtain the lowest CP value
cart_default$cptable[which.min(cart_default$cptable[,"xerror"]),"CP"]
```


# fit the tree again with CP updated to the lowest value
prune_cp <- cart_default$cptable[which.min(cart_default$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(cart_default, cp = prune_cp)
rpart.plot(pruned_tree)


pruned_tree




printcp(pruned_tree)

### Interpreting the pruned CART result
###### The tree splits on 5 nodes and ended up with 9 leaves. There are 4 levels of this tree in total. 
###### 3 variables are used to fit the tree.


#### Obtain OOS accuracy

# predict with the CART
pred_tree_2 = predict(pruned_tree, data_test, type = "class")
test_tree2 <- table(pred_tree_2,data_test$payment_default)
test_tree2



accuracy_tree2 <- (test_tree2[1]+test_tree2[4])/length(data_test$payment_default)
accuracy_tree2


###### Pruned tree OOS classification accuracy is 0.814. Very trivial difference from the 6 level tree. 





## Random Forest


install.packages("randomForest")
library(randomForest)
rf_fraud <- randomForest(payment_default~., data=data_train, ntree = 30, mtry = 30)
# had to use a small number of ntree here because larger numbers won't knit
print(rf_fraud)


# predict with test data
predictions <- predict(rf_fraud, data_test, type = "class")
test_tree3 <- table(predictions,data_test$payment_default)
test_tree3


accuracy_tree3 <- (test_tree3[1]+test_tree3[4])/length(data_test$payment_default)
accuracy_tree3

###### Pruned tree OOS classification accuracy is 0.8125, which is not much difference from the others. This is probably because the smaller value of ntree.

