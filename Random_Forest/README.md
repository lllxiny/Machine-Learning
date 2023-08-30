**Decision Trees (Classification and Regression) and Random Forest**

This R code demonstrates the use of decision trees and random forests for classification tasks using the "Transaction.csv" dataset.

1. **Classification Tree (CART):**
   - The code starts by loading the required packages, including `rpart` and `rpart.plot`.
   - The "Transaction.csv" dataset is read and dimensions are checked.
   - The "payment_default" column is converted into a factor variable.
   - The dataset is split into training and test sets.
   - A classification tree (CART) model is fitted to the training data with a specified complexity parameter (cp).
   - The tree is visualized using the `rpart.plot` function.
   - The fitted CART model is displayed along with its details and complexity parameters.
   - The code then calculates the accuracy of the model on the test set and prints the results.

2. **Pruning of the Classification Tree:**
   - The lowest complexity parameter (CP) value is obtained from the fitted CART model.
   - The CART model is pruned using the obtained CP value.
   - The pruned tree is visualized using the `rpart.plot` function.
   - The pruned tree details, including complexity parameters, are displayed.
   - The accuracy of the pruned tree on the test set is calculated and printed.

3. **Random Forest:**
   - The `randomForest` package is loaded.
   - A random forest model is fitted to the training data with a specified number of trees (ntree) and variables to consider (mtry).
   - The fitted random forest model is displayed, showing the number of trees, the number of input variables, and other parameters.
   - Predictions are made using the random forest model on the test set.
   - The predictions are tabulated against the actual "payment_default" values to calculate the accuracy.
   - The accuracy of the random forest model on the test set is calculated and printed.
