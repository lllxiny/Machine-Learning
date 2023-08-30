
**Logistic Regression and Double Logistic Regression for Treatment Effects**

This Python code performs logistic regression and double logistic regression to estimate the treatment effects of a 'different room being assigned' on the room being 'canceled' using the "hotel_cancellation.csv" dataset.

1. **Logistic Regression to Estimate Treatment Effects:**
   - The code reads the "hotel_cancellation.csv" dataset into the 'hotelData' dataframe.
   - The 'different_room_assigned' and 'is_canceled' columns are converted to binary numeric values (0 or 1).
   - A logistic regression model is built with 'is_canceled' as the dependent variable, 'different_room_assigned' as the treatment indicator, and other variables as covariates.
   - The model summary is displayed, including coefficients, p-values, and pseudo R-squared.
   - The interpretation of the treatment effect is provided, indicating that being assigned to a different room decreases the cancellation rate.
   - Observations about other variables' effects on cancellation are mentioned, such as the impact of arrival time and waiting time.

2. **Double Logistic Regression to Measure Treatment Effects:**
   - Another logistic regression model is built, regressing 'different_room_assigned' on other covariates to control for confounding effects.
   - The predicted values of 'different_room_assigned' based on the control variables are added to the dataframe.
   - A final logistic regression model is built with 'is_canceled' as the dependent variable, 'different_room_assigned' and the predicted values as covariates.
   - The second model's summary is displayed, showing coefficients, p-values, and pseudo R-squared.
   - The interpretation is provided, reaffirming the earlier conclusion that being assigned a different room decreases cancellation.
