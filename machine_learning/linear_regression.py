# =====================================================================
#  linear_regression.py
#  Develop a Machine Learning model using LINEAR REGRESSION (CPU only)
# ---------------------------------------------------------------------
#  WHAT IS LINEAR REGRESSION?
#    - Used when the OUTPUT is a NUMBER (continuous value).
#      Example: predict salary from years of experience,
#               predict house price from size.
#    - It fits a straight line:   y = m*x + c
#         m = slope (weight),  c = intercept (bias)
#    - "Best fit line" = the line with the SMALLEST error between the
#      predicted points and the real points (least squares).
#
#  Library used: scikit-learn (runs on CPU, very simple).
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# STEP 1: Create / load the data
#   X = input  (years of experience)
#   y = output (salary)
# Normally you read this from a CSV. Here we make simple sample data.
# ---------------------------------------------------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # shape (n, 1)
y = np.array([30, 35, 40, 50, 55, 60, 68, 72, 80, 88])        # salary (k)

# ---------------------------------------------------------------------
# STEP 2: Split data into training and testing sets
#   - Train set: model learns from it.
#   - Test set : we check how good the model is on unseen data.
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# STEP 3: Create the model and TRAIN it (fit = learn m and c)
# ---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope (m)     :", model.coef_[0])
print("Intercept (c) :", model.intercept_)
print("Learned line  :  y = %.2f * x + %.2f" % (model.coef_[0], model.intercept_))

# ---------------------------------------------------------------------
# STEP 4: Make predictions on the test data
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------
# STEP 5: Evaluate the model
#   - MSE (Mean Squared Error): lower is better.
#   - R2 score: how well the line fits (1.0 = perfect).
# ---------------------------------------------------------------------
print("\nMean Squared Error :", mean_squared_error(y_test, y_pred))
print("R2 Score           :", r2_score(y_test, y_pred))

# ---------------------------------------------------------------------
# STEP 6: Predict a NEW value (example: 11 years experience)
# ---------------------------------------------------------------------
new_x = np.array([[11]])
print("\nPredicted salary for 11 years exp :", model.predict(new_x)[0])

# ---------------------------------------------------------------------
# STEP 7: Visualise the result
#   - Blue dots  = real data points
#   - Red  line  = the best fit line the model learned
# ---------------------------------------------------------------------
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Best fit line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands)")
plt.title("Linear Regression")
plt.legend()
plt.savefig("linear_regression_plot.png")   # saved as image (no GUI needed)
print("\nPlot saved as 'linear_regression_plot.png'")
# plt.show()   # uncomment if you want a pop-up window
