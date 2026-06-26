# =====================================================================
#  logistic_regression.py
#  Develop a Machine Learning program using LOGISTIC REGRESSION (CPU)
# ---------------------------------------------------------------------
#  WHAT IS LOGISTIC REGRESSION?
#    - Used for CLASSIFICATION (the output is a CATEGORY, not a number).
#      Example: pass / fail, spam / not-spam, disease yes / no.
#    - It predicts a PROBABILITY between 0 and 1 using the
#      SIGMOID function:        sigmoid(z) = 1 / (1 + e^-z)
#    - If probability >= 0.5  -> class 1, else class 0.
#
#  KEY DIFFERENCE FROM LINEAR REGRESSION:
#      Linear   -> predicts a NUMBER      (e.g. salary = 75)
#      Logistic -> predicts a CLASS / YES-NO (e.g. pass = 1)
#
#  Library used: scikit-learn (CPU, simple).
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------------------------------------------------------------
# STEP 1: Data
#   X = hours studied
#   y = result  (0 = fail, 1 = pass)
# Students who study more hours tend to PASS.
# ---------------------------------------------------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])   # 0 = fail, 1 = pass

# ---------------------------------------------------------------------
# STEP 2: Split into training and testing data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# STEP 3: Create the model and TRAIN it
# ---------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# STEP 4: Predict on test data
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------
# STEP 5: Evaluate the model
#   - Accuracy: fraction of correct predictions.
#   - Confusion matrix: shows correct vs wrong for each class.
# ---------------------------------------------------------------------
print("Predicted :", y_pred)
print("Actual    :", y_test)
print("\nAccuracy         :", accuracy_score(y_test, y_pred))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------
# STEP 6: Predict for a NEW student
#   predict        -> gives the class (0 or 1)
#   predict_proba  -> gives the probability of each class
# ---------------------------------------------------------------------
new_hours = np.array([[5.5]])
print("\nStudent who studied 5.5 hours:")
print("  Predicted class       :", model.predict(new_hours)[0])
print("  Probability [fail, pass]:", model.predict_proba(new_hours)[0])

# ---------------------------------------------------------------------
# STEP 7: Visualise the sigmoid (S-shaped) curve
# ---------------------------------------------------------------------
x_line = np.linspace(0, 11, 100).reshape(-1, 1)
prob = model.predict_proba(x_line)[:, 1]   # probability of class 1 (pass)

plt.scatter(X, y, color="blue", label="Actual data (0=fail, 1=pass)")
plt.plot(x_line, prob, color="red", label="Sigmoid probability curve")
plt.axhline(0.5, color="green", linestyle="--", label="Threshold = 0.5")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.savefig("logistic_regression_plot.png")
print("\nPlot saved as 'logistic_regression_plot.png'")
# plt.show()   # uncomment for a pop-up window
