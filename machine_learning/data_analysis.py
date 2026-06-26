# =====================================================================
#  data_analysis.py
#  Analyse a dataset:  Cleaning -> Filling -> Visualising  (CPU only)
# ---------------------------------------------------------------------
#  STEPS OF DATA ANALYSIS (exam points):
#    1. Load the data set (e.g. data.csv)
#    2. CLEANING  -> identify the MISSING data
#    3. FILLING   -> fill the missing data (mean / median / mode)
#    4. VISUALISING -> draw charts to understand the data
#
#  Library used: pandas (data) + matplotlib (charts). CPU only.
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# STEP 1: Create a small sample dataset (normally you read a CSV file).
#   To read a real file instead, use:
#       data = pd.read_csv("data.csv")
#   np.nan means a MISSING value (empty cell).
# ---------------------------------------------------------------------
data = pd.DataFrame({
    "Name":   ["Asha", "Ravi", "Kiran", "Meena", "Sanjay", "Divya"],
    "Age":    [21, 22, np.nan, 23, 24, np.nan],     # 2 missing ages
    "Marks":  [85, np.nan, 78, 90, np.nan, 88],     # 2 missing marks
    "Result": ["Pass", "Pass", "Fail", "Pass", "Fail", "Pass"],
})

print("ORIGINAL DATA")
print(data)

# ---------------------------------------------------------------------
# STEP 2: CLEANING - identify the missing data
#   isnull()      -> True where a value is missing
#   isnull().sum()-> count of missing values per column
# ---------------------------------------------------------------------
print("\n--- Missing values in each column ---")
print(data.isnull().sum())

# ---------------------------------------------------------------------
# STEP 3: FILLING the missing data
#   - For NUMBER columns: fill with the MEAN (average) of that column.
#   - (You could also use median() or mode() depending on the data.)
# ---------------------------------------------------------------------
data["Age"]   = data["Age"].fillna(data["Age"].mean())
data["Marks"] = data["Marks"].fillna(data["Marks"].mean())

print("\nDATA AFTER FILLING MISSING VALUES")
print(data)

print("\n--- Missing values after filling (should be 0) ---")
print(data.isnull().sum())

# ---------------------------------------------------------------------
# STEP 4: Quick SUMMARY of the data
#   describe() -> count, mean, min, max etc. for number columns
# ---------------------------------------------------------------------
print("\n--- Summary statistics ---")
print(data.describe())

# ---------------------------------------------------------------------
# STEP 5: VISUALISING the data
#   We draw 3 simple charts and save them as images.
# ---------------------------------------------------------------------

# (a) Bar chart: Marks of each student
plt.figure()
plt.bar(data["Name"], data["Marks"], color="skyblue")
plt.xlabel("Name")
plt.ylabel("Marks")
plt.title("Marks of each student (Bar Chart)")
plt.savefig("marks_bar_chart.png")

# (b) Histogram: how ages are distributed
plt.figure()
plt.hist(data["Age"], bins=5, color="orange", edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Number of students")
plt.title("Age distribution (Histogram)")
plt.savefig("age_histogram.png")

# (c) Pie chart: Pass vs Fail count
plt.figure()
result_counts = data["Result"].value_counts()
plt.pie(result_counts, labels=result_counts.index, autopct="%1.1f%%",
        colors=["lightgreen", "salmon"])
plt.title("Pass vs Fail (Pie Chart)")
plt.savefig("result_pie_chart.png")

print("\nCharts saved as:")
print("  marks_bar_chart.png")
print("  age_histogram.png")
print("  result_pie_chart.png")
# plt.show()   # uncomment to see pop-up windows
