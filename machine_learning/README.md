# Machine Learning — Exam Notes (Linear & Logistic Regression + Numerical Methods)

Simple, **CPU-only** code. Everything here is meant to be read quickly the night before the exam.

---

## 📁 Files in this folder

| File | What it covers |
|------|----------------|
| `methods_roots.py` | Bisection, Newton-Raphson, Trapezoidal, Simpson's + solving linear equations |
| `linear_regression.py` | ML model using **Linear Regression** |
| `logistic_regression.py` | ML program using **Logistic Regression** |
| `data_analysis.py` | Data cleaning, filling missing data, visualising |

---

## ▶️ How to run

```bash
cd machine_learning

python methods_roots.py
python linear_regression.py
python logistic_regression.py
python data_analysis.py
```

If a library is missing, install once:

```bash
pip install numpy pandas scikit-learn matplotlib
```

> `methods_roots.py` needs **nothing extra** (only Python's built-in `math`).

---

## 1. Linear Regression (predicts a NUMBER)

- **Use when:** output is a continuous value (salary, price, temperature).
- **Equation of the line:** `y = m*x + c`
  - `m` = slope (weight), `c` = intercept (bias)
- **Goal:** find the line with the **smallest error** between predicted and real points (least squares).
- **Evaluate with:** Mean Squared Error (MSE, lower = better) and R² score (1.0 = perfect fit).

**Example:** predict salary from years of experience.

---

## 2. Logistic Regression (predicts a CLASS / YES–NO)

- **Use when:** output is a category (pass/fail, spam/not-spam, disease yes/no).
- **Sigmoid function:** `sigmoid(z) = 1 / (1 + e^(-z))` → gives a probability between 0 and 1.
- **Decision rule:** probability ≥ 0.5 → class **1**, else class **0**.
- **Evaluate with:** Accuracy and Confusion Matrix.

**Example:** predict pass/fail from hours studied.

### Linear vs Logistic (common exam question)

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | A number (continuous) | A class (0 or 1) |
| Task type | Regression | Classification |
| Curve | Straight line | S-shaped sigmoid |
| Example | Predict salary | Predict pass/fail |
| Metric | MSE, R² | Accuracy, Confusion matrix |

---

## 3. Numerical Methods (`methods_roots.py`)

### Bisection method (find a root)
If `f(a)` and `f(b)` have **opposite signs**, a root lies between them.
Repeatedly halve the interval: `mid = (a + b) / 2`, keep the half with the sign change.

### Newton-Raphson method (find a root, fast)
Start with a guess `x0`, follow the tangent to the x-axis, repeat:
```
x_new = x - f(x) / f'(x)
```

### Trapezoidal rule (area / integration)
Split area into strips, approximate each as a trapezium. With `h = (b-a)/n`:
```
integral ≈ (h/2) * [ f(x0) + 2(f(x1)+...+f(x_{n-1})) + f(xn) ]
```

### Simpson's rule (area, more accurate — n must be EVEN)
Fit parabolas instead of straight lines. With `h = (b-a)/n`:
```
integral ≈ (h/3) * [ f(x0) + f(xn) + 4(odd points) + 2(even points) ]
```

---

## 4. Solve the linear equations

```
2x + 3y = 7
4x + 3y = 7
```
Subtract equation 1 from equation 2 → `2x = 0` → **x = 0**, then `3y = 7` → **y = 7/3 ≈ 2.33**.
In code this is done with **Cramer's rule** (determinants) inside `methods_roots.py`.

---

## 5. Data Analysis steps (`data_analysis.py`)

1. **Load** the dataset → `pd.read_csv("data.csv")` (or sample data in the file).
2. **Cleaning** → find missing values: `data.isnull().sum()`.
3. **Filling** → replace missing values with the **mean** (or median/mode):
   `data["Age"].fillna(data["Age"].mean())`.
4. **Visualising** → bar chart, histogram, pie chart (saved as PNG images).

> Missing values are shown as `NaN`. Always check missing data **before** training a model.

---

### 🔑 Quick revision points
- Linear → number, Logistic → class.
- Sigmoid squashes any value into 0–1.
- Bisection needs opposite signs; Newton-Raphson needs the derivative.
- Simpson's rule needs an **even** number of strips and is more accurate than trapezoidal.
- Always **clean → fill → visualise** before building a model.

Good luck in your exam! 🎯
