# =====================================================================
#  methods_roots.py
#  Numerical Methods  +  Solving Linear Equations   (CPU only, no GPU)
# ---------------------------------------------------------------------
#  Topics covered (exam ready):
#    1. Bisection method        -> find root of an equation
#    2. Newton-Raphson method   -> find root of an equation (fast)
#    3. Trapezoidal rule        -> numerical integration (area)
#    4. Simpson's rule          -> numerical integration (more accurate)
#    5. Solve a pair of linear equations
#         2x + 3y = 7
#         4x + 3y = 7
#
#  Only standard library is used (math) so it runs anywhere.
# =====================================================================

import math


# ---------------------------------------------------------------------
# 1) BISECTION METHOD
# ---------------------------------------------------------------------
# Idea : If f(a) and f(b) have OPPOSITE signs, a root lies between them.
#        Keep cutting the interval [a, b] in HALF and keep the half
#        that still contains the sign change. The midpoint gets closer
#        and closer to the root.
#
# Formula :  mid = (a + b) / 2
# ---------------------------------------------------------------------
def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        print("Bisection failed: f(a) and f(b) must have opposite signs.")
        return None

    for i in range(max_iter):
        mid = (a + b) / 2.0          # midpoint of current interval
        if abs(f(mid)) < tol:        # close enough -> root found
            return mid
        # keep the half where the sign change happens
        if f(a) * f(mid) < 0:
            b = mid                  # root is in left half  [a, mid]
        else:
            a = mid                  # root is in right half [mid, b]
    return mid


# ---------------------------------------------------------------------
# 2) NEWTON-RAPHSON METHOD
# ---------------------------------------------------------------------
# Idea : Start with a guess x0. Draw the tangent line at that point and
#        see where it crosses the x-axis. That crossing is a better
#        guess. Repeat. Very fast when it works.
#
# Formula :  x_new = x  -  f(x) / f'(x)
#        (df is the derivative f'(x))
# ---------------------------------------------------------------------
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:                 # avoid divide-by-zero
            print("Newton-Raphson failed: derivative became 0.")
            return None
        x_new = x - fx / dfx         # the core update step
        if abs(x_new - x) < tol:     # change is tiny -> converged
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------------
# 3) TRAPEZOIDAL RULE  (numerical integration)
# ---------------------------------------------------------------------
# Idea : Area under the curve from a to b is split into n strips.
#        Each strip is approximated by a TRAPEZIUM.
#
# Formula :  h = (b - a) / n
#   integral ~= (h/2) * [ f(x0) + 2*(f(x1)+...+f(x_{n-1})) + f(xn) ]
# ---------------------------------------------------------------------
def trapezoidal(f, a, b, n=100):
    h = (b - a) / n                  # width of one strip
    total = f(a) + f(b)              # the two end points (weight 1)
    for i in range(1, n):
        total += 2 * f(a + i * h)    # the middle points (weight 2)
    return (h / 2) * total


# ---------------------------------------------------------------------
# 4) SIMPSON'S RULE  (1/3 rule, more accurate than trapezoidal)
# ---------------------------------------------------------------------
# Idea : Instead of straight lines, fit PARABOLAS over pairs of strips.
#        n MUST be even.
#
# Formula :  h = (b - a) / n
#   integral ~= (h/3) * [ f(x0) + f(xn)
#                         + 4*(odd index points)
#                         + 2*(even index points) ]
# ---------------------------------------------------------------------
def simpsons(f, a, b, n=100):
    if n % 2 == 1:                   # Simpson needs an even number of strips
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)              # end points
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            total += 4 * f(x)        # odd points get weight 4
        else:
            total += 2 * f(x)        # even points get weight 2
    return (h / 3) * total


# ---------------------------------------------------------------------
# 5) SOLVE THE PAIR OF LINEAR EQUATIONS
#        2x + 3y = 7
#        4x + 3y = 7
#
# Method used : Cramer's rule (determinants) - easy to write in code.
#   For   a1*x + b1*y = c1
#         a2*x + b2*y = c2
#   D  = a1*b2 - a2*b1
#   Dx = c1*b2 - c2*b1
#   Dy = a1*c2 - a2*c1
#   x = Dx / D ,   y = Dy / D
# ---------------------------------------------------------------------
def solve_linear(a1, b1, c1, a2, b2, c2):
    D = a1 * b2 - a2 * b1
    if D == 0:
        print("No unique solution (lines are parallel or same).")
        return None
    Dx = c1 * b2 - c2 * b1
    Dy = a1 * c2 - a2 * c1
    x = Dx / D
    y = Dy / D
    return x, y


# =====================================================================
#  DEMO  -  run this file directly:  python methods_roots.py
# =====================================================================
if __name__ == "__main__":
    print("=" * 55)
    print(" NUMERICAL METHODS DEMO")
    print("=" * 55)

    # --- Bisection & Newton find root of  f(x) = x^2 - 2  (answer = sqrt(2))
    f  = lambda x: x * x - 2         # function
    df = lambda x: 2 * x             # its derivative (for Newton)

    print("\n1) Bisection root of x^2 - 2 in [0, 2]:")
    print("   root =", bisection(f, 0, 2))

    print("\n2) Newton-Raphson root of x^2 - 2, start x0 = 1:")
    print("   root =", newton_raphson(f, df, 1))

    print("   (true value sqrt(2) =", math.sqrt(2), ")")

    # --- Integration of  g(x) = x^2  from 0 to 1   (true answer = 1/3)
    g = lambda x: x * x
    print("\n3) Trapezoidal  integral of x^2 from 0 to 1:")
    print("   area =", trapezoidal(g, 0, 1, n=100))

    print("\n4) Simpson's    integral of x^2 from 0 to 1:")
    print("   area =", simpsons(g, 0, 1, n=100))
    print("   (true value = 0.3333... )")

    # --- Solve the given linear system
    print("\n5) Solve  2x + 3y = 7  and  4x + 3y = 7 :")
    sol = solve_linear(2, 3, 7, 4, 3, 7)
    print("   x =", sol[0], ", y =", sol[1])
    print("   Check: 2x+3y =", 2 * sol[0] + 3 * sol[1])
    print("          4x+3y =", 4 * sol[0] + 3 * sol[1])
    print("=" * 55)
