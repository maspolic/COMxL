"""
One-dimensional search methods.

Ports of the single-variable solver forms:

  * `goldensection`   <- UserForm8.frm   "Golden Section Search Method 1D"
  * `intervalhalving` <- UserForm9.frm   "Interval Halving Method 1D"
  * `dichotomous`     <- UserForm10.frm  "Dichotomous Search Method 1D"
  * `fibonacci`       <- UserForm11.frm  "Fibonacci Search Method 1D"
  * `exhaustive`      <- UserForm12.frm  "Exhaustive Search Method 1D"
  * `secant`          <- UserForm13.frm  "Secant Method 1D"
  * `newtonraphson`   <- UserForm7.frm   "Newton Raphson 1D"
  * `divide_conquer`  <- Minimize1D in Optimization.bas

Every function takes the objective as a Python callable (or a string
expression), so the objective no longer has to live in a worksheet cell.

Every function also takes ``show_steps=True``: the iteration table that the
VBA wrote onto the sheet is printed to the console. Pass ``show_steps=False``
for the final answer only. Either way the table is available afterwards as
``result.steps``.

Usage
-----
>>> from comxl import gradientsearch
>>> r = gradientsearch.goldensection(lambda x: x**2 - 4*x + 3, 0, 5)
...                                              # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np

from .core import Result, StepTable, machine_precision, make_function

__all__ = [
    "goldensection",
    "intervalhalving",
    "dichotomous",
    "fibonacci",
    "exhaustive",
    "secant",
    "newtonraphson",
    "divide_conquer",
    "fibo",
]

TAU = 0.381967          # golden ratio complement, as hard-coded in the VBA


# ----------------------------------------------------------------------------
# Golden section search  (UserForm8.frm)
# ----------------------------------------------------------------------------
def goldensection(f, a, b, tol: float = 0.01, max_iter: int = 100,
                  show_steps: bool = True) -> Result:
    """
    Golden section search for the minimum of a unimodal function on [a, b].

    At each step the interval is reduced by the factor tau = 0.381967 and only
    one new function evaluation is needed, because one interior point is
    reused. Termination is on |f(alpha_1) - f(alpha_2)| < tol, matching the
    VBA rather than the more usual interval-width test.

    Parameters
    ----------
    f : callable or str
        Objective function of one variable.
    a, b : float
        Bracketing interval.
    tol : float
        Convergence tolerance on the function-value difference (default 0.01,
        the form's default).
    max_iter : int
        Iteration cap (the VBA loops while j < 100).
    show_steps : bool
        Print the iteration table.

    Returns
    -------
    Result

    Examples
    --------
    >>> r = goldensection(lambda x: (x - 2)**2, 0, 5, tol=1e-6,
    ...                   show_steps=False)
    >>> round(r.x, 3)
    2.0
    """
    g = make_function(f)
    steps = StepTable(["Iteration", "X_left", "X_1", "X_2", "X_right",
                       "f(X_1)", "f(X_2)", "Error"])

    alpha1 = a * (1 - TAU) + b * TAU
    alpha2 = a * TAU + b * (1 - TAU)
    f1, f2 = g(alpha1), g(alpha2)
    err = abs(f1 - f2)
    steps.add(0, a, alpha1, alpha2, b, f1, f2, err)

    j, converged = 1, False
    while j < max_iter:
        if f1 > f2:
            a = alpha1
            alpha1, f1 = alpha2, f2
            alpha2 = TAU * a + (1 - TAU) * b
            f2 = g(alpha2)
        else:
            b = alpha2
            alpha2, f2 = alpha1, f1
            alpha1 = TAU * b + (1 - TAU) * a
            f1 = g(alpha1)

        err = abs(f1 - f2)
        steps.add(j, a, alpha1, alpha2, b, f1, f2, err)
        if err < tol:
            converged = True
            break
        j += 1

    xmin = 0.5 * (a + b)
    return _wrap(xmin, g(xmin), j, converged, steps,
                 "Golden section search", tol, show_steps)


# ----------------------------------------------------------------------------
# Interval halving  (UserForm9.frm)
# ----------------------------------------------------------------------------
def intervalhalving(f, a, b, tol: float = 0.01, max_iter: int = 200,
                    show_steps: bool = True) -> Result:
    """
    Interval halving (three-point equal-interval search).

    The interval is sampled at its centre and at +/- L/4. Depending on which
    of the three values is smallest, either half the interval is discarded or
    both ends are pulled in, halving the interval each pass. Stops when the
    interval length falls below ``tol``.
    """
    g = make_function(f)
    steps = StepTable(["Iteration", "X_left", "X_1", "X_0", "X_2", "X_right",
                       "f(X_1)", "f(X_0)", "f(X_2)", "L"])

    xleft, xright = a, b
    L = xright - xleft
    delta = 0.25 * L
    xcenter = 0.5 * (xleft + xright)
    x1, x2 = xcenter - delta, xcenter + delta
    f1, f2, fcenter = g(x1), g(x2), g(xcenter)
    steps.add(0, xleft, x1, xcenter, x2, xright, f1, fcenter, f2, L)

    j, converged = 1, False
    while j <= max_iter:
        if fcenter > f1:
            # left sample is best: discard the right half
            xright, xcenter, fcenter = xcenter, x1, f1
            shrink = 1.0
        elif f2 < fcenter:
            # right sample is best: discard the left half
            xleft, xcenter, fcenter = xcenter, x2, f2
            shrink = 1.0
        else:
            # centre is best: pull both ends in, and halve the spacing
            xleft, xright = x1, x2
            shrink = 0.5

        L = xright - xleft
        delta = 0.25 * L
        x1, x2 = xcenter - shrink * delta, xcenter + shrink * delta
        f1, f2 = g(x1), g(x2)

        steps.add(j, xleft, x1, xcenter, x2, xright, f1, fcenter, f2, L)
        if L <= tol:
            converged = True
            break
        j += 1

    return _wrap(xcenter, fcenter, j, converged, steps,
                 "Interval halving", tol, show_steps)


# ----------------------------------------------------------------------------
# Dichotomous search  (UserForm10.frm)
# ----------------------------------------------------------------------------
def dichotomous(f, a, b, tol: float = 0.01, delta: float = 0.001,
                max_iter: int = 200, show_steps: bool = True) -> Result:
    """
    Dichotomous (paired-point) search.

    Two points a small distance ``delta`` either side of the interval centre
    are compared; whichever side is worse is discarded. ``delta`` must be well
    below ``tol`` or the method cannot resolve the minimum.

    Parameters
    ----------
    delta : float
        Separation of the paired test points (form default 0.001).
    """
    g = make_function(f)
    steps = StepTable(["Iteration", "X_left", "X_1", "X_2", "X_right",
                       "f(X_1)", "f(X_2)", "L"])

    xleft, xright = a, b
    xcenter = 0.5 * (xleft + xright)
    x1, x2 = xcenter - 0.5 * delta, xcenter + 0.5 * delta
    f1, f2 = g(x1), g(x2)
    L = xright - xleft
    steps.add(0, xleft, x1, x2, xright, f1, f2, L)

    j, converged = 1, False
    while j <= max_iter:
        if f2 >= f1:
            xright = x2
        else:
            xleft = x1
        xcenter = 0.5 * (xleft + xright)
        x1, x2 = xcenter - 0.5 * delta, xcenter + 0.5 * delta
        f1, f2 = g(x1), g(x2)
        L = xright - xleft

        steps.add(j, xleft, x1, x2, xright, f1, f2, L)
        if L <= tol:
            converged = True
            break
        j += 1

    return _wrap(xcenter, g(xcenter), j, converged, steps,
                 "Dichotomous search", tol, show_steps)


# ----------------------------------------------------------------------------
# Fibonacci search  (UserForm11.frm)
# ----------------------------------------------------------------------------
def fibo(n: int) -> float:
    """Fibonacci number F(n) with F(1) = F(2) = 1, built iteratively."""
    fi = [0.0] * (n + 2)
    fi[1] = 0.0
    if n + 1 >= 2:
        fi[2] = 1.0
    for i in range(3, n + 2):
        fi[i] = fi[i - 1] + fi[i - 2]
    return fi[n + 1]


def fibonacci(f, a, b, n: int = 10, show_steps: bool = True) -> Result:
    """
    Fibonacci search.

    Unlike golden section, the number of function evaluations is fixed in
    advance by ``n``; the reduction ratio at step k is F(n-k+1)/F(n+1) applied
    to the *original* interval length, which is what makes the final interval
    as small as possible for a given evaluation budget.

    Parameters
    ----------
    n : int
        Number of Fibonacci steps (the form's only numeric input).
    """
    g = make_function(f)
    steps = StepTable(["Iteration", "X_left", "X_1", "X_2", "X_right",
                       "f(X_1)", "f(X_2)", "L"])

    if n < 3:
        raise ValueError("Fibonacci search needs n >= 3.")

    k = 2
    L0 = b - a
    L_ks = (fibo(n - k + 1) / fibo(n + 1)) * L0
    x1, x2 = a + L_ks, b - L_ks
    f1, f2 = g(x1), g(x2)
    steps.add(1, a, x1, x2, b, f1, f2, L_ks)

    k, j = 3, 2
    while k <= n:
        if f2 < f1:
            a = x1
        else:
            b = x2
        L_ks = (fibo(n - k + 1) / fibo(n + 1)) * L0
        x1, x2 = a + L_ks, b - L_ks
        f1, f2 = g(x1), g(x2)
        steps.add(j, a, x1, x2, b, f1, f2, L_ks)
        k += 1
        j += 1

    xmin = 0.5 * (a + b)
    return _wrap(xmin, g(xmin), j, True, steps,
                 "Fibonacci search", None, show_steps,
                 message=f"Completed {n} Fibonacci steps; "
                         f"final interval = {b - a:.6g}")


# ----------------------------------------------------------------------------
# Exhaustive search  (UserForm12.frm)
# ----------------------------------------------------------------------------
def exhaustive(f, a, b, n: int = 20, show_steps: bool = True) -> Result:
    """
    Exhaustive (uniform grid) search.

    The interval is divided into ``n`` equal parts and a sliding three-point
    window is swept across it; the centre point of any window lower than both
    neighbours is recorded as the running minimum. Robust for multimodal
    functions but the most expensive of the bracketing methods.
    """
    g = make_function(f)
    steps = StepTable(["Iteration", "X_1", "X_min", "X_3",
                       "f(X_1)", "f(X_min)", "f(X_3)"])

    dx = (b - a) / n
    x1 = a
    x2 = x1 + dx
    x3 = x2 + dx
    x_min = x2
    f_min = g(x_min)
    f1, f2, f3 = g(x1), g(x2), g(x3)
    steps.add(0, x1, x_min, x3, f1, f_min, f3)

    j = 1
    while x3 <= b:
        if f2 <= f1 and f2 <= f3:
            x_min = x2
            f_min = g(x_min)
        f1, f2, f3 = g(x1), g(x2), g(x3)
        steps.add(j, x1, x_min, x3, f1, f_min, f3)
        x1 = x2
        x2 = x1 + dx
        x3 = x2 + dx
        j += 1

    return _wrap(x_min, f_min, j, True, steps,
                 "Exhaustive search", None, show_steps,
                 message=f"Scanned {n} intervals of width {dx:.6g}")


# ----------------------------------------------------------------------------
# Secant method  (UserForm13.frm)
# ----------------------------------------------------------------------------
def secant(f, a, b, tol: float = 0.01, max_iter: int = 100,
           delx: float = 0.01, show_steps: bool = True) -> Result:
    """
    Secant method applied to the derivative, i.e. a root-find on f'(x) = 0.

    Derivatives are obtained by central differences with step ``delx``, so no
    analytical derivative is required. The sign of f'(alpha) decides which end
    of the bracket moves.

    Parameters
    ----------
    tol : float
        Convergence tolerance on |f'(alpha)| (form default 0.01).
    max_iter : int
        Iteration cap (form default 100).
    delx : float
        Finite-difference step for the derivative (0.01 in the VBA).
    """
    g = make_function(f)

    def dg(x):
        return (g(x + delx) - g(x - delx)) / (2.0 * delx)

    steps = StepTable(["Iteration", "X_left", "X_right", "X_val",
                       "f'(X_left)", "f'(X_right)", "f'(X_val)", "Error"])

    da, db = dg(a), dg(b)
    if db - da == 0:
        raise ZeroDivisionError("f'(b) - f'(a) is zero; choose a different bracket.")
    alpha = b - db * (b - a) / (db - da)
    dalpha = dg(alpha)
    steps.add(0, a, b, alpha, da, db, dalpha, abs(dalpha))

    j, converged = 1, False
    for j in range(1, max_iter + 1):
        da, db = dg(a), dg(b)
        if db - da == 0:
            break
        alpha = b - db * (b - a) / (db - da)
        dalpha = dg(alpha)
        if dalpha > 0:
            b = alpha
        else:
            a = alpha
        steps.add(j, a, b, alpha, da, db, dalpha, abs(dalpha))
        if abs(dalpha) < tol:
            converged = True
            break

    return _wrap(alpha, g(alpha), j, converged, steps,
                 "Secant method", tol, show_steps)


# ----------------------------------------------------------------------------
# Newton-Raphson  (UserForm7.frm)
# ----------------------------------------------------------------------------
def newtonraphson(f, x0, tol: float = 0.01, max_iter: int = 20,
                  h: float = 1.0e-4, show_steps: bool = True) -> Result:
    """
    Newton-Raphson minimisation: x <- x - f'(x) / f''(x).

    Both derivatives come from central differences with step ``h``. The
    denominator carries the VBA's 1e-18 guard against division by zero at an
    inflection point.

    Converges quadratically near a well-behaved minimum, but will happily
    converge to a maximum or diverge from a poor starting point -- the sign of
    f''(x) in the iteration table is worth watching.

    Parameters
    ----------
    x0 : float
        Starting guess.
    max_iter : int
        Iteration cap (the VBA form allows 20).
    """
    g = make_function(f)

    def d1(x):
        return (g(x + h) - g(x - h)) / (2.0 * h)

    def d2(x):
        return (g(x - h) - 2.0 * g(x) + g(x + h)) / (h * h)

    steps = StepTable(["Iteration", "XK", "f'(XK)", "f''(XK)", "XK+1", "Error"])

    x = float(x0)
    fp, fpp = d1(x), d2(x)
    x1 = x - fp / (fpp + 1e-18)
    steps.add(0, x, fp, fpp, x1, abs(x - x1))

    j, converged = 1, False
    for j in range(1, max_iter + 1):
        fp, fpp = d1(x1), d2(x1)
        x = x1
        x1 = x - fp / (fpp + 1e-18)
        err = abs(x - x1)
        steps.add(j, x, fp, fpp, x1, err)
        if err <= tol:
            converged = True
            break

    note = ""
    if fpp < 0:
        note = "f''(x) < 0 at the solution - this is a MAXIMUM, not a minimum."
    return _wrap(x1, g(x1), j, converged, steps,
                 "Newton-Raphson", tol, show_steps, message=note)


# ----------------------------------------------------------------------------
# Divide and conquer  (Minimize1D in Optimization.bas)
# ----------------------------------------------------------------------------
def divide_conquer(f, a, b, n: int = 10, max_points: int = 400,
                   show_steps: bool = True) -> Result:
    """
    Divide-and-conquer minimisation by repeated equispaced sampling.

    Each pass lays ``n`` equally spaced points across the current bracket,
    finds the lowest, and re-brackets around it. Because it re-samples rather
    than assuming unimodality within a pass, it is more tolerant of noisy or
    multimodal functions than golden section.

    Parameters
    ----------
    n : int
        Samples per pass.
    max_points : int
        Total function-evaluation budget (the VBA's `PointMax`, default 400).
    """
    g = make_function(f)
    steps = StepTable(["Pass", "a", "b", "X_min", "f(X_min)",
                       "Rel. error", "Points used"])

    tiny = 10.0 * machine_precision()
    total, xmin, fmin = 0, 0.5 * (a + b), None
    passno, converged = 0, False

    while True:
        passno += 1
        total += n
        h = (b - a) / (n - 1)
        V = np.array([a + i * h for i in range(n)])
        fv = np.array([g(v) for v in V])

        imin = int(np.argmin(fv))
        fmin = float(fv[imin])

        if imin == n - 1:
            a, b = V[imin - 1], V[imin]
        elif imin == 0:
            a, b = V[imin], V[imin + 1]
        else:
            a, b = V[imin - 1], V[imin + 1]

        xerr = abs(a - b)
        xmin = 0.5 * (a + b)
        if abs(xmin) > 1:
            xerr = xerr / abs(xmin)

        steps.add(passno, a, b, xmin, fmin, xerr, total)
        if xerr < tiny:
            converged = True
            break
        if total > max_points:
            break

    fmin = g(xmin)
    return _wrap(xmin, fmin, passno, converged, steps,
                 "Divide and conquer (1D)", None, show_steps,
                 message=f"{total} function evaluations used")


# ----------------------------------------------------------------------------
def _wrap(x, fx, iterations, converged, steps, method, tol, show_steps,
          message="") -> Result:
    if not message:
        if converged:
            message = f"Converged to tolerance {tol}." if tol else "Completed."
        else:
            message = ("Iteration limit reached before the tolerance was met - "
                       "increase max_iter or loosen tol.")
    res = Result(x=x, f=fx, iterations=iterations, converged=converged,
                 message=message, steps=steps, method=method)
    if show_steps:
        res.display()
    return res
