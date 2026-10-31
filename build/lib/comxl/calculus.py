"""
Numerical differentiation.

All use the same central-difference formulas and step sizes as the VBA
(delx = 1e-4 for gradients and Hessians, h = 1e-4 for the 1D derivatives),
so results reproduce the spreadsheet numbers rather than merely approximating
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Result, make_function, make_vector_function

__all__ = [
    "first_derivative",
    "second_derivative",
    "gradient",
    "hessian",
    "jacobian",
    "gradient_vector",
    "hessian_matrix",
]

DELX = 1.0e-4      

def first_derivative(f, x, h: float = DELX) -> float:
    """Central-difference first derivative: (f(x+h) - f(x-h)) / 2h."""
    g = make_function(f)
    return (g(x + h) - g(x - h)) / (2.0 * h)


def second_derivative(f, x, h: float = DELX) -> float:
    """Central-difference second derivative: (f(x-h) - 2f(x) + f(x+h)) / h^2."""
    g = make_function(f)
    return (g(x - h) - 2.0 * g(x) + g(x + h)) / (h * h)



def gradient_vector(f, x, delx: float = DELX, n: int | None = None):
    """
    Gradient as a plain ndarray, with no Result or iteration table built
    
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = len(x)
    g = make_vector_function(f, n=n or k)

    grad = np.empty(k)
    for i in range(k):
        step = np.zeros(k)
        step[i] = delx
        grad[i] = (g(x + step) - g(x - step)) / (2.0 * delx)
    return grad



def hessian_matrix(f, x, delx: float = DELX, n: int | None = None):
    """
    Hessian as a plain ndarray, with no Result or iteration table built.

    The solver-facing counterpart of `hessian`, for the same reason as
    `gradient_vector`.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = len(x)
    g = make_vector_function(f, n=n or k)

    H = np.empty((k, k))
    f0 = g(x)
    for i in range(k):
        for j in range(k):
            if i == j:
                step = np.zeros(k)
                step[i] = delx
                H[i, j] = (g(x + step) - 2.0 * f0 + g(x - step)) / (delx ** 2)
            else:
                si, sj = np.zeros(k), np.zeros(k)
                si[i] = delx
                sj[j] = delx
                H[i, j] = (g(x + si + sj) - g(x + si - sj)
                           - g(x - si + sj) + g(x - si - sj)) / (4.0 * delx ** 2)
    return H


def gradient(f, x, delx: float = DELX, show_steps: bool = True, n: int | None = None):
    """
    Gradient of a scalar function of several variables.

    Each component uses a central difference in that coordinate alone
    
    Parameters
    ----------
    f : callable or str
        Objective. A callable may take one array or n scalars; a string is
        evaluated with variable names X_1 ... X_n.
    x : array-like
        Point at which the gradient is evaluated.
    delx : float
        Perturbation step (default 1e-4, as in the VBA).
    show_steps : bool
        Print the component-by-component table.

    Returns
    -------
    Result
        ``.x`` is the evaluation point and ``.extras['gradient']`` the vector;
        ``.f`` is the objective value there.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = len(x)
    g = make_vector_function(f, n=n or k)

    grad = np.zeros(k)
    rows = []
    for i in range(k):
        xf, xb = x.copy(), x.copy()
        xf[i] = x[i] + delx
        xb[i] = x[i] - delx
        f_fwd, f_bwd = g(xf), g(xb)
        grad[i] = (f_fwd - f_bwd) / (2.0 * delx)
        rows.append([f"X_{i + 1}", x[i], f_fwd, f_bwd, grad[i]])

    steps = pd.DataFrame(
        rows, columns=["Variable", "Value", "f(x+dx)", "f(x-dx)", "df/dx"]
    )
    res = Result(
        x=x, f=g(x), iterations=k, converged=True, steps=steps,
        method="Gradient vector", gradient=grad, norm=float(np.linalg.norm(grad)),
    )
    if show_steps:
        res.display()
    return res



def hessian(f, x, delx: float = DELX, show_steps: bool = True, n: int | None = None):
    """
    Hessian (matrix of second partial derivatives).

    Diagonal terms use  (f(x+h) - 2f(x) + f(x-h)) / h^2 and off-diagonal terms
    the four-point formula  (f++ - f+- - f-+ + f--) / 4h^2

    Returns
    -------
    Result
        ``.extras['hessian']`` holds the matrix as a DataFrame-friendly ndarray,
        along with its eigenvalues and a classification of the stationary point.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = len(x)
    g = make_vector_function(f, n=n or k)

    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += delx
                xm[i] -= delx
                H[i, j] = (g(xp) - 2.0 * g(x) + g(xm)) / (delx ** 2)
            else:
                xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
                xpp[i] += delx; xpp[j] += delx
                xpm[i] += delx; xpm[j] -= delx
                xmp[i] -= delx; xmp[j] += delx
                xmm[i] -= delx; xmm[j] -= delx
                H[i, j] = (g(xpp) - g(xpm) - g(xmp) + g(xmm)) / (4.0 * delx ** 2)

    labels = [f"X_{i + 1}" for i in range(k)]
    steps = pd.DataFrame(H, index=labels, columns=labels)

    eigvals = np.linalg.eigvalsh((H + H.T) / 2.0)
    if np.all(eigvals > 0):
        nature = "positive definite -> local MINIMUM"
    elif np.all(eigvals < 0):
        nature = "negative definite -> local MAXIMUM"
    elif np.all(eigvals >= 0):
        nature = "positive semi-definite -> inconclusive"
    elif np.all(eigvals <= 0):
        nature = "negative semi-definite -> inconclusive"
    else:
        nature = "indefinite -> SADDLE point"

    res = Result(
        x=x, f=g(x), iterations=k * k, converged=True, steps=steps,
        method="Hessian matrix", message=f"Eigenvalues indicate: {nature}",
        hessian=H, eigenvalues=eigvals, nature=nature,
    )
    if show_steps:
        res.display()
    return res



def jacobian(funcs, x, show_steps: bool = True):
    """
    Jacobian of a system of equations f_i(x_1..x_n).

    Uses the adaptive step of the VBA `JACOBI1`:
    ``dx = 1e-5 * |x_j| + 1e-5``, with a central difference in each column.

    Parameters
    ----------
    funcs : sequence of callables/strings, or a single callable returning a vector
    x : array-like
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)

    if callable(funcs) and not isinstance(funcs, (list, tuple)):
        probe = np.atleast_1d(np.asarray(funcs(x), dtype=float))
        m = len(probe)

        def F(v):
            return np.atleast_1d(np.asarray(funcs(v), dtype=float))
    else:
        adapted = [make_vector_function(fn, n=n) for fn in funcs]
        m = len(adapted)

        def F(v):
            return np.array([fn(v) for fn in adapted], dtype=float)

    J = np.zeros((m, n))
    for j in range(n):
        dx = 1.0e-5 * abs(x[j]) + 1.0e-5
        xf, xb = x.copy(), x.copy()
        xf[j] = x[j] + dx
        xb[j] = x[j] - dx
        J[:, j] = (F(xf) - F(xb)) / (2.0 * dx)

    steps = pd.DataFrame(
        J,
        index=[f"f_{i + 1}" for i in range(m)],
        columns=[f"X_{j + 1}" for j in range(n)],
    )
    res = Result(
        x=x, f=float(np.linalg.norm(F(x))), iterations=n, converged=True,
        steps=steps, method="Jacobian matrix", jacobian=J, residuals=F(x),
    )
    if show_steps:
        res.display()
    return res
