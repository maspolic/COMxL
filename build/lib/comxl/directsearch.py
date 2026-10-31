"""
Multivariable unconstrained optimisation.

Ports of:
  * `neldermead` <- NMSimplex / Orden (NelderMead_algo.bas) driven by
    Minimization1 (UserForm18.frm)
  * `powell`     <- Optimize_PowellsMethod / LNSRCH / LNFUNC
    (UserForm15.frm, "Unconstrained Optimization")

Both accept box constraints. The VBA applied simple clipping to the box
(`Constrain_Point` in Optimization.bas) plus decade scaling of the variables
(`XScaling`); both behaviours are preserved here.
"""

from __future__ import annotations

import numpy as np

from .core import Result, StepTable, machine_precision, make_vector_function

__all__ = ["neldermead", "powell", "constrain_point", "scale_factor"]

SMALL = 1.0e-12


# ----------------------------------------------------------------------------
# Helpers from Optimization.bas
# ----------------------------------------------------------------------------
def scale_factor(a: float) -> float:
    """Nearest power of ten below |a| -- the VBA `ScaleFactor`."""
    if a != 0:
        return 10.0 ** int(np.log(abs(a)) / np.log(10.0))
    return 1.0


def constrain_point(x, box):
    """Clip x into the box [lo, hi] element-wise (`Constrain_Point`)."""
    if box is None:
        return x
    lo = np.asarray(box[0], dtype=float)
    hi = np.asarray(box[1], dtype=float)
    return np.clip(x, lo, hi)


def _as_box(bounds, n):
    """
    Accept either ``[(lo_1, hi_1), ..., (lo_n, hi_n)]`` (per variable) or
    ``((lo_1..lo_n), (hi_1..hi_n))`` (the VBA's two-row layout) and normalise
    to ``(lo_array, hi_array)``.

    When n == 2 the two layouts have the same shape, so they are told apart by
    the fact that a lower bound never exceeds its upper bound.
    """
    if bounds is None:
        return None
    arr = np.asarray(bounds, dtype=float)

    per_var = arr.shape == (n, 2) and np.all(arr[:, 0] <= arr[:, 1])
    two_row = arr.shape == (2, n) and np.all(arr[0] <= arr[1])

    if per_var and two_row:
        # Ambiguous only when n == 2. A reading that collapses the box to zero
        # width is almost certainly the wrong one, so fall back to the other;
        # otherwise prefer the per-variable form used throughout the docs.
        if np.all(arr[:, 0] == arr[:, 1]) and not np.all(arr[0] == arr[1]):
            return arr[0], arr[1]
        return arr[:, 0], arr[:, 1]
    if per_var:
        return arr[:, 0], arr[:, 1]
    if two_row:
        return arr[0], arr[1]

    raise ValueError(
        f"bounds must be [(lo, hi)] per variable (shape ({n}, 2)) or "
        f"((lo...), (hi...)) (shape (2, {n})), with every lower bound not "
        f"exceeding its upper bound. Got array of shape {arr.shape}."
    )


# ----------------------------------------------------------------------------
# Nelder-Mead downhill simplex  (NelderMead_algo.bas)
# ----------------------------------------------------------------------------
def neldermead(f, x0, bounds=None, tol: float | None = None,
               xtol: float = 1.0e-8, max_iter: int = 1000,
               maximize: bool = False, show_steps: bool = True) -> Result:
    """
    Nelder-Mead downhill simplex minimisation.

    Derivative-free, so it copes with objectives that are noisy, non-smooth or
    only available as a black box -- but it can stall on a degenerate simplex,
    which is why convergence is judged by the *relative spread* of function
    values across the simplex:

        Delta = 2|y_best - y_worst| / (|y_best| + |y_worst| + 1e-10)

    That test alone is not sufficient. Vertices placed symmetrically about the
    minimum have equal function values, driving Delta to zero while the
    simplex is still wide -- on ``(x - 3)**2`` from ``x0 = 0`` the VBA stops at
    2.9 with vertices at 2.9 and 3.1. A second test on the simplex's geometric
    size (``xtol``) is therefore required as well, and both must be satisfied.

    The move sequence follows the VBA exactly: reflection (2c - x_w),
    expansion (3c - 2x_w), outside contraction (1.5c - 0.5x_w), inside
    contraction (0.5(c + x_w)), and shrinkage toward the best vertex.

    Parameters
    ----------
    f : callable or str
        Objective. A callable may take one array or n separate scalars.
    x0 : array-like
        Starting point; its length sets the dimensionality.
    bounds : optional
        Box constraints as ``[(lo, hi), ...]`` per variable or
        ``((lo...), (hi...))``. Points are clipped into the box at every
        evaluation.
    tol : float, optional
        Tolerance on the function-value spread (default 2 x machine epsilon,
        as in the form).
    xtol : float
        Tolerance on the simplex size, relative to the scale of the current
        best vertex. Set to 0 to reproduce the VBA's function-value-only test.
    max_iter : int
        Iteration cap (the form's Itmax, default 1000).
    maximize : bool
        Maximise instead of minimise (the form's Max/Min option button).
    show_steps : bool
        Print the iteration table.

    Returns
    -------
    Result
        ``.extras['simplex']`` holds the final simplex.

    Examples
    --------
    >>> rosen = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    >>> r = neldermead(rosen, [-1.2, 1.0], show_steps=False)
    >>> bool(np.allclose(r.x, [1, 1], atol=1e-3))
    True
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    if n < 1:
        raise ValueError("x0 must contain at least one variable.")
    if tol is None:
        tol = 2.0 * machine_precision()

    g_raw = make_vector_function(f, n=n)
    box = _as_box(bounds, n)
    sign = -1.0 if maximize else 1.0

    def fun(v):
        v = constrain_point(np.asarray(v, dtype=float), box)
        return sign * g_raw(v)

    # ---- build the initial simplex (Minimization1 in UserForm18) ----------
    s = np.tile(x0, (n + 1, 1))
    if box is None:
        offsets = [0.1, 0.1, 0.1]           # formula_a / _b / _c in the VBA
        for i in range(n):
            k = (i + 1) % 3
            s[i, i] += offsets[2] if k == 0 else (offsets[1] if k == 2 else offsets[0])
    else:
        # The VBA seeded the simplex at (x0 + lo + hi)/3 with a fixed 0.1
        # perturbation. That perturbation is meaningless when the box is much
        # wider or narrower than 0.1, so here it scales with the box instead.
        lo, hi = box
        x0c = constrain_point(x0, box)
        span = np.maximum(hi - lo, SMALL)
        for i in range(n + 1):
            s[i, :] = (x0c + lo + hi) / 3.0
            if i < n:
                s[i, i] += 0.1 * span[i]
        s = np.array([constrain_point(row, box) for row in s])

    y = np.array([fun(s[i]) for i in range(n + 1)])
    order = np.argsort(y)
    s, y = s[order], y[order]

    steps = StepTable(["Iteration", "Best f", "Worst f", "Delta", "Size"] +
                      [f"X_{i + 1}" for i in range(n)])

    def simplex_size(s):
        """Simplex extent relative to the scale of the best vertex."""
        spread = np.max(np.abs(s[1:, :] - s[0, :])) if n >= 1 else 0.0
        return float(spread / (1.0 + np.max(np.abs(s[0, :]))))

    size = simplex_size(s)
    steps.add(0, sign * y[0], sign * y[-1], 1.0, size, *s[0])

    delta, its = 1.0, 0
    while (delta >= tol or (xtol > 0 and size >= xtol)) and its < max_iter:
        cm = s[:n, :].mean(axis=0)                # centroid of the n best
        xw = s[n, :]                              # worst vertex

        r = constrain_point(2.0 * cm - xw, box)
        yr = fun(r)

        if yr < y[n - 1]:
            if y[0] <= yr:
                s[n, :], y[n] = r, yr                       # accept reflection
            else:
                e = constrain_point(3.0 * cm - 2.0 * xw, box)
                ye = fun(e)
                if ye < yr:
                    s[n, :], y[n] = e, ye                   # accept expansion
                else:
                    s[n, :], y[n] = r, yr
        else:
            if yr < y[n]:
                c = constrain_point(1.5 * cm - 0.5 * xw, box)
                yc = fun(c)
                if yc <= yr:
                    s[n, :], y[n] = c, yc                   # outside contraction
                else:
                    s, y = _shrink(s, y, n, fun, box)
            else:
                cc = constrain_point(0.5 * (cm + xw), box)
                ycc = fun(cc)
                if ycc < y[n]:
                    s[n, :], y[n] = cc, ycc                 # inside contraction
                else:
                    s, y = _shrink(s, y, n, fun, box)

        order = np.argsort(y)
        s, y = s[order], y[order]
        delta = 2.0 * abs(y[0] - y[n]) / (abs(y[0]) + abs(y[n]) + 1e-10)
        size = simplex_size(s)
        its += 1
        steps.add(its, sign * y[0], sign * y[n], delta, size, *s[0])

    converged = delta < tol and (xtol <= 0 or size < xtol)
    xsol = s[0, :]
    res = Result(
        x=xsol, f=sign * y[0], iterations=its, converged=converged,
        message=("Converged." if converged else
                 f"Stopped at the {max_iter}-iteration limit."),
        steps=steps, method="Nelder-Mead simplex", simplex=s, delta=delta,
        size=size,
    )
    if show_steps:
        res.display()
    return res


def _shrink(s, y, n, fun, box):
    """Shrink every vertex toward the best one (the VBA general contraction)."""
    for i in range(1, n + 1):
        s[i, :] = 0.5 * (s[0, :] + s[i, :])
        s[i, :] = constrain_point(s[i, :], box)
        y[i] = fun(s[i, :])
    return s, y


# ----------------------------------------------------------------------------
# Powell's conjugate direction method  (UserForm15.frm)
# ----------------------------------------------------------------------------
def powell(f, x0, tol: float = 1.0e-6, max_iter: int = 1000,
           line_search: str = "golden", maximize: bool = False,
           show_steps: bool = True) -> Result:
    """
    Powell's conjugate direction method.

    Minimises along each of n directions in turn, then replaces the direction
    of largest improvement with the net displacement of the whole cycle. On a
    quadratic this builds a conjugate set and terminates in n cycles.

    Each line search first brackets the minimum by accelerated stepping
    (x3 = 3x2 - 2x1) and then refines it, either by golden section or by
    quadratic interpolation.

    Parameters
    ----------
    tol : float
        Relative convergence tolerance on x and f (form default 1e-6).
    line_search : {'golden', 'quadratic'}
        Refinement rule -- the VBA `lsm` switch (6 = golden section).
    maximize : bool
        Maximise instead of minimise.

    Returns
    -------
    Result
        ``.extras['directions']`` holds the final direction matrix and
        ``.extras['nfe']`` the function-evaluation count.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    g_raw = make_vector_function(f, n=n)
    sign = -1.0 if maximize else 1.0

    counter = {"nfe": 0}

    def fun(v):
        counter["nfe"] += 1
        return sign * g_raw(np.asarray(v, dtype=float))

    x = x0.copy()
    xs = x.copy()
    fold = fun(x)

    # initial bracket step sizes and unit direction matrix
    bs = np.zeros(n + 1)
    bsn1 = 1.0
    for i in range(n):
        bs[i] = 0.1 * abs(x[i]) + SMALL
        bsn1 = min(bsn1, bs[i])
    bs[n] = bsn1
    d = np.eye(n)

    steps = StepTable(["Iteration", "f(x)", "Rel. change", "Func. evals"] +
                      [f"X_{i + 1}" for i in range(n)])
    steps.add(0, sign * fold, np.nan, counter["nfe"], *x)

    converged, k, ss = False, 0, np.inf
    for k in range(1, max_iter + 1):
        fx = fold
        isrch, df, dfa, fxxt = 0, 0.0, 0.0, fold

        for i in range(n):                       # search each direction
            dx = d[:, i].copy()
            fxxt = fold
            x, fold, dx = _lnsrch(bs, dx, fold, fun, i, line_search, n, x)
            dfa = abs(fxxt - fold)
            if dfa > df:
                df, isrch = dfa, i

        if 2.0 * dfa <= tol * (abs(fxxt) + abs(fold) + SMALL):
            converged = True
            steps.add(k, sign * fold, dfa, counter["nfe"], *x)
            break

        ss = max(abs(xs[j] - x[j]) / (abs(x[j]) + SMALL) for j in range(n))
        steps.add(k, sign * fold, ss, counter["nfe"], *x)
        if ss <= tol:
            converged = True
            break

        # extrapolate along the net direction of this cycle
        xe = 2.0 * x - xs
        dx = x - xs
        xs = x.copy()
        fxxt = fun(xe)

        if fxxt < fx:
            sd = (2.0 * (fx - 2.0 * fold + fxxt) * (fx - fold - df) ** 2
                  - df * (fx - fxxt) ** 2)
            if sd < 0:
                x, fold, dx = _lnsrch(bs, dx, fold, fun, n, line_search, n, x)
                d[:, isrch] = dx

    res = Result(
        x=x, f=sign * fold, iterations=k, converged=converged,
        message=("Converged." if converged else
                 f"Not converged after {max_iter} iterations."),
        steps=steps, method="Powell's method",
        directions=d, nfe=counter["nfe"], relative_error=ss,
    )
    if show_steps:
        res.display()
    return res


def _lnfunc(dx, xm, x, fun):
    """Objective restricted to the line x + xm*dx (`LNFUNC`)."""
    return fun(x + xm * dx)


def _lnsrch(bs, dx, f1, fun, idx, method, n, x):
    """
    Line search along direction dx (`LNSRCH`).

    Brackets by accelerated stepping, then refines by golden section or
    quadratic interpolation. Returns the updated point, its objective value
    and the scaled direction vector.
    """
    x1 = 0.0
    f1 = _lnfunc(dx, x1, x, fun)
    x2 = bs[idx]
    f2 = _lnfunc(dx, x2, x, fun)

    if f1 < f2:
        x1, x2 = x2, x1
        f1, f2 = f2, f1

    df1 = f2 - f1
    i, x3, f3 = 0, x2, f2
    for _ in range(200):                       # accelerated bracketing
        x3 = 3.0 * x2 - 2.0 * x1
        f3 = _lnfunc(dx, x3, x, fun)
        df2 = f3 - f2
        if df2 * df1 < 0:
            break
        x1, f1, x2, f2, df1 = x2, f2, x3, f3, df2
        i += 1

    bs[idx] = 1.5 * bs[idx] if i > 1 else bs[idx] / 2.0

    if method == "golden":
        xlow, xup = min(x1, x3), max(x1, x3)
        delx = 0.381966 * (xup - xlow)
        xa = xlow + delx
        fa = _lnfunc(dx, xa, x, fun)
        xb = xup - delx
        fb = _lnfunc(dx, xb, x, fun)
        xmin = 0.5 * (xa + xb)
        for _ in range(200):
            if fb > fa:
                xup, xb, fb = xb, xa, fa
                xa = xlow + 0.381966 * (xup - xlow)
                fa = _lnfunc(dx, xa, x, fun)
            else:
                xlow, xa, fa = xa, xb, fb
                xb = xup - 0.381966 * (xup - xlow)
                fb = _lnfunc(dx, xb, x, fun)
            xmin = 0.5 * (xa + xb)
            if abs(xup - xlow) / (SMALL + abs(xmin)) <= 0.5:
                break
    else:                                       # quadratic interpolation
        denom_b = (x2 - x1) if (x2 - x1) != 0 else SMALL
        b = (f2 - f1) / denom_b
        denom_c = ((x3 - x1) if (x3 - x1) != 0 else SMALL)
        c = ((f3 - f1) / denom_c - b) / ((x3 - x2) if (x3 - x2) != 0 else SMALL)
        xmin = (c * (x1 + x2) - b) / (2.0 * c) if c != 0 else 0.0

    dx = xmin * dx
    x = x + dx
    return x, fun(x), dx
