"""
Linear programming.

Ports of:
  * `simplex`        <- Simplex / MakeInitialTab / GetNextTab / GetPivotCol /
                        GetPivotRow / PrintMaxSol / PrintMinSol
                        (LinearProgramming.bas, driven by UserForm4.frm)
  * `feasible`       <- `testing` (UserForm3.frm)
  * `constraint_line`<- `drawLine` (Module5.bas, UserForm3.frm)

Problem statement follows the workbook's convention. The problem matrix has
one row per constraint plus a final row for the objective, and one column per
variable plus a final column for the right-hand side:

        x1   x2   RHS
    c1 [ 2    1    10 ]      2 x1 +   x2 <= 10
    c2 [ 1    3    15 ]        x1 + 3 x2 <= 15
    z  [ 4    3     0 ]      maximise z = 4 x1 + 3 x2

A minimisation is solved by transposing this matrix -- the dual -- which is
why the minimum solution is read from the objective row's slack columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Result

__all__ = ["simplex", "feasible", "feasible_vertices", "region_polygon",
           "line_intersection", "constraint_line"]


# ----------------------------------------------------------------------------
# Simplex method
# ----------------------------------------------------------------------------
def simplex(matrix, mode: str = "max", max_iter: int = 40,
            show_steps: bool = True) -> Result:
    """
    Solve a linear program by the simplex tableau method.

    Parameters
    ----------
    matrix : array-like, shape (m+1, n+1)
        Constraint coefficients with the RHS in the last column and the
        objective coefficients in the last row. All constraints are treated as
        <= for a maximisation.
    mode : {'max', 'min'}
        For 'min' the matrix is transposed first and the problem solved as its
        dual, matching the workbook's approach.
    max_iter : int
        Pivot cap (40 in the VBA).
    show_steps : bool
        Print every tableau, as the form wrote them down the worksheet.

    Returns
    -------
    Result
        ``.x`` holds the decision variables, ``.f`` the optimal objective, and
        ``.extras['tableaux']`` the full list of intermediate tableaux.

    Examples
    --------
    >>> m = [[2, 1, 10], [1, 3, 15], [4, 3, 0]]
    >>> r = simplex(m, mode='max', show_steps=False)
    >>> round(r.f, 4), [float(v) for v in r.x]
    (24.0, [3.0, 4.0])
    """
    mat = np.atleast_2d(np.asarray(matrix, dtype=float))
    mode = mode.lower()
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")

    if mode == "min":
        mat = mat.T.copy()

    nrows, ncols_in = mat.shape
    num_x = ncols_in - 1            # structural variables
    num_s = nrows - 1               # slack variables (one per constraint)
    num_v = num_x + num_s
    ncols = num_v + 2               # + objective column + RHS

    # --- initial tableau (MakeInitialTab) ---------------------------------
    t = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(num_x):
            t[i, j] = mat[i, j]
            if i == nrows - 1:
                t[i, j] = -t[i, j]      # objective row enters negated
        t[i, ncols - 1] = mat[i, num_x]
        for j in range(num_x, num_v + 1):
            if i + num_x == j:
                t[i, j] = 1.0

    labels = ([f"x{i + 1}" for i in range(num_x)] +
              [f"s{i + 1}" for i in range(num_s)] + ["z"])
    active = list(range(num_x, num_x + num_s))     # basis: slacks initially
    vstat = [0] * num_x + [1] * num_s

    def frame(tab, basis):
        return pd.DataFrame(
            tab,
            index=[labels[b] for b in basis] + ["z"],
            columns=labels + ["RHS"],
        )

    tableaux = [frame(t, active)]

    # --- pivoting (GetNextTab) --------------------------------------------
    n_iter = 0
    while n_iter < max_iter:
        # most negative entry in the objective row
        pivot_col = int(np.argmin(t[nrows - 1, :]))
        if t[nrows - 1, pivot_col] >= 0:
            break                                  # optimal

        # minimum non-negative ratio test
        pivot_row, ratio = -1, np.inf
        for i in range(nrows - 1):
            if t[i, pivot_col] > 0:
                r = t[i, ncols - 1] / t[i, pivot_col]
                if r < ratio:
                    ratio, pivot_row = r, i
        if pivot_row < 0:
            return _unbounded(t, labels, active, tableaux, n_iter, show_steps)

        # clear the pivot column (ClearMatCol)
        t[pivot_row, :] /= t[pivot_row, pivot_col]
        for i in range(nrows):
            if i != pivot_row:
                t[i, :] = np.round(
                    t[i, :] - t[i, pivot_col] * t[pivot_row, :], 10
                )

        leaving = active[pivot_row]
        active[pivot_row] = pivot_col
        vstat[pivot_col] = 1
        vstat[leaving] = 0

        n_iter += 1
        tableaux.append(frame(t, active))

    # --- read off the solution (PrintMaxSol / PrintMinSol) ----------------
    z = t[nrows - 1, ncols - 1]
    if mode == "max":
        values = np.zeros(num_v)
        for j in range(num_v):
            if vstat[j] == 1:
                for i in range(nrows):
                    if t[i, j] != 0:
                        values[j] = t[i, ncols - 1] / t[i, j]
                        break
        x = values[:num_x]
        var_names = labels[:num_x]
    else:
        # dual: primal minimum variables sit in the objective row's slacks
        x = np.array([t[nrows - 1, num_x + j] for j in range(num_s)])
        values = np.concatenate([x, np.zeros(max(0, num_v - len(x)))])
        var_names = [f"x{j + 1}" for j in range(num_s)]

    solution = pd.DataFrame({"Variable": var_names, "Value": x})

    res = Result(
        x=x, f=float(z), iterations=n_iter, converged=True,
        message=f"Optimal {'maximum' if mode == 'max' else 'minimum'} "
                f"found after {n_iter} pivots.",
        steps=solution, method=f"Simplex ({mode})",
        tableaux=tableaux, final_tableau=tableaux[-1],
        slack=values[num_x:] if mode == "max" else None,
        variable_names=var_names,
    )

    if show_steps:
        for i, tab in enumerate(tableaux):
            print(f"Tableau {i}" + (" (initial)" if i == 0 else ""))
            print(tab.to_string(float_format=lambda v: "%.4g" % v))
            print()
        res.display()
    return res


def _unbounded(t, labels, active, tableaux, n_iter, show_steps):
    res = Result(
        x=np.array([]), f=np.inf, iterations=n_iter, converged=False,
        message="Problem is unbounded - no finite optimum exists.",
        steps=pd.DataFrame(), method="Simplex", tableaux=tableaux,
    )
    if show_steps:
        res.display()
    return res


# ----------------------------------------------------------------------------
# Two-dimensional feasibility test  (UserForm3.frm `testing`)
# ----------------------------------------------------------------------------
def feasible(a, b, senses, point, tol: float = 1e-9) -> bool:
    """
    Test whether a point satisfies every 2D linear constraint.

    Parameters
    ----------
    a : array-like, shape (m, 2)
        Constraint coefficients [a1, a2] per row.
    b : array-like, shape (m,)
        Right-hand sides.
    senses : sequence of str
        One of '<=', '<', '>=', '>', '=' per constraint.
    point : (x1, x2)

    Returns
    -------
    bool
    """
    A = np.atleast_2d(np.asarray(a, dtype=float))
    B = np.atleast_1d(np.asarray(b, dtype=float))
    x1, x2 = float(point[0]), float(point[1])

    for i in range(A.shape[0]):
        lhs = A[i, 0] * x1 + A[i, 1] * x2
        s = senses[i].replace(" ", "")
        if s in ("<=", "<") and lhs > B[i] + tol:
            return False
        if s in (">=", ">") and lhs < B[i] - tol:
            return False
        if s == "=" and abs(lhs - B[i]) > 1e-6:
            return False
    return True


# ----------------------------------------------------------------------------
# Feasible-region corner points
# ----------------------------------------------------------------------------
def line_intersection(a1, b1, a2, b2, tol: float = 1e-12):
    """
    Intersection of the two lines a1.x = b1 and a2.x = b2 in the plane.

    Returns ``None`` when the lines are parallel or identical, otherwise the
    point as a length-2 array.
    """
    m00, m01 = float(a1[0]), float(a1[1])
    m10, m11 = float(a2[0]), float(a2[1])
    det = m00 * m11 - m01 * m10
    if abs(det) < tol:
        return None
    return np.array([
        (float(b1) * m11 - float(b2) * m01) / det,
        (m00 * float(b2) - m10 * float(b1)) / det,
    ])


def feasible_vertices(a, b, senses, nonnegative: bool = True,
                      tol: float = 1e-7, decimals: int = 9,
                      show_steps: bool = True):
    """
    Corner points of a two-variable feasible region.

    Every pair of constraint boundaries is intersected, and an intersection is
    kept when it satisfies all the other constraints. The survivors are the
    vertices of the feasible polygon -- and by the fundamental theorem of
    linear programming they are the only places an optimum can sit, which is
    what makes listing them worthwhile rather than merely decorative.

    Parameters
    ----------
    a : array-like, shape (m, 2)
        Constraint coefficients, one ``[a1, a2]`` row per constraint.
    b : array-like, shape (m,)
        Right-hand sides.
    senses : sequence of str
        ``'<='``, ``'<'``, ``'>='``, ``'>'`` or ``'='`` per constraint.
    nonnegative : bool
        Also impose x1 >= 0 and x2 >= 0, the usual LP convention. Their
        intersections with the constraints, and the origin, then appear as
        vertices too.
    tol : float
        Feasibility tolerance when testing a candidate intersection.
    decimals : int
        Rounding used to merge duplicates where three or more lines meet.
    show_steps : bool
        Print the vertex table.

    Returns
    -------
    pandas.DataFrame
        Columns ``x1``, ``x2`` and ``lines`` (which boundaries met there),
        ordered counter-clockwise around the region.

    Examples
    --------
    >>> v = feasible_vertices([[2, 1], [1, 3]], [10, 15], ['<=', '<='],
    ...                       show_steps=False)
    >>> [(round(r.x1, 4), round(r.x2, 4)) for r in v.itertuples()]
    [(0.0, 0.0), (5.0, 0.0), (3.0, 4.0), (0.0, 5.0)]
    """
    A = np.atleast_2d(np.asarray(a, dtype=float))
    B = np.atleast_1d(np.asarray(b, dtype=float))
    senses = [s.replace(" ", "") for s in senses]
    m = A.shape[0]

    all_a = [A[i] for i in range(m)]
    all_b = [B[i] for i in range(m)]
    all_s = list(senses)
    names = [f"C{i + 1}" for i in range(m)]

    if nonnegative:
        all_a += [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        all_b += [0.0, 0.0]
        all_s += [">=", ">="]
        names += ["x1>=0", "x2>=0"]

    all_a = np.array(all_a)
    all_b = np.array(all_b)

    found = {}
    for i in range(len(all_a)):
        for j in range(i + 1, len(all_a)):
            p = line_intersection(all_a[i], all_b[i], all_a[j], all_b[j])
            if p is None:
                continue
            if not feasible(all_a, all_b, all_s, p, tol=tol):
                continue
            key = tuple(np.round(p, decimals))
            if key in found:
                found[key][1].append(f"{names[i]} & {names[j]}")
            else:
                found[key] = (p, [f"{names[i]} & {names[j]}"])

    if not found:
        df = pd.DataFrame(columns=["x1", "x2", "lines"])
        if show_steps:
            print("No feasible vertices - the region is empty.")
        return df

    pts = np.array([v[0] for v in found.values()]) + 0.0  # normalise -0.0
    labels = [", ".join(v[1]) for v in found.values()]

    # order counter-clockwise about the centroid, so the rows trace the
    # boundary instead of appearing in discovery order
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(ang)

    df = pd.DataFrame({
        "x1": pts[order, 0],
        "x2": pts[order, 1],
        "lines": [labels[k] for k in order],
    })

    if show_steps:
        print(df.to_string(index=False, float_format=lambda v: "%.6g" % v))
        print(f"\n{len(df)} vertices of the feasible region.")
    return df


def region_polygon(a, b, senses, xlim, ylim, nonnegative: bool = True,
                   tol: float = 1e-7):
    """
    Vertices of the feasible region clipped to a viewing box.

    Same construction as `feasible_vertices`, but the four box edges are added
    as extra constraints so that an unbounded region still closes into a
    polygon that can be shaded. Returns an ordered ``(k, 2)`` array, empty when
    nothing is feasible.
    """
    A = np.atleast_2d(np.asarray(a, dtype=float))
    B = np.atleast_1d(np.asarray(b, dtype=float))
    senses = [s.replace(" ", "") for s in senses]

    all_a = np.vstack([A, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    all_b = np.concatenate([B, [xlim[0], xlim[1], ylim[0], ylim[1]]])
    all_s = list(senses) + [">=", "<=", ">=", "<="]

    if nonnegative:
        all_a = np.vstack([all_a, [[1.0, 0.0], [0.0, 1.0]]])
        all_b = np.concatenate([all_b, [0.0, 0.0]])
        all_s = all_s + [">=", ">="]

    pts = []
    n = all_a.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            p = line_intersection(all_a[i], all_b[i], all_a[j], all_b[j])
            if p is not None and feasible(all_a, all_b, all_s, p, tol=tol):
                pts.append(p)

    if not pts:
        return np.empty((0, 2))

    pts = np.unique(np.round(np.array(pts), 9) + 0.0, axis=0)
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    return pts[np.argsort(ang)]


# ----------------------------------------------------------------------------
# Constraint line geometry  (drawLine in Module5.bas)
# ----------------------------------------------------------------------------
def constraint_line(x1, x2, a, b, c, sense="<="):
    """
    Endpoints of the line a*X1 + b*X2 = c over [x1, x2], plus a slightly
    offset companion line indicating the feasible side.

    Reproduces `drawLine`: the offset factor depends on the magnitude of c
    (2.5%, 5% or 10%) and its direction on the constraint sense.

    Returns
    -------
    dict with keys 'x', 'y', 'y_offset', 'vertical'
    """
    sense = sense.replace(" ", "")
    if sense == "=":
        cmult = 1
    elif sense in ("<=", "<"):
        cmult = 1
    else:
        cmult = -1

    if abs(c) >= 10:
        cfac = 0.025
    elif abs(c) > 5:
        cfac = 0.05
    else:
        cfac = 0.1

    cdum = cmult * 0.1 if c == 0 else (1 + cmult * cfac) * c

    if b != 0:
        y1 = (c - a * x1) / b
        y2 = (c - a * x2) / b
        y1n = (cdum - a * x1) / b
        y2n = (cdum - a * x2) / b
        return {"x": (x1, x2), "y": (y1, y2),
                "y_offset": (y1n, y2n), "vertical": False}

    # b == 0 -> vertical line at x = c/a
    xv = c / a if a != 0 else np.nan
    return {"x": (xv, xv), "y": (x1, x2),
            "y_offset": (x1, x2), "vertical": True}
