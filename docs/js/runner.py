"""
Bridge between the web front end and the comxl package.

Loaded into Pyodide after the comxl wheel is installed. Every function here
takes a plain dict of form values, calls the real package, and returns a dict
of JSON-safe results -- no algorithm is reimplemented for the browser.

The one browser-specific job is turning a matplotlib figure into a base64 PNG
so it can go straight into an <img> tag.
"""

import base64
import io
import json

import matplotlib
matplotlib.use("AGG")                    # no display in the browser runtime
import matplotlib.pyplot as plt
import numpy as np

from comxl import (calculus, constrained, directsearch, gradientsearch,
                   linprog, matrixops, plotting)
from comxl.core import SAFE_NAMESPACE

plotting.set_style()
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _figure_to_png() -> str:
    """Current matplotlib figure as a base64 data URI, then close it."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close("all")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _table(df) -> dict:
    """A DataFrame as column names plus stringified rows."""
    if df is None or getattr(df, "empty", True):
        return {"columns": [], "rows": []}

    def cell(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{float(v):.6g}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    # Show the index only when it carries meaning (a labelled Hessian, a
    # tableau's basis column) rather than for a plain 0..n-1 range.
    import pandas as pd
    show_index = not isinstance(df.index, pd.RangeIndex)
    columns = ([str(df.index.name or "")] if show_index else []) + \
              [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        values = [cell(v) for v in row.tolist()]
        rows.append(([str(idx)] if show_index else []) + values)
    return {"columns": columns, "rows": rows}


def _make_callable(expression: str, n_vars: int):
    """
    Compile a user-typed expression into a callable.

    One variable is addressed as ``x``; several as ``x1, x2, ...`` (``X_1`` and
    ``x[0]`` style indexing also work). Only the maths namespace that
    ``comxl.core`` already whitelists is exposed -- no builtins, no imports.
    """
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("Enter an objective function.")
    source = expression.lstrip("=").replace("^", "**")
    code = compile(source, "<objective>", "eval")

    # Build the namespace once and rewrite only the variable entries per call.
    # A solver evaluates the objective tens of thousands of times, and copying
    # the maths namespace each time dominated the cost in the browser runtime.
    env = dict(SAFE_NAMESPACE)
    names = [(f"x{i + 1}", f"X{i + 1}", f"X_{i + 1}", f"x_{i + 1}")
             for i in range(max(n_vars, 1))]
    empty_builtins = {"__builtins__": {}}

    def f(*args):
        if len(args) == 1:
            vec = np.atleast_1d(np.asarray(args[0], dtype=float))
        else:
            vec = np.asarray(args, dtype=float)

        env["x"] = vec if n_vars > 1 else float(vec[0])
        env["X"] = env["x"]
        for i in range(min(len(vec), len(names))):
            value = float(vec[i])
            a, b, c, d = names[i]
            env[a] = env[b] = env[c] = env[d] = value
        return float(eval(code, empty_builtins, env))

    return f


def _numbers(text: str):
    """Parse whitespace/comma separated numbers."""
    cleaned = (text or "").replace(",", " ").split()
    return [float(v) for v in cleaned]


def _matrix(text: str):
    """Parse a matrix written one row per line."""
    rows = []
    for line in (text or "").strip().splitlines():
        if line.strip():
            rows.append(_numbers(line))
    if not rows:
        raise ValueError("Enter a matrix.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Every row must have the same number of entries.")
    return np.array(rows, dtype=float)


def _result_payload(res, plot=None):
    return {
        "ok": True,
        "x": (res.x.tolist() if hasattr(res.x, "tolist") else float(res.x)),
        "f": float(res.f),
        "iterations": int(res.iterations),
        "converged": bool(res.converged),
        "message": res.message,
        "method": res.method,
        "summary": repr(res),
        "table": _table(res.steps),
        "plot": plot,
    }


def _fail(exc):
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 1. One-dimensional search
# ---------------------------------------------------------------------------
ONE_D = {
    "goldensection": gradientsearch.goldensection,
    "intervalhalving": gradientsearch.intervalhalving,
    "dichotomous": gradientsearch.dichotomous,
    "fibonacci": gradientsearch.fibonacci,
    "exhaustive": gradientsearch.exhaustive,
    "secant": gradientsearch.secant,
    "newtonraphson": gradientsearch.newtonraphson,
    "divide_conquer": gradientsearch.divide_conquer,
}


def run_1d(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        method = opts["method"]
        f = _make_callable(opts["expression"], 1)
        a = float(opts.get("a", 0.0))
        b = float(opts.get("b", 1.0))

        kwargs = {"show_steps": False}
        if method == "newtonraphson":
            args = (f, float(opts.get("x0", a)))
            kwargs["tol"] = float(opts.get("tol", 0.01))
            kwargs["max_iter"] = int(opts.get("max_iter", 20))
        elif method in ("fibonacci", "exhaustive", "divide_conquer"):
            args = (f, a, b)
            kwargs["n"] = int(opts.get("n", 20))
        else:
            args = (f, a, b)
            kwargs["tol"] = float(opts.get("tol", 0.01))
            kwargs["max_iter"] = int(opts.get("max_iter", 100))
            if method == "dichotomous":
                kwargs["delta"] = float(opts.get("delta", 0.001))

        res = ONE_D[method](*args, **kwargs)

        lo, hi = (min(a, b), max(a, b))
        if method == "newtonraphson":
            span = max(abs(res.x) * 0.6, 2.0)
            lo, hi = res.x - span, res.x + span
        plotting.plot_function(f, lo, hi, mark=res.x,
                               title=f"{res.method}: objective and solution")
        return _result_payload(res, _figure_to_png())
    except Exception as exc:                      # surfaced in the UI
        return _fail(exc)


# ---------------------------------------------------------------------------
# 2. Multivariable unconstrained
# ---------------------------------------------------------------------------
def run_multivariable(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        x0 = _numbers(opts.get("x0", ""))
        if not x0:
            raise ValueError("Enter a starting point, e.g. -1.2 1.0")
        n = len(x0)
        f = _make_callable(opts["expression"], n)

        bounds = None
        lo_text, hi_text = opts.get("lower", ""), opts.get("upper", "")
        if lo_text.strip() and hi_text.strip():
            lo, hi = _numbers(lo_text), _numbers(hi_text)
            if len(lo) != n or len(hi) != n:
                raise ValueError(
                    f"Bounds must each have {n} entries to match the starting point."
                )
            bounds = [(lo[i], hi[i]) for i in range(n)]

        method = opts.get("method", "neldermead")
        maximize = bool(opts.get("maximize", False))
        common = {"show_steps": False, "maximize": maximize,
                  "max_iter": int(opts.get("max_iter", 1000))}

        if method == "powell":
            if bounds is not None:
                raise ValueError(
                    "Powell's method does not take bounds. Use Nelder-Mead for a "
                    "box-constrained problem, or clear the bounds."
                )
            res = directsearch.powell(f, x0, tol=float(opts.get("tol", 1e-6)),
                                      **common)
        else:
            res = directsearch.neldermead(f, x0, bounds=bounds, **common)

        plot = None
        if n == 2:
            pts = res.steps[["X_1", "X_2"]].values
            pad_x = max(0.5, 0.25 * (pts[:, 0].max() - pts[:, 0].min()))
            pad_y = max(0.5, 0.25 * (pts[:, 1].max() - pts[:, 1].min()))
            plotting.plot_contours(
                f,
                (pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x),
                (pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y),
                n=70, path=pts, log_levels=True,
                title=f"{res.method}: contours and search path")
            plot = _figure_to_png()
        elif n == 1:
            span = max(abs(float(res.x[0])) * 0.6, 2.0)
            plotting.plot_function(lambda t: f(np.array([t])),
                                   float(res.x[0]) - span,
                                   float(res.x[0]) + span,
                                   mark=float(res.x[0]),
                                   title=f"{res.method}: objective")
            plot = _figure_to_png()
        return _result_payload(res, plot)
    except Exception as exc:
        return _fail(exc)


# ---------------------------------------------------------------------------
# 3. Linear programming
# ---------------------------------------------------------------------------
def run_lp(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        mat = _matrix(opts["matrix"])
        if mat.shape[0] < 2:
            raise ValueError(
                "Need at least one constraint row plus the objective row."
            )
        mode = opts.get("mode", "max")
        res = linprog.simplex(mat, mode=mode, show_steps=False)

        tableaux = [_table(t) for t in res.extras.get("tableaux", [])]

        plot, vertices = None, {"columns": [], "rows": []}
        n_x = mat.shape[1] - 1
        if n_x == 2 and mode == "max":
            a = mat[:-1, :2]
            b = mat[:-1, 2]
            senses = ["<="] * len(b)
            corners = linprog.feasible_vertices(a, b, senses, show_steps=False)
            vertices = _table(corners)

            # frame on the feasible region itself: using the widest axis
            # intercept pushes the region into a corner of the plot
            if not corners.empty:
                hi_x = max(1.0, float(corners["x1"].max()) * 1.35)
                hi_y = max(1.0, float(corners["x2"].max()) * 1.35)
            else:
                hi_x = max(1.0, float(res.x[0]) * 1.6)
                hi_y = max(1.0, float(res.x[1]) * 1.6)
            plotting.plot_constraints_2d(
                a, b, senses, xlim=(0, hi_x * 1.05), ylim=(0, hi_y * 1.05),
                point=(float(res.x[0]), float(res.x[1])),
                hatch=opts.get("hatch", "++"),
                title="Feasible region, corner points and optimum")
            plot = _figure_to_png()

        out = _result_payload(res, plot)
        out["tableaux"] = tableaux
        out["vertices"] = vertices
        out["variable_names"] = res.extras.get("variable_names", [])
        return out
    except Exception as exc:
        return _fail(exc)


# ---------------------------------------------------------------------------
# 4. Matrix operations and calculus
# ---------------------------------------------------------------------------
def run_matrix(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        op = opts.get("operation", "inverse")
        A = _matrix(opts["matrix"])

        if op == "inverse":
            out = matrixops.inverse(A, show_steps=False)
            return {"ok": True, "title": "Inverse", "table": _table_array(out)}
        if op == "transpose":
            out = matrixops.transpose(A, show_steps=False)
            return {"ok": True, "title": "Transpose", "table": _table_array(out)}
        if op == "determinant":
            return {"ok": True, "title": "Determinant",
                    "scalar": f"{matrixops.determinant(A):.8g}"}
        if op == "eigenvalues":
            vals = matrixops.eigenvalues(A, show_steps=False)
            return {"ok": True, "title": "Eigenvalues",
                    "table": {"columns": ["#", "lambda"],
                              "rows": [[str(i + 1), f"{v:.8g}"]
                                       for i, v in enumerate(vals)]}}
        if op == "row_echelon":
            out = matrixops.row_echelon(A, show_steps=False)
            return {"ok": True, "title": "Row echelon form",
                    "table": _table_array(out)}
        if op == "solve":
            b = _numbers(opts.get("rhs", ""))
            if len(b) != A.shape[0]:
                raise ValueError(
                    f"The right-hand side needs {A.shape[0]} values, one per row."
                )
            x = matrixops.gauss_pivot(A, b, show_steps=False)
            residual = float(np.max(np.abs(A @ x - np.array(b))))
            return {"ok": True, "title": "Solution of A x = b",
                    "table": {"columns": ["Variable", "Value"],
                              "rows": [[f"x{i + 1}", f"{v:.8g}"]
                                       for i, v in enumerate(x)]},
                    "note": f"Largest residual |Ax - b| = {residual:.3g}"}
        raise ValueError(f"Unknown operation: {op}")
    except Exception as exc:
        return _fail(exc)


def _table_array(arr):
    arr = np.atleast_2d(arr)
    return {
        "columns": [""] + [f"c{j + 1}" for j in range(arr.shape[1])],
        "rows": [[f"r{i + 1}"] + [f"{v:.6g}" for v in arr[i]]
                 for i in range(arr.shape[0])],
    }


def run_calculus(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        point = _numbers(opts.get("point", ""))
        if not point:
            raise ValueError("Enter the point at which to evaluate, e.g. 1 2")
        n = len(point)
        f = _make_callable(opts["expression"], n)
        what = opts.get("what", "gradient")

        if what == "gradient":
            res = calculus.gradient(f, point, show_steps=False, n=n)
            return {"ok": True, "title": "Gradient vector",
                    "table": _table(res.steps),
                    "note": f"||grad f|| = {res.extras['norm']:.6g}   "
                            f"f(x) = {res.f:.6g}"}
        res = calculus.hessian(f, point, show_steps=False, n=n)
        return {"ok": True, "title": "Hessian matrix",
                "table": _table(res.steps),
                "note": f"Eigenvalues "
                        f"{np.array2string(res.extras['eigenvalues'], precision=4)} "
                        f"-> {res.extras['nature']}"}
    except Exception as exc:
        return _fail(exc)


# ---------------------------------------------------------------------------
# 5. KKT / constrained
# ---------------------------------------------------------------------------
def run_kkt(payload):
    try:
        opts = json.loads(payload) if isinstance(payload, str) else dict(payload)
        n = int(opts.get("n_vars", 2))
        f = _make_callable(opts["expression"], n)
        x0 = _numbers(opts.get("x0", "")) or [1.0] * n
        if len(x0) != n:
            raise ValueError(f"The starting guess needs {n} values.")

        ineq = [_make_callable(line, n)
                for line in (opts.get("inequality", "") or "").splitlines()
                if line.strip()]
        eq = [_make_callable(line, n)
              for line in (opts.get("equality", "") or "").splitlines()
              if line.strip()]
        if len(ineq) > 12:
            raise ValueError(
                "This solver enumerates all 2^m active sets, so more than 12 "
                "inequality constraints is impractical."
            )

        res = constrained.kkt(f, n, inequality=ineq, equality=eq, x0=x0,
                              maximize=bool(opts.get("maximize", False)),
                              show_steps=False)
        out = _result_payload(res)
        out["active_set"] = res.extras.get("active_set", "")
        return out
    except Exception as exc:
        return _fail(exc)
