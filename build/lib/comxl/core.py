"""
Core infrastructure for COMxL-Py.

Provides the two things every solver in this package needs:

1. `make_function` / `make_vector_function` -- adapters that turn whatever the
   user supplies (a Python callable, a string expression) into a uniform
   callable. This replaces the VBA `Evaluate(Range(...).Formula)` machinery,
   where the objective lived in a worksheet cell.

2. `Result` and `StepTable` -- containers that hold the final answer plus the
   full iteration history. The original VBA wrote every iteration to the
   worksheet; here that table is a pandas DataFrame which is printed when
   `show_steps=True` (the default) and always available as `result.steps`.
"""

from __future__ import annotations

import inspect
import math
import numpy as np
import pandas as pd

__all__ = [
    "make_function",
    "make_vector_function",
    "Result",
    "StepTable",
    "machine_precision",
    "SAFE_NAMESPACE",
]


# ----------------------------------------------------------------------------
# Expression evaluation namespace
# ----------------------------------------------------------------------------
# The VBA code evaluated worksheet formulas, so users wrote things like
# "X_1^2 - 3*X_1 + EXP(X_1)". We keep string support (with ^ meaning power and
# Excel-style function names accepted) but a plain Python callable is the
# preferred, and more powerful, way to supply an objective.

SAFE_NAMESPACE = {
    # math
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pi": math.pi,
    "e": math.e,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "power": np.power,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "np": np,
}


def _prepare_expression(expr: str) -> str:
    """Translate an Excel-flavoured expression into Python syntax."""
    s = expr.strip()
    if s.startswith("="):          # worksheet formulas start with '='
        s = s[1:]
    s = s.replace("^", "**")       # Excel power operator
    return s


def _compile_expression(expr: str, varnames):
    """
    Compile a string expression into a callable of the given variables.

    The maths namespace is built once and the variable entries rewritten per
    call. Solvers evaluate an objective tens of thousands of times, so copying
    a 35-entry dict each time is real cost -- and far more of it in a slower
    runtime such as the browser.
    """
    code = compile(_prepare_expression(expr), "<comxl-expression>", "eval")
    env = dict(SAFE_NAMESPACE)
    # the three spellings each variable answers to, resolved once
    aliases = [(n, n.upper(), n.lower()) for n in varnames]
    empty_builtins = {"__builtins__": {}}

    def _f(*args):
        for (a, b, c), value in zip(aliases, args):
            env[a] = env[b] = env[c] = value
        return float(eval(code, empty_builtins, env))

    _f.__doc__ = f"Compiled from expression: {expr}"
    return _f


_MULTIVAR_HINT = (
    "The one-dimensional methods in comxl.gradientsearch take a function of a "
    "SINGLE variable, f(x) -> float, and search between two scalar bounds.\n\n"
    "The objective supplied looks like a function of several variables "
    "(it indexes or unpacks its argument, e.g. `lambda x: x[0]**2 + x[1]**2`).\n\n"
    "For a multivariable objective use one of:\n"
    "    comxl.directsearch.neldermead(f, x0)      # simplex, derivative-free\n"
    "    comxl.directsearch.powell(f, x0)          # conjugate directions\n"
    "    comxl.constrained.sqp(...) / .kkt(...)    # with constraints\n"
    "each of which takes a starting VECTOR x0 rather than bounds.\n\n"
    "To minimise a multivariable f along a single line with a 1D method, pass "
    "a function of the step length, e.g.\n"
    "    comxl.gradientsearch.goldensection(lambda t: f(x0 + t*d), 0, 1)"
)


def _is_vector_indexing_error(exc: Exception) -> bool:
    """
    True when the exception looks like a vector objective called with a scalar.

    Covers ``x[0]`` on a float (TypeError: not subscriptable) and unpacking a
    float (TypeError: cannot unpack / not iterable).
    """
    msg = str(exc).lower()
    return any(s in msg for s in (
        "not subscriptable",
        "cannot unpack",
        "is not iterable",
        "unsubscriptable",
    ))


def make_function(f, varname: str = "x"):
    """
    Return a single-variable callable ``g(x) -> float``.

    Accepts either a Python callable or a string expression. The callable must
    take one scalar; passing a multivariable objective raises a ``TypeError``
    naming the multivariable solvers to use instead, rather than failing
    obscurely inside the user's own function.

    >>> g = make_function(lambda x: x**2 - 4*x)
    >>> g(2.0)
    -4.0
    >>> h = make_function("X_1^2 - 4*X_1")     # worksheet-style string
    >>> h(2.0)
    -4.0

    >>> make_function(lambda x1, x2: x1**2 + x2**2)   # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: This objective takes 2 required arguments...
    """
    if callable(f):
        # A callable needing two or more arguments is unambiguously
        # multivariable, so reject it up front rather than at first call.
        try:
            sig = inspect.signature(f)
            required = [
                p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]
            n_required = len(required)
        except (TypeError, ValueError):        # builtins, C functions
            n_required = 1

        if n_required > 1:
            raise TypeError(
                f"This objective takes {n_required} required arguments, but a "
                f"one-dimensional method needs a function of one variable.\n\n"
                + _MULTIVAR_HINT
            )

        def _g(x):
            try:
                return float(f(x))
            except TypeError as exc:
                if _is_vector_indexing_error(exc):
                    raise TypeError(
                        f"Could not evaluate the objective at x = {x!r}.\n\n"
                        + _MULTIVAR_HINT
                    ) from exc
                raise
            except IndexError as exc:
                # An IndexError may well be a vector objective, but it can
                # equally come from a genuine 1D function indexing a table, so
                # keep the original exception type and just add the hint.
                raise IndexError(
                    f"{exc} -- raised while evaluating the objective at "
                    f"x = {x!r}, which a one-dimensional method passes as a "
                    f"scalar. If this objective is a function of several "
                    f"variables, see comxl.directsearch instead."
                ) from exc

        return _g

    if isinstance(f, str):
        # accept x, X, X_1, X1 and the caller's chosen name
        names = [varname, "x", "X", "X_1", "X1", "x_1", "x1"]
        compiled = _compile_expression(f, names)

        def _from_string(x):
            try:
                return compiled(*([x] * len(names)))
            except NameError as exc:
                raise NameError(
                    f"{exc}. A one-dimensional objective may only use the "
                    f"single variable x (also accepted as X, X_1 or X1). "
                    f"An expression in two or more variables needs a "
                    f"multivariable solver -- see comxl.directsearch."
                ) from exc

        return _from_string

    raise TypeError(
        "Objective must be a callable or a string expression, "
        f"got {type(f).__name__}"
    )


def make_vector_function(f, n: int | None = None, varnames=None):
    """
    Return a multivariable callable ``g(x_array) -> float``.

    Handles three common ways of writing an objective:

    * ``lambda x: x[0]**2 + x[1]**2``      -- takes one array
    * ``lambda x1, x2: x1**2 + x2**2``     -- takes separate scalars
    * ``"X_1^2 + X_2^2"``                  -- worksheet-style string

    The arity of a callable is detected automatically, so users are free to
    write whichever form reads best for their problem.
    """
    if isinstance(f, str):
        if n is None:
            raise ValueError("Number of variables `n` is required for string objectives.")
        if varnames is None:
            varnames = [f"X_{i + 1}" for i in range(n)]
        compiled = _compile_expression(f, varnames)

        def _from_string(x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            return compiled(*x)

        return _from_string

    if not callable(f):
        raise TypeError(
            "Objective must be a callable or a string expression, "
            f"got {type(f).__name__}"
        )

    # Decide whether the callable wants one array or n scalars.
    try:
        sig = inspect.signature(f)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        n_params = len(params)
        has_varargs = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    except (TypeError, ValueError):       # builtins, C functions
        n_params, has_varargs = 1, False

    if n_params == 1 and not has_varargs:
        def _from_array(x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            value = f(x)
            return _as_scalar(value, x)
        return _from_array

    def _from_scalars(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return _as_scalar(f(*x), x)

    return _from_scalars


def _as_scalar(value, x):
    """
    Coerce an objective's return value to a float, explaining the common
    failure where a scalar formula is applied elementwise to the whole vector.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise TypeError(
            f"The objective returned {arr.size} values at x = "
            f"{np.array2string(x, precision=4)}, but an objective must return "
            f"a single number.\n\n"
            f"This usually means a formula written for one variable is being "
            f"applied to the whole vector -- `lambda x: x**2` squares every "
            f"element and returns a vector. Write the objective in terms of "
            f"the individual components instead, e.g.\n"
            f"    lambda x: x[0]**2 + x[1]**2\n"
            f"    lambda x1, x2: x1**2 + x2**2\n"
            f"Or, if the problem really has one variable, use a "
            f"one-dimensional method from comxl.gradientsearch."
        )
    return float(arr.reshape(()))


# ----------------------------------------------------------------------------
# Machine precision (ports MachinePrecision() from Optimization.bas)
# ----------------------------------------------------------------------------
def machine_precision() -> float:
    """Smallest x with 1 + x != 1, found by repeated halving as in the VBA."""
    x = 1.0
    while 1.0 + x != 1.0:
        x = 0.5 * x
    return 2.0 * x


# ----------------------------------------------------------------------------
# Iteration history and results
# ----------------------------------------------------------------------------
class StepTable:
    """
    Accumulates one row per iteration.

    Stands in for the block of worksheet cells the VBA forms filled in as they
    iterated, so the teaching value of watching the method work is preserved.
    """

    def __init__(self, columns):
        self.columns = list(columns)
        self.rows: list[list] = []

    def add(self, *values):
        if len(values) != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} values for columns "
                f"{self.columns}, got {len(values)}"
            )
        self.rows.append(list(values))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def __len__(self):
        return len(self.rows)


class Result:
    """
    What every solver returns.

    Attributes
    ----------
    x : float or ndarray
        Location of the optimum (or the root, for equation solvers).
    f : float
        Objective value at ``x``.
    iterations : int
        Iterations actually performed.
    converged : bool
        Whether the convergence test was satisfied before the iteration cap.
    message : str
        Human-readable termination reason.
    steps : pandas.DataFrame
        Full iteration history -- always populated, regardless of
        ``show_steps``.
    method : str
        Name of the method that produced the result.
    extras : dict
        Method-specific extras (gradient, Hessian, tableaux, multipliers...).
    """

    def __init__(self, x, f, iterations=0, converged=True, message="",
                 steps=None, method="", **extras):
        self.x = x
        self.f = f
        self.iterations = iterations
        self.converged = bool(converged)
        self.message = message
        self.method = method
        self.extras = extras
        if isinstance(steps, StepTable):
            self.steps = steps.to_frame()
        elif steps is None:
            self.steps = pd.DataFrame()
        else:
            self.steps = steps

    # convenience -----------------------------------------------------------
    def _fmt_x(self):
        if np.isscalar(self.x):
            return f"{self.x:.8g}"
        return np.array2string(np.asarray(self.x, dtype=float),
                               precision=6, separator=", ")

    def __repr__(self):
        status = "converged" if self.converged else "NOT converged"
        head = f"{self.method} result ({status} in {self.iterations} iterations)"
        fval = 0.0 if self.f == 0 else self.f       # avoid printing "-0"
        body = f"  x = {self._fmt_x()}\n  f(x) = {fval:.8g}"
        if self.message:
            body += f"\n  {self.message}"
        return f"{head}\n{body}"

    def display(self, float_format="%.6g"):
        """Print the iteration table followed by the summary."""
        if not self.steps.empty:
            print(self.steps.to_string(index=False,
                                       float_format=lambda v: float_format % v))
            print()
        print(self)

    def plot_convergence(self, column=None, ax=None, **kwargs):
        """
        Plot an iteration-history column (defaults to the error/objective
        column if the method recorded one).
        """
        import matplotlib.pyplot as plt

        if self.steps.empty:
            raise ValueError("No iteration history was recorded.")
        if column is None:
            for candidate in ("Error", "L", "f(X_min)", "f(x)", "OF value"):
                if candidate in self.steps.columns:
                    column = candidate
                    break
            else:
                column = self.steps.columns[-1]
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.steps.index, self.steps[column], marker="o", **kwargs)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(column)
        ax.set_title(f"{self.method}: {column} vs iteration")
        ax.grid(alpha=0.3)
        return ax


def _finish(result: Result, show_steps: bool) -> Result:
    """Print the table if requested, then hand the result back."""
    if show_steps:
        result.display()
    return result
