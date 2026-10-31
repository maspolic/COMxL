"""
COMxL-Py
========

A pure-Python port of the COMxL Excel/VBA optimisation workbook.

The original was a set of VBA user forms driving worksheet formulas: you typed
a function into a cell, pointed a form at it, and the macro wrote its
iterations back onto the sheet. This package keeps the algorithms and the
step-by-step output, but you supply the objective as an ordinary Python
function.

Modules
-------
``gradientsearch``
    One-dimensional search: ``goldensection``, ``fibonacci``, ``dichotomous``,
    ``intervalhalving``, ``exhaustive``, ``secant``, ``newtonraphson``,
    ``divide_conquer``.
``directsearch``
    Multivariable unconstrained: ``neldermead``, ``powell``.
``constrained``
    ``sqp``, ``quasinewton``, ``kkt``, ``lagrangian``.
``linprog``
    ``simplex``, ``feasible``, ``constraint_line``.
``calculus``
    ``gradient``, ``hessian``, ``jacobian``, ``first_derivative``,
    ``second_derivative``.
``matrixops``
    ``inverse``, ``transpose``, ``multiply``, ``gauss_pivot``,
    ``eigenvalues``, ``determinant``, ``row_echelon``.
``plotting``
    ``plot_function``, ``plot_contours``, ``plot_constraints_2d``,
    ``set_style``.

Quick start
-----------
>>> from comxl import gradientsearch
>>> r = gradientsearch.goldensection(lambda x: x**2 - 4*x + 3, 0, 5)
...                                              # doctest: +SKIP

Every solver prints its iteration table by default. Pass ``show_steps=False``
for the final answer only; the table is still available as ``r.steps``.

>>> r = gradientsearch.goldensection(lambda x: x**2 - 4*x + 3, 0, 5,
...                                  tol=1e-6, show_steps=False)
>>> round(r.x, 3)
2.0
"""

from . import calculus
from . import constrained
from . import directsearch
from . import gradientsearch
from . import linprog
from . import matrixops
from . import plotting
from .core import Result, make_function, make_vector_function, machine_precision

__version__ = "0.1.0"

__all__ = [
    "gradientsearch",
    "directsearch",
    "constrained",
    "linprog",
    "calculus",
    "matrixops",
    "plotting",
    "Result",
    "make_function",
    "make_vector_function",
    "machine_precision",
    "__version__",
]
