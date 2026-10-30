"""
Constrained optimisation and nonlinear equation solving.

Ports of:
  * `sqp`         <- SQP (UserForm2.frm, "Sequential Quadratic Programming")
  * `quasinewton` <- QUASINEWTON (UserForm16.frm) and Run_Solver (UserForm1.frm)
  * `lagrangian`  <- the Lagrangian assembly used by UserForm1/UserForm2
  * `kkt`         <- the case analysis of UserForm16.frm
    ("Karush-Kuhn-Tucker Solver")

All of these reduce to the same core idea: build the Lagrangian, then solve
the stationarity conditions as a system of nonlinear equations by Newton's
method with a numerical Jacobian and reduced (halved) steps for robustness.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .calculus import gradient_vector, hessian_matrix
from .core import Result, StepTable, make_vector_function
from .matrixops import gauss_pivot, inverse

__all__ = ["sqp", "quasinewton", "lagrangian", "kkt"]

SMALL = 1.0e-12


# ----------------------------------------------------------------------------
# Lagrangian assembly
# ----------------------------------------------------------------------------
def lagrangian(objective, constraints, n_vars: int):
    """
    Build L(x, lambda) = f(x) + sum_j lambda_j * g_j(x).

    Returns a callable of the extended vector z = [x_1..x_n, lam_1..lam_m], so
    it can be handed straight to `sqp` or `quasinewton`.

    Parameters
    ----------
    objective : callable or str
        f(x), a function of the n decision variables.
    constraints : sequence of callables/strings
        Each g_j(x), written so that the constraint is g_j(x) = 0.
    n_vars : int
        Number of decision variables n.
    """
    f = make_vector_function(objective, n=n_vars)
    gs = [make_vector_function(c, n=n_vars) for c in constraints]
    m = len(gs)

    def L(z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        x, lam = z[:n_vars], z[n_vars:n_vars + m]
        return f(x) + sum(lam[j] * gs[j](x) for j in range(m))

    L.n_vars = n_vars
    L.n_constraints = m
    L.objective = f
    L.constraints = gs
    return L


# ----------------------------------------------------------------------------
# Sequential quadratic programming  (UserForm2.frm)
# ----------------------------------------------------------------------------
def sqp(lagrange_function, x0, objective=None, n_vars: int | None = None,
        tol: float = 1.0e-6, max_iter: int = 100, show_steps: bool = True) -> Result:
    """
    Sequential quadratic programming via Newton's method on the Lagrangian.

    Each step solves the KKT system by taking a full Newton step on the
    stationarity conditions:

        z <- z - H(L)^-1 . grad(L)

    where z stacks the decision variables and the Lagrange multipliers, and
    both the gradient and Hessian are central-difference approximations. This
    is the textbook SQP of the original workbook: no line search, no BFGS
    update, no active-set logic -- so it converges fast near a solution and
    needs a decent starting guess.

    Convergence is judged on the norm of the *decision-variable* part of the
    gradient, matching the VBA.

    Parameters
    ----------
    lagrange_function : callable or str
        L(x, lambda) as a function of the full vector z. Build one with
        `lagrangian(...)` if you have f and g separately.
    x0 : array-like
        Starting values for [x_1..x_n, lam_1..lam_m].
    objective : callable or str, optional
        f(x) alone, reported alongside L in the iteration table.
    n_vars : int, optional
        Number of decision variables (defaults to all of them, i.e. no
        multipliers). Needed to split z correctly.
    tol : float
        Convergence tolerance on ||grad_x L|| (form default field).
    max_iter : int
        Iteration cap.

    Returns
    -------
    Result
        ``.x`` is the full z vector; ``.extras['variables']`` and
        ``.extras['multipliers']`` split it for convenience.
    """
    z = np.atleast_1d(np.asarray(x0, dtype=float))
    k = len(z)
    nx = n_vars if n_vars is not None else getattr(lagrange_function, "n_vars", k)

    L = make_vector_function(lagrange_function, n=k)
    obj = make_vector_function(objective, n=nx) if objective is not None else None

    cols = (["Iteration"] + [f"X_{i + 1}" for i in range(nx)] +
            [f"lambda_{i + 1}" for i in range(k - nx)] + ["OF value", "Error"])
    steps = StepTable(cols)

    def of_value(zz):
        if obj is not None:
            return obj(zz[:nx])
        return L(zz)

    grad = gradient_vector(L, z, n=k)
    err = float(np.sqrt(np.sum(grad[:nx] ** 2)))
    steps.add(0, *z, of_value(z), err)

    j, converged = 0, False
    for j in range(1, max_iter + 1):
        H = hessian_matrix(L, z, n=k)
        grad = gradient_vector(L, z, n=k)

        try:
            step = gauss_pivot(H, grad, show_steps=False)
        except np.linalg.LinAlgError:
            H = H + SMALL * np.eye(k)          # nudge a singular Hessian
            step = inverse(H, show_steps=False) @ grad

        z = z - step

        grad = gradient_vector(L, z, n=k)
        err = float(np.sqrt(np.sum(grad[:nx] ** 2)))
        steps.add(j, *z, of_value(z), err)

        if err <= tol:
            converged = True
            break

    res = Result(
        x=z, f=of_value(z), iterations=j, converged=converged,
        message=("Converged." if converged else
                 f"Not converged after {max_iter} iterations - "
                 "try a different starting guess."),
        steps=steps, method="Sequential quadratic programming",
        variables=z[:nx], multipliers=z[nx:], gradient=grad,
    )
    if show_steps:
        res.display()
    return res


# ----------------------------------------------------------------------------
# Reduced-step quasi-Newton solver  (UserForm16.frm / UserForm1.frm)
# ----------------------------------------------------------------------------
def quasinewton(funcs, x0, tol: float = 1.0e-8, max_iter: int = 999,
                show_steps: bool = True) -> Result:
    """
    Solve a system of nonlinear equations f_i(x) = 0.

    Reduced-step Newton method: the Jacobian is approximated by finite
    differences, the linear system is solved by Gauss elimination with maximum
    column pivoting and one error-correction pass, and if a full Newton step
    does not reduce the residual norm the step is halved up to six times. That
    damping is what makes the method usable from mediocre starting guesses.

    This is the engine behind the KKT solver: the stationarity conditions of a
    constrained problem are exactly such a system.

    Parameters
    ----------
    funcs : sequence of callables/strings, or one callable returning a vector
        The equations, each written to equal zero.
    x0 : array-like
        Starting guess. Must have as many entries as there are equations.
    tol : float
        Relative convergence tolerance (1e-8 in the VBA).
    max_iter : int
        Newton iteration cap (999 in the VBA).

    Returns
    -------
    Result
        ``.x`` is the root; ``.extras['residuals']`` the final f(x).
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x)

    if callable(funcs) and not isinstance(funcs, (list, tuple)):
        def F(v):
            return np.atleast_1d(np.asarray(funcs(v), dtype=float))
    else:
        adapted = [make_vector_function(fn, n=n) for fn in funcs]

        def F(v):
            return np.array([fn(v) for fn in adapted], dtype=float)

    if len(F(x)) != n:
        raise ValueError(
            f"Number of equations ({len(F(x))}) must match number of "
            f"unknowns ({n})."
        )

    steps = StepTable(["Iteration"] + [f"X_{i + 1}" for i in range(n)] +
                      ["||f(x)||", "Residual"])

    fnorm1 = float(np.sqrt(np.sum(F(x) ** 2)))
    steps.add(0, *x, fnorm1, np.nan)

    k, converged = 0, False
    for k in range(1, max_iter + 1):
        fx = F(x)

        # numerical Jacobian, VBA step: dx = 1e-5 |x_j| + 1e-5
        J = np.zeros((n, n))
        for j in range(n):
            dx = 1.0e-5 * abs(x[j]) + 1.0e-5
            xf, xb = x.copy(), x.copy()
            xf[j] += dx
            xb[j] -= dx
            J[:, j] = (F(xf) - F(xb)) / (2.0 * dx)

        try:
            dxv = gauss_pivot(J, fx, show_steps=False)
        except np.linalg.LinAlgError:
            res = Result(x=x, f=fnorm1, iterations=k, converged=False,
                         message="Singular Jacobian - try different guesses.",
                         steps=steps, method="Quasi-Newton solver",
                         residuals=F(x))
            if show_steps:
                res.display()
            return res

        residual = max(abs(dxv[j]) / (abs(x[j]) + SMALL) for j in range(n))
        x_new = x - dxv

        if residual < tol:
            x = x_new
            converged = True
            steps.add(k, *x, float(np.sqrt(np.sum(F(x) ** 2))), residual)
            break

        # reduced-step search: halve until the residual norm improves
        for m in range(1, 7):
            fnorm = float(np.sqrt(np.sum(F(x_new) ** 2)))
            if fnorm < fnorm1:
                fnorm1 = fnorm
                break
            x_new = x - dxv / (2.0 ** m)

        x = x_new
        steps.add(k, *x, float(np.sqrt(np.sum(F(x) ** 2))), residual)

    res = Result(
        x=x, f=float(np.sqrt(np.sum(F(x) ** 2))), iterations=k,
        converged=converged,
        message=("Converged." if converged else
                 f"Not converged after {max_iter} iterations."),
        steps=steps, method="Quasi-Newton solver", residuals=F(x),
    )
    if show_steps:
        res.display()
    return res


# ----------------------------------------------------------------------------
# Karush-Kuhn-Tucker solver  (UserForm16.frm)
# ----------------------------------------------------------------------------
def kkt(objective, n_vars: int, inequality=(), equality=(), x0=None,
        tol: float = 1.0e-8, maximize: bool = False,
        show_steps: bool = True) -> Result:
    """
    Karush-Kuhn-Tucker solver by enumeration of active sets.

    For the problem

        minimise   f(x)
        subject to g_j(x) <= 0,  h_i(x) = 0

    the KKT conditions are

        grad f + sum_j mu_j grad g_j + sum_i lam_i grad h_i = 0
        mu_j g_j(x) = 0,   mu_j >= 0,   g_j(x) <= 0,   h_i(x) = 0

    The complementary-slackness products make the system combinatorial: each
    inequality is either active (g_j = 0, mu_j free) or inactive (mu_j = 0).
    This routine enumerates all 2^m cases exactly as the VBA form did through
    its case1 / case2_0 / case2_1 / case3_2 branches, solves each resulting
    square system with `quasinewton`, and keeps every candidate that satisfies
    primal and dual feasibility.

    Enumeration is exponential in the number of inequalities, so this is
    intended for the small teaching problems the workbook targeted -- with
    more than about ten inequalities, use an active-set or interior-point
    method instead.

    Parameters
    ----------
    objective : callable or str
        f(x).
    n_vars : int
        Number of decision variables.
    inequality : sequence
        Constraints g_j(x) <= 0.
    equality : sequence
        Constraints h_i(x) = 0.
    x0 : array-like, optional
        Starting guess for x (defaults to ones).
    maximize : bool
        Maximise f instead of minimising.

    Returns
    -------
    Result
        ``.x`` is the best KKT point found; ``.steps`` lists every case tried,
        with its active set, feasibility verdict and objective value.
    """
    sign = -1.0 if maximize else 1.0
    f_raw = make_vector_function(objective, n=n_vars)

    def f(x):
        return sign * f_raw(x)

    gs = [make_vector_function(c, n=n_vars) for c in inequality]
    hs = [make_vector_function(c, n=n_vars) for c in equality]
    m, p = len(gs), len(hs)

    if x0 is None:
        x0 = np.ones(n_vars)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))

    rows, candidates = [], []

    for active in itertools.product([False, True], repeat=m):
        n_active = sum(active)
        dim = n_vars + p + n_active

        def system(z, active=active):
            x = z[:n_vars]
            lam = z[n_vars:n_vars + p]
            mu_active = z[n_vars + p:]

            gL = gradient_vector(f, x, n=n_vars)
            for i in range(p):
                gL = gL + lam[i] * gradient_vector(hs[i], x, n=n_vars)
            idx = 0
            for j in range(m):
                if active[j]:
                    gL = gL + mu_active[idx] * gradient_vector(gs[j], x, n=n_vars)
                    idx += 1

            eqs = list(gL)
            eqs += [hs[i](x) for i in range(p)]
            eqs += [gs[j](x) for j in range(m) if active[j]]
            return np.array(eqs, dtype=float)

        z0 = np.concatenate([x0, np.ones(p + n_active) * 0.5])
        label = "{" + ", ".join(f"g{j + 1}" for j in range(m) if active[j]) + "}"
        if n_active == 0:
            label = "{} (interior)"

        try:
            sol = quasinewton(system, z0, tol=tol, show_steps=False)
        except Exception as exc:                       # singular / no solution
            rows.append([label, "no solution", np.nan, str(exc)[:40]] +
                        [np.nan] * n_vars)
            continue

        z = sol.x
        x = z[:n_vars]
        mu_active = z[n_vars + p:]

        primal = all(gs[j](x) <= 1e-6 for j in range(m)) and \
                 all(abs(hs[i](x)) <= 1e-6 for i in range(p))
        dual = all(mu >= -1e-6 for mu in mu_active)
        ok = sol.converged and primal and dual

        verdict = ("KKT point" if ok else
                   "infeasible" if not primal else
                   "mu < 0" if not dual else "not converged")
        fval = sign * f(x)
        rows.append([label, verdict, fval, f"{sol.iterations} its"] + list(x))
        if ok:
            candidates.append((fval, x, z, label))

    steps = pd.DataFrame(
        rows,
        columns=["Active set", "Status", "f(x)", "Newton"] +
                [f"X_{i + 1}" for i in range(n_vars)],
    )

    if candidates:
        best = max(candidates, key=lambda c: c[0]) if maximize \
            else min(candidates, key=lambda c: c[0])
        fbest, xbest, zbest, lbest = best
        message = f"Best KKT point found with active set {lbest}."
        converged = True
    else:
        fbest, xbest, zbest, lbest = np.nan, x0, x0, "none"
        message = ("No case satisfied the KKT conditions. Try a different "
                   "starting guess or check the constraint signs "
                   "(all inequalities must be written as g(x) <= 0).")
        converged = False

    res = Result(
        x=xbest, f=fbest, iterations=2 ** m, converged=converged,
        message=message, steps=steps, method="KKT solver",
        active_set=lbest, full_solution=zbest, cases=steps,
    )
    if show_steps:
        res.display()
    return res
