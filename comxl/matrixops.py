"""
Matrix utilities.

Ports of:
  * `matrixadd_subtract`, `matrix_inv`, `matrix_trans`  (MatrixSubs.bas)
  * `GAUSSPIVOT1` / `gausspivoting`                     (Helper_Sub.bas, Helper_Sub_2.bas)
  * `EigenValues_SUB`, `quadfun`                        (Helper_Sub_2.bas)
  * `RowEchelon`, `RowOpCalc`, `ClearMatCol`, `Trans`   (LinearProgramming.bas)

The algorithms are reimplemented rather than delegated to numpy.linalg: this
package exists to *be* the numerical toolkit, so Gauss elimination and the
eigenvalue routine are written out in full. numpy is used only for array
storage and arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Result

__all__ = [
    "add", "subtract", "multiply", "transpose",
    "inverse", "determinant", "gauss_pivot", "row_echelon",
    "eigenvalues", "row_operation",
]

SMALL = 1.0e-12       # `small` constant used throughout the VBA


# ----------------------------------------------------------------------------
# Elementary operations
# ----------------------------------------------------------------------------
def _as_matrix(a):
    m = np.atleast_2d(np.asarray(a, dtype=float))
    return m


def add(a, b, show_steps: bool = True):
    """Matrix addition. Shapes must match, as enforced in the VBA."""
    A, B = _as_matrix(a), _as_matrix(b)
    if A.shape != B.shape:
        raise ValueError(f"Shapes must match for addition: {A.shape} vs {B.shape}")
    C = A + B
    res = Result(x=C, f=float(np.sum(C)), steps=pd.DataFrame(C),
                 method="Matrix addition", result=C)
    if show_steps:
        res.display()
    return C


def subtract(a, b, show_steps: bool = True):
    """Matrix subtraction (the `typ = -1` branch of `matrixadd_subtract`)."""
    A, B = _as_matrix(a), _as_matrix(b)
    if A.shape != B.shape:
        raise ValueError(f"Shapes must match for subtraction: {A.shape} vs {B.shape}")
    C = A - B
    res = Result(x=C, f=float(np.sum(C)), steps=pd.DataFrame(C),
                 method="Matrix subtraction", result=C)
    if show_steps:
        res.display()
    return C


def multiply(a, b, show_steps: bool = True):
    """
    Matrix multiplication using the explicit triple loop of the VBA
    (`typ = 0` branch), vectorised per output element.
    """
    A, B = _as_matrix(a), _as_matrix(b)
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Inner dimensions must agree: {A.shape} x {B.shape}"
        )
    n1, m1, m2 = A.shape[0], B.shape[0], B.shape[1]
    C = np.zeros((n1, m2))
    for i in range(n1):
        for j in range(m2):
            total = 0.0
            for k in range(m1):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    res = Result(x=C, f=float(np.sum(C)), steps=pd.DataFrame(C),
                 method="Matrix multiplication", result=C)
    if show_steps:
        res.display()
    return C


def transpose(a, show_steps: bool = True):
    """Matrix transpose (`matrix_trans` / `Trans`)."""
    A = _as_matrix(a)
    T = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            T[j, i] = A[i, j]
    res = Result(x=T, f=0.0, steps=pd.DataFrame(T),
                 method="Matrix transpose", result=T)
    if show_steps:
        res.display()
    return T


# ----------------------------------------------------------------------------
# Inverse by Gauss-Jordan on the augmented matrix (matrix_inv)
# ----------------------------------------------------------------------------
def inverse(a, show_steps: bool = True):
    """
    Matrix inverse by Gauss-Jordan elimination on [A | I].

    Follows `matrix_inv` in MatrixSubs.bas, including its handling of a zero
    pivot (add a lower row that has a non-zero entry in that column).
    """
    A = _as_matrix(a)
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got {A.shape}")

    aug = np.zeros((n, 2 * n))
    aug[:, :n] = A
    for k in range(n):
        aug[k, n + k] = 1.0

    for k in range(n):
        if aug[k, k] == 0.0:
            for i in range(k + 1, n):
                if A[i, k] != 0.0:
                    aug[k, :] = aug[k, :] + aug[i, :]
                    break
            else:
                raise np.linalg.LinAlgError("The matrix inverse does not exist.")
        c = 1.0 / aug[k, k]
        aug[k, :] *= c
        for i in range(n):
            if i != k:
                d = aug[i, k]
                aug[i, :] -= d * aug[k, :]

    inv = aug[:, n:]
    res = Result(x=inv, f=0.0, steps=pd.DataFrame(inv),
                 method="Matrix inverse", result=inv)
    if show_steps:
        res.display()
    return inv


def determinant(a) -> float:
    """
    Determinant via row echelon reduction, tracking the product of pivots and
    the number of row switches (`RowEchelon` in LinearProgramming.bas).
    """
    A = _as_matrix(a).copy()
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix must be square, got {A.shape}")

    prod, switches = 1.0, 0
    toprow, col = 0, 0
    while toprow < n and col < m:
        row = toprow
        while row < n - 1 and A[row, col] == 0:
            row += 1
        if A[row, col] != 0:
            prod *= A[row, col]
            if row != toprow:
                switches += 1
                A[[row, toprow]] = A[[toprow, row]]
            pivot = A[toprow, col]
            A[toprow, :] /= pivot
            for i in range(n):
                if i != toprow:
                    A[i, :] -= A[i, col] * A[toprow, :]
            toprow += 1
        col += 1
    return prod * ((-1.0) ** switches)


# ----------------------------------------------------------------------------
# Gauss elimination with maximum column pivoting and one error correction
# ----------------------------------------------------------------------------
def gauss_pivot(a, b, show_steps: bool = True):
    """
    Solve A x = b by Gauss elimination with maximum column pivoting.

    Faithful port of `GAUSSPIVOT1` (Helper_Sub.bas): the system is solved
    twice -- once normally, then again on the residual b - A x -- and the two
    solutions are summed. That single iterative refinement step recovers much
    of the accuracy lost to round-off in ill-conditioned systems.
    """
    A0 = _as_matrix(a)
    b0 = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    n = A0.shape[0]
    if A0.shape[1] != n or len(b0) != n:
        raise ValueError("A must be square and b must match its size.")

    A = A0.copy()
    bb = b0.copy()
    b1 = b0.copy()
    r = np.zeros(n, dtype=int)

    # forward elimination, storing multipliers in the lower triangle
    for k in range(n - 1):
        d = 0.0
        for i in range(k, n):
            c = abs(A[i, k])
            if d < c:
                d, r[k] = c, i
        if abs(d) < SMALL:
            raise np.linalg.LinAlgError(
                "Pivot is effectively zero - the system is singular. "
                "Try different guess values."
            )
        r1 = r[k]
        if r1 != k:
            A[[k, r1], k:] = A[[r1, k], k:]
        for i in range(k + 1, n):
            A[i, k] = -A[i, k] / A[k, k]
            A[i, k + 1:] += A[i, k] * A[k, k + 1:]

    # The VBA only tested pivots 1..n-1; the last one can still be zero for a
    # rank-deficient system, which would otherwise divide by zero silently.
    if abs(A[n - 1, n - 1]) < SMALL:
        raise np.linalg.LinAlgError(
            "Final pivot is effectively zero - the system is singular. "
            "Try different guess values."
        )

    x = np.zeros(n)
    x1 = np.zeros((n, 2))
    for pass_ in range(2):
        for k in range(n - 1):
            r1 = r[k]
            if r1 != k:
                bb[k], bb[r1] = bb[r1], bb[k]
            for i in range(k + 1, n):
                bb[i] += A[i, k] * bb[k]

        x[n - 1] = bb[n - 1] / A[n - 1, n - 1]
        x1[n - 1, pass_] = x[n - 1]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                bb[i] -= A[i, j] * x[j]
            x[i] = bb[i] / A[i, i]
            x1[i, pass_] = x[i]

        if pass_ == 0:                    # residual for the correction pass
            b1 = b1 - A0 @ x
            bb = b1.copy()

    solution = x1[:, 0] + x1[:, 1]
    steps = pd.DataFrame({
        "Variable": [f"x_{i + 1}" for i in range(n)],
        "First pass": x1[:, 0],
        "Correction": x1[:, 1],
        "Solution": solution,
        "Residual": b0 - A0 @ solution,
    })
    res = Result(x=solution, f=float(np.linalg.norm(b0 - A0 @ solution)),
                 iterations=2, steps=steps, method="Gauss elimination (pivoted)",
                 solution=solution)
    if show_steps:
        res.display()
    return solution


def row_echelon(a, show_steps: bool = True):
    """Row-echelon form (`RowEchelon` in LinearProgramming.bas)."""
    A = _as_matrix(a).copy()
    n, m = A.shape
    toprow, col = 0, 0
    while toprow < n and col < m:
        row = toprow
        while row < n - 1 and A[row, col] == 0:
            row += 1
        if A[row, col] != 0:
            if row != toprow:
                A[[row, toprow]] = A[[toprow, row]]
            A[toprow, :] /= A[toprow, col]
            for i in range(n):
                if i != toprow:
                    A[i, :] = np.round(A[i, :] - A[i, col] * A[toprow, :], 10)
            toprow += 1
        col += 1
    if show_steps:
        print(pd.DataFrame(A).to_string(index=False))
    return A


def row_operation(mat, optype: int, scalar: float = 1.0, rowa: int = 0, rowb: int = 0):
    """
    Elementary row operation (`RowOpCalc`), 0-indexed.

    optype 1 : swap rowa and rowb
    optype 2 : rowa <- scalar * rowa
    optype 3 : rowb <- rowb + scalar * rowa   (rounded to 10 dp, as in the VBA)
    """
    A = _as_matrix(mat).copy()
    if optype == 1:
        A[[rowa, rowb]] = A[[rowb, rowa]]
    elif optype == 2:
        A[rowa, :] = scalar * A[rowa, :]
    elif optype == 3:
        A[rowb, :] = np.round(A[rowb, :] + scalar * A[rowa, :], 10)
    else:
        raise ValueError("optype must be 1 (swap), 2 (scale) or 3 (add multiple)")
    return A


# ----------------------------------------------------------------------------
# Eigenvalues: characteristic polynomial + Bairstow root finding
# ----------------------------------------------------------------------------
def _quadfun(b: float, c: float):
    """Roots of z^2 - b z - c (the `quadfun` helper)."""
    s = np.sign(b) if b != 0 else 1.0
    d = b * b + 4.0 * c
    if d > 0:
        r1 = 0.5 * (b + s * np.sqrt(d))
        r2 = -c / r1 if r1 != 0 else 0.0
        return (r1, 0.0), (r2, 0.0)
    r1 = 0.5 * b
    i1 = 0.5 * np.sqrt(abs(d))
    return (r1, i1), (r1, -i1)


def eigenvalues(a, tol: float = 1.0e-8, show_steps: bool = True, seed: int | None = 0):
    """
    Eigenvalues of a square matrix.

    Reproduces `EigenValues_SUB` (Helper_Sub_2.bas): the characteristic
    polynomial is recovered by evaluating det(A - lambda I) at n+1 sample
    values of lambda and solving the resulting Vandermonde system, then the
    polynomial is factored by Bairstow's method (quadratic deflation).

    Because sample points are drawn at random the VBA result varies slightly
    between runs; ``seed`` makes that reproducible here.
    """
    A = _as_matrix(a)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError(f"Matrix must be square, got {A.shape}")

    rng = np.random.default_rng(seed)
    np1 = n + 1

    # sample the characteristic polynomial at np1 distinct points
    lam = 10.0 * (rng.random(np1) - 0.5)
    det = np.array([determinant(A - l * np.eye(n)) for l in lam])

    # Vandermonde system for the polynomial coefficients
    V = np.vander(lam, np1, increasing=True)
    xc = gauss_pivot(V, det, show_steps=False)

    # Bairstow deflation down to a quadratic or linear factor
    coeffs = xc.astype(float).copy()
    re = np.zeros(np1 + 1)
    b_, c_ = 0.0, 0.0
    nc = np1
    guard = 0
    while nc >= 4 and guard < 500:
        guard += 1
        p = np.zeros(nc + 2)
        q = np.zeros(nc + 2)
        for _ in range(500):
            p[nc - 1] = coeffs[nc - 1]
            p[nc - 2] = coeffs[nc - 2] + b_ * p[nc - 1]
            q[nc - 1] = p[nc - 1]
            q[nc - 2] = p[nc - 2] + b_ * q[nc - 1]
            for j in range(nc - 3, -1, -1):
                p[j] = coeffs[j] + b_ * p[j + 1] + c_ * p[j + 2]
                q[j] = p[j] + b_ * q[j + 1] + c_ * q[j + 2]
            den = q[2] ** 2 - q[3] * q[1] + SMALL
            db = (-p[1] * q[2] + p[0] * q[3]) / den
            dc = (-p[0] * q[2] + p[1] * q[1]) / den
            b_ += db
            c_ += dc
            if abs(db / (b_ + SMALL)) <= tol and abs(dc / (c_ + SMALL)) <= tol:
                break
        (r1, _), (r2, _) = _quadfun(b_, c_)
        re[nc - 1] = r1
        re[nc - 2] = r2
        nc -= 2
        coeffs = np.concatenate([p[2:nc + 2], np.zeros(max(0, nc - len(p[2:])))])[:nc]

    if nc == 3:
        b_ = -coeffs[1] / coeffs[2]
        c_ = -coeffs[0] / coeffs[2]
        (r1, _), (r2, _) = _quadfun(b_, c_)
        re[nc - 1] = r1
        re[nc - 2] = r2
    elif nc == 2:
        re[nc - 1] = -coeffs[0] / coeffs[1]

    lambdas = np.sort(re[1:np1])[::-1]

    steps = pd.DataFrame({
        "Eigenvalue": [f"lambda_{i + 1}" for i in range(n)],
        "Value": lambdas[:n],
        "|A - lambda.I|": [determinant(A - l * np.eye(n)) for l in lambdas[:n]],
    })
    res = Result(x=lambdas[:n], f=float(np.max(np.abs(lambdas[:n]))),
                 iterations=guard, steps=steps, method="Eigenvalues (Bairstow)",
                 eigenvalues=lambdas[:n])
    if show_steps:
        res.display()
    return lambdas[:n]
