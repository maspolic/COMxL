"""
Test suite for comxl.

Each solver is checked against a problem with a known analytical answer, so a
regression shows up as a wrong number rather than merely a crash.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from comxl import (calculus, constrained, directsearch, gradientsearch,
                   linprog, matrixops)


# ---------------------------------------------------------------------------
# Test objectives with known optima
# ---------------------------------------------------------------------------
def quad(x):
    """Minimum at x = 2, f = -1."""
    return x ** 2 - 4 * x + 3


def quartic(x):
    """Local minimum at x = 2.25, f = -6.54296875."""
    return x ** 4 - 3 * x ** 3 + 2


def rosenbrock(x):
    """Minimum at (1, 1), f = 0."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def sphere(x):
    """Minimum at (1, 2, 3), f = 0."""
    return float(np.sum((np.asarray(x) - np.array([1.0, 2.0, 3.0])) ** 2))


# ---------------------------------------------------------------------------
# One-dimensional search
# ---------------------------------------------------------------------------
ONE_D = [
    ("goldensection", dict(tol=1e-8, max_iter=300)),
    ("intervalhalving", dict(tol=1e-8, max_iter=500)),
    ("dichotomous", dict(tol=1e-6, delta=1e-8, max_iter=500)),
    ("divide_conquer", dict(n=12, max_points=2000)),
]


@pytest.mark.parametrize("name,kwargs", ONE_D)
def test_bracketing_on_quadratic(name, kwargs):
    r = getattr(gradientsearch, name)(quad, 0, 5, show_steps=False, **kwargs)
    assert r.x == pytest.approx(2.0, abs=1e-3)
    assert r.f == pytest.approx(-1.0, abs=1e-5)


@pytest.mark.parametrize("name,kwargs", ONE_D)
def test_bracketing_on_quartic(name, kwargs):
    r = getattr(gradientsearch, name)(quartic, 0, 4, show_steps=False, **kwargs)
    assert r.x == pytest.approx(2.25, abs=1e-3)


def test_fibonacci():
    r = gradientsearch.fibonacci(quad, 0, 5, n=30, show_steps=False)
    assert r.x == pytest.approx(2.0, abs=1e-4)


def test_fibonacci_numbers():
    assert [gradientsearch.fibo(n) for n in range(1, 8)] == \
        [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]


def test_fibonacci_rejects_small_n():
    with pytest.raises(ValueError):
        gradientsearch.fibonacci(quad, 0, 5, n=2, show_steps=False)


def test_exhaustive():
    r = gradientsearch.exhaustive(quad, 0, 5, n=1000, show_steps=False)
    assert r.x == pytest.approx(2.0, abs=1e-2)


def test_secant():
    r = gradientsearch.secant(quad, 0, 5, tol=1e-8, show_steps=False)
    assert r.x == pytest.approx(2.0, abs=1e-5)
    assert r.converged


def test_newtonraphson():
    r = gradientsearch.newtonraphson(quartic, 3.0, tol=1e-10, max_iter=50,
                                     show_steps=False)
    assert r.x == pytest.approx(2.25, abs=1e-6)


def test_newtonraphson_flags_a_maximum():
    # -(x-1)^2 has a maximum at x = 1; Newton finds it and should say so
    r = gradientsearch.newtonraphson(lambda x: -(x - 1) ** 2, 3.0,
                                     tol=1e-8, show_steps=False)
    assert "MAXIMUM" in r.message


def test_steps_are_recorded_even_when_hidden():
    r = gradientsearch.goldensection(quad, 0, 5, show_steps=False)
    assert not r.steps.empty
    assert list(r.steps.columns)[:2] == ["Iteration", "X_left"]


def test_string_objective_matches_callable():
    a = gradientsearch.goldensection(quad, 0, 5, tol=1e-8, show_steps=False)
    b = gradientsearch.goldensection("X_1^2 - 4*X_1 + 3", 0, 5, tol=1e-8,
                                     show_steps=False)
    assert a.x == pytest.approx(b.x)


# ---------------------------------------------------------------------------
# Multivariable direct search
# ---------------------------------------------------------------------------
def test_neldermead_rosenbrock():
    r = directsearch.neldermead(rosenbrock, [-1.2, 1.0], tol=1e-12,
                                max_iter=2000, show_steps=False)
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-4)
    assert r.f == pytest.approx(0.0, abs=1e-8)


def test_powell_rosenbrock():
    r = directsearch.powell(rosenbrock, [-1.2, 1.0], tol=1e-10,
                            show_steps=False)
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-4)


def test_powell_quadratic_line_search():
    r = directsearch.powell(rosenbrock, [-1.2, 1.0], tol=1e-10,
                            line_search="quadratic", show_steps=False)
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-4)


def test_neldermead_three_variables():
    r = directsearch.neldermead(sphere, [0, 0, 0], tol=1e-12, max_iter=3000,
                                show_steps=False)
    assert np.allclose(r.x, [1, 2, 3], atol=1e-4)


def test_scalar_arity_callable():
    r = directsearch.neldermead(lambda x1, x2: (x1 - 3) ** 2 + (x2 + 1) ** 2,
                                [0, 0], tol=1e-12, show_steps=False)
    assert np.allclose(r.x, [3, -1], atol=1e-4)


def test_string_objective_multivariable():
    r = directsearch.neldermead("(X_1-3)^2 + (X_2+1)^2", [0, 0], tol=1e-12,
                                show_steps=False)
    assert np.allclose(r.x, [3, -1], atol=1e-4)


def test_maximize():
    r = directsearch.neldermead(lambda x: -(x[0] - 1) ** 2 - (x[1] - 2) ** 2 + 10,
                                [0, 0], maximize=True, tol=1e-12,
                                show_steps=False)
    assert np.allclose(r.x, [1, 2], atol=1e-4)
    assert r.f == pytest.approx(10.0, abs=1e-6)


def test_bounds_per_variable_layout():
    r = directsearch.neldermead(rosenbrock, [0.1, 0.1],
                                bounds=[(0, 0.5), (0, 0.5)], tol=1e-12,
                                max_iter=3000, show_steps=False)
    # constrained optimum sits on the x1 = 0.5 face at x2 = 0.25
    assert np.allclose(r.x, [0.5, 0.25], atol=1e-3)


def test_bounds_two_row_layout():
    r = directsearch.neldermead(rosenbrock, [0.1, 0.1],
                                bounds=((0.0, 0.0), (0.5, 0.5)), tol=1e-12,
                                max_iter=3000, show_steps=False)
    assert np.allclose(r.x, [0.5, 0.25], atol=1e-3)


def test_bounds_are_respected():
    r = directsearch.neldermead(rosenbrock, [0.1, 0.1],
                                bounds=[(0, 2), (0, 3)], tol=1e-12,
                                max_iter=3000, show_steps=False)
    assert np.all(r.x >= -1e-9) and r.x[0] <= 2 + 1e-9 and r.x[1] <= 3 + 1e-9


def test_reversed_bounds_rejected():
    # 3 variables removes the (2, n) / (n, 2) ambiguity, so a reversed
    # lower/upper pair cannot be reinterpreted and must be rejected
    with pytest.raises(ValueError):
        directsearch.neldermead(sphere, [0, 0, 0],
                                bounds=[(5, 0), (5, 0), (5, 0)],
                                show_steps=False)


def test_wrong_shaped_bounds_rejected():
    with pytest.raises(ValueError):
        directsearch.neldermead(rosenbrock, [0, 0], bounds=[(0, 1, 2)],
                                show_steps=False)


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------
def test_gradient():
    f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2
    g = calculus.gradient(f, [1.0, 2.0], show_steps=False).extras["gradient"]
    assert np.allclose(g, [8.0, 11.0], atol=1e-5)


def test_hessian_and_classification():
    f = lambda x: x[0] ** 2 + x[1] ** 2
    res = calculus.hessian(f, [1.0, 1.0], show_steps=False)
    assert np.allclose(res.extras["hessian"], [[2, 0], [0, 2]], atol=1e-3)
    assert "MINIMUM" in res.extras["nature"]


def test_hessian_detects_saddle():
    f = lambda x: x[0] ** 2 - x[1] ** 2
    res = calculus.hessian(f, [0.0, 0.0], show_steps=False)
    assert "SADDLE" in res.extras["nature"]


def test_jacobian():
    fs = [lambda x: x[0] ** 2 + x[1], lambda x: x[0] - x[1] ** 3]
    J = calculus.jacobian(fs, [2.0, 3.0], show_steps=False).extras["jacobian"]
    assert np.allclose(J, [[4, 1], [1, -27]], atol=1e-3)


def test_gradient_vector_matches_gradient():
    f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2
    lean = calculus.gradient_vector(f, [1.0, 2.0])
    full = calculus.gradient(f, [1.0, 2.0], show_steps=False).extras["gradient"]
    assert np.allclose(lean, full)
    assert np.allclose(lean, [8.0, 11.0], atol=1e-5)


def test_hessian_matrix_matches_hessian():
    f = lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2
    lean = calculus.hessian_matrix(f, [1.0, 2.0])
    full = calculus.hessian(f, [1.0, 2.0], show_steps=False).extras["hessian"]
    assert np.allclose(lean, full, atol=1e-6)
    assert np.allclose(lean, [[2, 3], [3, 4]], atol=1e-3)


def test_lean_derivatives_return_plain_arrays():
    # the point of these is to skip Result/DataFrame construction entirely
    out = calculus.gradient_vector(lambda x: x[0] ** 2, [1.0])
    assert isinstance(out, np.ndarray)
    assert not hasattr(out, "steps")


def test_scalar_derivatives():
    assert calculus.first_derivative(lambda x: x ** 3, 2) == pytest.approx(12, abs=1e-5)
    assert calculus.second_derivative(lambda x: x ** 3, 2) == pytest.approx(12, abs=1e-4)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------
A_SPD = np.array([[4.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 4.0]])


def test_inverse():
    assert np.allclose(matrixops.inverse(A_SPD, show_steps=False) @ A_SPD,
                       np.eye(3), atol=1e-10)


def test_determinant():
    assert matrixops.determinant(A_SPD) == pytest.approx(np.linalg.det(A_SPD))


def test_gauss_pivot():
    b = np.array([11.0, -16.0, 17.0])
    x = matrixops.gauss_pivot(A_SPD, b, show_steps=False)
    assert np.allclose(A_SPD @ x, b, atol=1e-10)


def test_gauss_pivot_rejects_singular():
    S = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError):
        matrixops.gauss_pivot(S, [1.0, 2.0], show_steps=False)


def test_multiply_and_transpose():
    assert np.allclose(matrixops.multiply(A_SPD, A_SPD, show_steps=False),
                       A_SPD @ A_SPD)
    assert np.allclose(matrixops.transpose(A_SPD, show_steps=False), A_SPD.T)


def test_add_subtract_shape_check():
    with pytest.raises(ValueError):
        matrixops.add(A_SPD, np.eye(2), show_steps=False)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_eigenvalues_match_numpy(n):
    rng = np.random.default_rng(n)
    M = rng.standard_normal((n, n))
    M = M + M.T
    got = np.sort(matrixops.eigenvalues(M, show_steps=False))
    want = np.sort(np.linalg.eigvals(M).real)
    assert np.allclose(got, want, atol=1e-4)


# ---------------------------------------------------------------------------
# Linear programming
# ---------------------------------------------------------------------------
def test_simplex_max():
    r = linprog.simplex([[2, 1, 10], [1, 3, 15], [4, 3, 0]], mode="max",
                        show_steps=False)
    assert np.allclose(r.x, [3.0, 4.0], atol=1e-8)
    assert r.f == pytest.approx(24.0)


def test_simplex_max_classic():
    r = linprog.simplex([[1, 0, 4], [0, 2, 12], [3, 2, 18], [3, 5, 0]],
                        mode="max", show_steps=False)
    assert np.allclose(r.x, [2.0, 6.0], atol=1e-8)
    assert r.f == pytest.approx(36.0)


def test_simplex_min_duality():
    r = linprog.simplex([[1, 2, 4], [3, 1, 6], [6, 3, 0]], mode="min",
                        show_steps=False)
    # objective value must equal c . x for the recovered primal variables
    assert r.f == pytest.approx(6 * r.x[0] + 3 * r.x[1], abs=1e-8)
    assert r.f == pytest.approx(13.2, abs=1e-6)


def test_simplex_records_tableaux():
    r = linprog.simplex([[2, 1, 10], [1, 3, 15], [4, 3, 0]], mode="max",
                        show_steps=False)
    assert len(r.extras["tableaux"]) == r.iterations + 1


def test_simplex_rejects_bad_mode():
    with pytest.raises(ValueError):
        linprog.simplex([[1, 1, 1], [1, 1, 0]], mode="minimise",
                        show_steps=False)


def test_feasibility():
    a, b, s = [[2, 1], [1, 3]], [10, 15], ["<=", "<="]
    assert linprog.feasible(a, b, s, (3, 4))
    assert not linprog.feasible(a, b, s, (9, 9))


def test_constraint_line_vertical():
    line = linprog.constraint_line(0, 6, a=1.0, b=0.0, c=3.0, sense="<=")
    assert line["vertical"]
    assert line["x"] == (3.0, 3.0)


# ---------------------------------------------------------------------------
# Feasible-region corner points
# ---------------------------------------------------------------------------
def _pairs(df):
    return sorted((round(r.x1, 6), round(r.x2, 6)) for r in df.itertuples())


def test_line_intersection():
    p = linprog.line_intersection([2, 1], 10, [1, 3], 15)
    assert np.allclose(p, [3.0, 4.0])


def test_line_intersection_parallel_returns_none():
    assert linprog.line_intersection([1, 1], 2, [2, 2], 8) is None


def test_vertices_of_standard_region():
    v = linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                  show_steps=False)
    assert _pairs(v) == [(0.0, 0.0), (0.0, 5.0), (3.0, 4.0), (5.0, 0.0)]


def test_vertices_are_ordered_around_the_boundary():
    # consecutive rows must be adjacent corners, so the traced path is convex
    v = linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                  show_steps=False)
    pts = v[["x1", "x2"]].values
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    assert np.all(np.diff(ang) > 0)


def test_vertices_report_which_lines_met():
    v = linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                  show_steps=False)
    interior = v[(v.x1 > 1) & (v.x2 > 1)]
    assert interior.iloc[0]["lines"] == "C1 & C2"


def test_vertices_with_mixed_senses():
    v = linprog.feasible_vertices([[1, 1], [2, 1], [1, 3]], [2, 10, 15],
                                  [">=", "<=", "<="], show_steps=False)
    assert _pairs(v) == [(0.0, 2.0), (0.0, 5.0), (2.0, 0.0), (3.0, 4.0),
                         (5.0, 0.0)]


def test_vertices_without_nonnegativity():
    v = linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                  nonnegative=False, show_steps=False)
    # only the two constraints meet, so a single corner remains
    assert _pairs(v) == [(3.0, 4.0)]


def test_vertices_of_infeasible_region_is_empty():
    v = linprog.feasible_vertices([[1, 1], [1, 1]], [2, 8], ["<=", ">="],
                                  show_steps=False)
    assert v.empty


def test_vertices_with_equality_gives_segment_ends():
    v = linprog.feasible_vertices([[1, 1], [2, 1]], [4, 10], ["=", "<="],
                                  show_steps=False)
    assert _pairs(v) == [(0.0, 4.0), (4.0, 0.0)]


def test_vertices_match_simplex_optimum():
    # the simplex optimum must be one of the corner points
    problem = [[2, 1, 10], [1, 3, 15], [4, 3, 0]]
    lp = linprog.simplex(problem, mode="max", show_steps=False)
    v = linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                  show_steps=False)
    assert any(np.allclose([r.x1, r.x2], lp.x, atol=1e-8)
               for r in v.itertuples())


def test_region_polygon_clips_unbounded_region():
    poly = linprog.region_polygon([[1, 1], [1, -1]], [3, 1], [">=", "<="],
                                  xlim=(0, 8), ylim=(0, 8))
    assert len(poly) >= 3                      # closed by the viewing box
    assert poly[:, 0].max() <= 8 + 1e-9
    assert poly[:, 1].max() <= 8 + 1e-9


def test_region_polygon_empty_when_infeasible():
    poly = linprog.region_polygon([[1, 1], [1, 1]], [2, 8], ["<=", ">="],
                                  xlim=(0, 8), ylim=(0, 8))
    assert len(poly) == 0


def test_every_vertex_is_feasible():
    a, b, s = [[1, 1], [2, 1], [1, 3]], [2, 10, 15], [">=", "<=", "<="]
    v = linprog.feasible_vertices(a, b, s, show_steps=False)
    for r in v.itertuples():
        assert linprog.feasible(a, b, s, (r.x1, r.x2), tol=1e-7)


# ---------------------------------------------------------------------------
# Constrained optimisation
# ---------------------------------------------------------------------------
def test_quasinewton_circle_line():
    r = constrained.quasinewton(
        [lambda x: x[0] ** 2 + x[1] ** 2 - 4, lambda x: x[0] - x[1]],
        [1.0, 0.5], show_steps=False)
    assert np.allclose(r.x, [np.sqrt(2), np.sqrt(2)], atol=1e-6)
    assert r.converged


def test_quasinewton_mismatched_sizes():
    with pytest.raises(ValueError):
        constrained.quasinewton([lambda x: x[0] - 1], [1.0, 2.0],
                                show_steps=False)


def test_sqp_equality_constrained():
    # min x1^2 + x2^2 subject to x1 + x2 = 4  ->  (2, 2), lambda = -4
    L = constrained.lagrangian(lambda x: x[0] ** 2 + x[1] ** 2,
                               [lambda x: x[0] + x[1] - 4], n_vars=2)
    r = constrained.sqp(L, [0.5, 0.5, 0.1],
                        objective=lambda x: x[0] ** 2 + x[1] ** 2,
                        n_vars=2, show_steps=False)
    assert np.allclose(r.extras["variables"], [2.0, 2.0], atol=1e-6)
    assert r.extras["multipliers"][0] == pytest.approx(-4.0, abs=1e-5)
    assert r.f == pytest.approx(8.0, abs=1e-6)


def test_sqp_on_circle():
    # nearest point on the unit circle to (1, 2) is (1, 2)/sqrt(5)
    f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2
    L = constrained.lagrangian(f, [lambda x: x[0] ** 2 + x[1] ** 2 - 1],
                               n_vars=2)
    r = constrained.sqp(L, [0.5, 0.5, 0.5], objective=f, n_vars=2,
                        show_steps=False)
    expected = np.array([1.0, 2.0]) / np.sqrt(5.0)
    assert np.allclose(r.extras["variables"], expected, atol=1e-5)


def test_kkt_active_constraint():
    # min (x1-2)^2 + (x2-1)^2 s.t. x1 + x2 <= 2, x1 >= 0, x2 >= 0
    r = constrained.kkt(
        lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2, 2,
        inequality=[lambda x: x[0] + x[1] - 2,
                    lambda x: -x[0],
                    lambda x: -x[1]],
        x0=[0.5, 0.5], show_steps=False)
    assert np.allclose(r.x, [1.5, 0.5], atol=1e-5)
    assert r.extras["active_set"] == "{g1}"
    assert len(r.steps) == 8          # 2^3 active-set cases enumerated


def test_kkt_inactive_constraint():
    # unconstrained optimum is already feasible, so no constraint binds
    r = constrained.kkt(lambda x: (x[0] - 0.2) ** 2 + (x[1] - 0.3) ** 2, 2,
                        inequality=[lambda x: x[0] + x[1] - 2],
                        x0=[0.5, 0.5], show_steps=False)
    assert np.allclose(r.x, [0.2, 0.3], atol=1e-5)
    assert "interior" in r.extras["active_set"]


def test_kkt_maximize_with_equality():
    r = constrained.kkt(lambda x: x[0] * x[1], 2,
                        equality=[lambda x: x[0] + x[1] - 10],
                        x0=[1.0, 2.0], maximize=True, show_steps=False)
    assert np.allclose(r.x, [5.0, 5.0], atol=1e-5)
    assert r.f == pytest.approx(25.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpful errors for the wrong kind of objective
# ---------------------------------------------------------------------------
def test_vector_objective_in_1d_method_is_explained():
    # the exact mistake a user hit: a 2-variable lambda given to goldensection
    with pytest.raises(TypeError) as exc:
        gradientsearch.goldensection(lambda x: x[0]**2 + x[1]**2, 0, 5,
                                     show_steps=False)
    msg = str(exc.value)
    assert "single variable" in msg.lower()
    assert "directsearch" in msg          # points at the right solver


def test_two_argument_objective_rejected_up_front():
    with pytest.raises(TypeError) as exc:
        gradientsearch.goldensection(lambda x1, x2: x1**2 + x2**2, 0, 5,
                                     show_steps=False)
    assert "2 required arguments" in str(exc.value)


def test_two_variable_string_in_1d_method_is_explained():
    with pytest.raises(NameError) as exc:
        gradientsearch.goldensection("X_1^2 + X_2^2", 0, 5, show_steps=False)
    assert "X_2" in str(exc.value)
    assert "multivariable" in str(exc.value).lower()


@pytest.mark.parametrize("name,args", [
    ("goldensection", (0, 5)),
    ("intervalhalving", (0, 5)),
    ("dichotomous", (0, 5)),
    ("exhaustive", (0, 5)),
    ("newtonraphson", (1.0,)),
])
def test_all_1d_methods_explain_vector_objectives(name, args):
    with pytest.raises(TypeError) as exc:
        getattr(gradientsearch, name)(lambda x: x[0]**2 + x[1]**2, *args,
                                      show_steps=False)
    assert "directsearch" in str(exc.value)


def test_index_error_keeps_its_type():
    # a genuine 1D objective that raises IndexError must not be misdiagnosed
    table = [1.0, 2.0]
    with pytest.raises(IndexError):
        gradientsearch.goldensection(lambda x: table[int(x)], 0, 5,
                                     show_steps=False)


def test_unrelated_errors_propagate_unchanged():
    with pytest.raises(ZeroDivisionError):
        gradientsearch.goldensection(lambda x: 1 / 0, 0, 5, show_steps=False)


def test_scalar_objective_in_multivariable_solver_is_explained():
    # `lambda x: x**2` squares the whole vector and returns a vector
    with pytest.raises(TypeError) as exc:
        directsearch.neldermead(lambda x: x**2, [1.0, 2.0], show_steps=False)
    msg = str(exc.value)
    assert "single number" in msg
    assert "x[0]**2" in msg               # shows the correct form


def test_non_callable_objective_rejected():
    with pytest.raises(TypeError):
        gradientsearch.goldensection(42, 0, 5, show_steps=False)


# ---------------------------------------------------------------------------
# Nelder-Mead convergence
# ---------------------------------------------------------------------------
def test_neldermead_one_variable_reaches_the_true_minimum():
    # the VBA's function-value-only test stops at 2.9 with vertices straddling
    # the minimum; the simplex-size test must carry it to 3.0
    r = directsearch.neldermead(lambda x: (x[0] - 3) ** 2, [0.0], tol=1e-12,
                                max_iter=500, show_steps=False)
    assert r.x[0] == pytest.approx(3.0, abs=1e-6)
    assert r.converged


def test_xtol_zero_reproduces_the_vba_behaviour():
    r = directsearch.neldermead(lambda x: (x[0] - 3) ** 2, [0.0], tol=1e-12,
                                xtol=0, max_iter=500, show_steps=False)
    assert r.x[0] == pytest.approx(2.9, abs=1e-6)      # the original defect


def test_neldermead_records_simplex_size():
    r = directsearch.neldermead(sphere, [0, 0, 0], show_steps=False)
    assert "Size" in r.steps.columns
    assert r.steps["Size"].iloc[-1] < r.steps["Size"].iloc[0]


def test_result_repr_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    r = gradientsearch.goldensection(quad, 0, 5, show_steps=False)
    assert "Golden section" in repr(r)
    ax = r.plot_convergence()
    assert ax is not None


# ---------------------------------------------------------------------------
# Plot saving and labelling
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="module")
def _agg_backend():
    import matplotlib
    matplotlib.use("Agg")


def test_coord_label_trims_trailing_zeros():
    from comxl.plotting import _coord_label
    assert _coord_label(3.0, 4.0) == "(3, 4)"
    assert _coord_label(0.5, 0.25) == "(0.5, 0.25)"
    assert _coord_label(-0.0, 5.0) == "(0, 5)"


def test_coord_label_respects_precision():
    from comxl.plotting import _coord_label
    assert _coord_label(1 / 3, 2 / 3, precision=4) == "(0.3333, 0.6667)"


def test_save_figure_returns_path(tmp_path):
    import matplotlib.pyplot as plt
    from comxl.plotting import save_figure
    fig, _ = plt.subplots()
    path = save_figure(fig, "demo.png", directory=tmp_path, verbose=False)
    assert (tmp_path / "demo.png").exists()
    assert path.endswith("demo.png")
    plt.close(fig)


def test_save_figure_accepts_an_axes(tmp_path):
    import matplotlib.pyplot as plt
    from comxl.plotting import save_figure
    _, ax = plt.subplots()
    save_figure(ax, "from_axes.png", directory=tmp_path, verbose=False)
    assert (tmp_path / "from_axes.png").exists()


def test_save_figure_adds_default_extension(tmp_path):
    import matplotlib.pyplot as plt
    from comxl.plotting import save_figure
    fig, _ = plt.subplots()
    save_figure(fig, "noext", directory=tmp_path, verbose=False)
    assert (tmp_path / "noext.png").exists()


def test_save_figure_creates_missing_directory(tmp_path):
    import matplotlib.pyplot as plt
    from comxl.plotting import save_figure
    target = tmp_path / "nested" / "deeper"
    fig, _ = plt.subplots()
    save_figure(fig, "x.png", directory=target, verbose=False)
    assert (target / "x.png").exists()


def test_plot_constraints_saves_with_auto_name(tmp_path):
    from comxl import plotting
    plotting.plot_constraints_2d([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                 save=True, directory=tmp_path)
    assert (tmp_path / "constraints.png").exists()


def test_plot_constraints_saves_with_given_name(tmp_path):
    from comxl import plotting
    plotting.plot_constraints_2d([[2, 1], [1, 3]], [10, 15], ["<=", "<="],
                                 save="region.pdf", directory=tmp_path)
    assert (tmp_path / "region.pdf").exists()


def test_plot_function_and_contours_save(tmp_path):
    from comxl import plotting
    plotting.plot_function(quad, 0, 5, save=True, directory=tmp_path)
    plotting.plot_contours(lambda x: x[0]**2 + x[1]**2, (-2, 2), (-2, 2),
                           n=20, save=True, directory=tmp_path)
    assert (tmp_path / "function.png").exists()
    assert (tmp_path / "contours.png").exists()


def test_plots_do_not_save_by_default(tmp_path):
    from comxl import plotting
    plotting.plot_constraints_2d([[2, 1]], [10], ["<="], directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("hatch", ["++", "xx", "//", None])
def test_hatch_patterns_all_render(hatch):
    from comxl import plotting
    ax = plotting.plot_constraints_2d([[2, 1], [1, 3]], [10, 15],
                                      ["<=", "<="], hatch=hatch)
    assert ax is not None
