# COMXL

**Classical optimisation methods with the working shown.**

This repository keeps the algorithms and the
step-by-step output, but the objective is now an ordinary Python function.

**▶ [Run it in your browser](https://maspolic-amo-yeboah.github.io/COMXL/)** —
no install; everything executes locally via Pyodide.

![The linear-programming panel of the COMXL web app, showing the simplex
result, the corner points of the feasible region and the shaded region
plot](docs/assets/screenshot.png)

```python
from comxl import gradientsearch

r = gradientsearch.goldensection(lambda x: x**2 - 4*x + 3, 0, 5)
```

```
 Iteration  X_left     X_1     X_2  X_right    f(X_1)    f(X_2)      Error
         0       0 1.90983 3.09017        5  -0.99187   0.18846    1.18033
         1       0 1.18034 1.90983  3.09017 -0.328159  -0.99187   0.663711
         2 1.18034 1.90983 2.36068  3.09017  -0.99187 -0.869913   0.121957
         ...

Golden section search result (converged in 4 iterations)
  x = 1.9959325
  f(x) = -0.99998346
```

---

## Contents

| Path | What it holds |
|---|---|
| `comxl/` | The Python package |
| `tests/` | 104 tests, each against a problem with a known analytical answer |
| `docs/` | The GitHub Pages site — the browser version |
| `notebooks/` | A 70-cell walkthrough of every module ||
| `tools/` | Build the Pages wheel; end-to-end test the site |
| `PUBLISHING.md` | One-time checklist for pushing and enabling Pages |

## Installation

```bash
pip install .                 # or: pip install -e ".[dev,plots]"
```

Python 3.9+, with NumPy, pandas and Matplotlib. seaborn is optional — it only
supplies styling and the categorical palette, and plotting falls back to
Matplotlib without it, which is what keeps the wheel installable in a browser.

No optimisation library is used anywhere: Gauss elimination, Bairstow
root-finding, the simplex pivots and every search method are written out in
full, since providing them is the point.

## The two things that changed from the VBA

**Objectives are functions, not cells.** Any of these work, and the adapter
works out which you meant:

```python
f = lambda x: x[0]**2 + x[1]**2      # one array argument
f = lambda x1, x2: x1**2 + x2**2     # separate scalars
f = "X_1^2 + X_2^2"                  # worksheet-style string, for porting
```

**Steps are printed by default.** Every solver takes `show_steps=True`, which
prints the same iteration table the VBA wrote to the worksheet — same columns,
same values. Pass `show_steps=False` for the answer alone. Either way the table
is kept:

```python
r = gradientsearch.goldensection(f, 0, 5, show_steps=False)
r.x, r.f, r.iterations, r.converged
r.steps            # pandas DataFrame of the full history
r.plot_convergence()
```

## Modules

| Module | Functions |
|---|---|---|
| `gradientsearch` | `goldensection`, `intervalhalving`, `dichotomous`, `fibonacci`, `exhaustive`, `secant`, `newtonraphson`, `divide_conquer` |
| `directsearch` | `neldermead`, `powell` |
| `constrained` | `sqp`, `quasinewton`, `kkt`, `lagrangian` |
| `linprog` | `simplex`, `feasible`, `feasible_vertices`, `region_polygon`, `line_intersection`, `constraint_line` |
| `calculus` | `gradient`, `hessian`, `jacobian`, `first_derivative`, `second_derivative`, `gradient_vector`, `hessian_matrix` |
| `matrixops` | `inverse`, `transpose`, `multiply`, `add`, `subtract`, `gauss_pivot`, `eigenvalues`, `determinant`, `row_echelon` |
| `plotting` | `plot_function`, `plot_contours`, `plot_constraints_2d`, `plot_path`, `save_figure`, `axis_scale` |

## Examples

Multivariable, with box constraints:

```python
from comxl import directsearch

rosen = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
r = directsearch.neldermead(rosen, [-1.2, 1.0], show_steps=False)
r.x                                    # array([1., 1.])

directsearch.powell(rosen, [-1.2, 1.0], line_search='golden')
directsearch.neldermead(rosen, [0.1, 0.1], bounds=[(0, 0.5), (0, 0.5)])
```

Linear programming — constraint rows then the objective row, RHS in the last
column:

```python
from comxl import linprog

#   2x1 +  x2 <= 10
#    x1 + 3x2 <= 15
#   max z = 4x1 + 3x2
r = linprog.simplex([[2, 1, 10],
                     [1, 3, 15],
                     [4, 3,  0]], mode='max')
r.x, r.f                               # [3., 4.], 24.0
```

The corner points of the feasible region, with the boundaries that meet at
each — by the fundamental theorem of linear programming these are the only
places an optimum can sit, so the simplex answer must be one of them:

```python
linprog.feasible_vertices([[2, 1], [1, 3]], [10, 15], ['<=', '<='])
```

```
 x1  x2         lines
  0   0 x1>=0 & x2>=0
  5   0    C1 & x2>=0
  3   4       C1 & C2
  0   5    C2 & x1>=0

4 vertices of the feasible region.
```

Plotting fills that polygon exactly — crisp edges rather than the ragged
outline a sampled grid gives — with dotted constraint boundaries as in the
original workbook chart, and every corner marked and labelled `(x, y)`:

```python
plotting.plot_constraints_2d([[2, 1], [1, 3]], [10, 15], ['<=', '<='],
                             xlim=(0, 8), ylim=(0, 8), point=(3, 4))
```

The region is cross-hatched with a grid. The `hatch` argument takes any
Matplotlib pattern: `'++'` for a grid (default), `'xx'` for a diagonal mesh,
`'//'` for diagonal lines, or `None` for a plain fill. Corner labels are boxed
so they stay legible over the hatching, and nudged inward for corners on an
axis so nothing is written off the edge.

Degenerate cases are handled rather than crashing: an unbounded region is
clipped to the viewing box so it can still be shaded, an equality constraint
collapses the region to a line segment which is drawn as one, and an infeasible
system returns an empty frame and is labelled on the plot.

Constrained, via the KKT conditions (inequalities written as `g(x) <= 0`):

```python
from comxl import constrained

r = constrained.kkt(
    lambda x: (x[0] - 2)**2 + (x[1] - 1)**2, n_vars=2,
    inequality=[lambda x: x[0] + x[1] - 2,
                lambda x: -x[0],
                lambda x: -x[1]],
    x0=[0.5, 0.5])
r.x                                    # [1.5, 0.5]
```

## Showing and saving figures

Every plotting function takes the same four output arguments:

```python
plotting.plot_constraints_2d(a, b, senses,
                             save=True,             # or 'region.pdf'
                             directory='figures',   # default: current dir
                             dpi=300,
                             show=True)             # calls plt.show()
```

`save=True` writes under an automatic name (`constraints.png`, `function.png`,
`contours.png`); a string is used as the filename, its extension choosing the
format. `directory` defaults to the current working directory and is created if
missing. Nothing is written unless you ask.

`save_figure` is available directly for figures you assemble yourself:

```python
path = plotting.save_figure(fig, 'my_chart.png', directory='figures', dpi=300)
```

It accepts an Axes as well as a Figure, appends `.png` when the name has no
extension, and returns the absolute path written.

## The browser version

`docs/` is a static site that runs the package itself — it does not
reimplement anything. On load it boots [Pyodide](https://pyodide.org), installs
the wheel from `docs/wheels/`, and calls `comxl` through a thin bridge in
`docs/js/runner.py`. Nothing is uploaded; nothing runs on a server.

Panels cover one-dimensional search, multivariable search, linear programming
with the feasible-region plot, the KKT solver, matrix operations and
derivatives. Each shows the same iteration table as the library, and renders
Matplotlib figures as inline PNGs.

### Publishing it

1. **Settings → Pages → Source: GitHub Actions.** The `deploy pages` workflow
   publishes `docs/` on every push to `main` that touches it.
2. The site lands at `https://<user>.github.io/COMXL/`. Asset paths are
   resolved against the page URL, so the project subpath works without
   configuration.

### After changing the package

The site installs a committed wheel, so rebuild it or the published page keeps
running the old code:

```bash
python tools/build_web_wheel.py
git add docs/wheels && git commit -m "Rebuild Pages wheel"
```

The `tests` workflow fails if `docs/wheels/` holds a different version from the
one it just built, so a stale wheel cannot reach the site unnoticed.

### Testing the site

```bash
pip install playwright && python -m playwright install chromium
python tools/test_site.py
```

This serves `docs/` locally, loads it in headless Chromium, waits for the
runtime, then drives every panel and asserts the numbers — golden section
reaching 2, the simplex reaching 24, the Hessian reporting a saddle, and so on.
It also fails on any uncaught console error.

## Tests

```bash
pytest tests/ -q                          # 104 tests
pytest --doctest-modules comxl/ -q        # docstring examples
```

Every solver is checked against a problem with a known analytical answer, and
`eigenvalues` is checked against `numpy.linalg.eigvals` for n = 2..5.

## Errors when the objective has the wrong shape

The one-dimensional methods in `gradientsearch` take `f(x) -> float` and search
between two scalar bounds. Passing a multivariable objective used to fail deep
inside the user's own lambda with a bare `TypeError: 'float' object is not
subscriptable`. It now says what is wrong and what to use instead:

```python
gradientsearch.goldensection(lambda x: x[0]**2 + x[1]**2, 0, 5)
# TypeError: Could not evaluate the objective at x = 1.909835.
#
# The one-dimensional methods in comxl.gradientsearch take a function of a
# SINGLE variable, f(x) -> float, and search between two scalar bounds.
# ...
# For a multivariable objective use one of:
#     comxl.directsearch.neldermead(f, x0)      # simplex, derivative-free
#     comxl.directsearch.powell(f, x0)          # conjugate directions
```

A callable needing two or more arguments is rejected before any evaluation, and
a string expression naming `X_2` says so by name. The reverse mistake is caught
too: `lambda x: x**2` given to a multivariable solver squares the whole vector
and returns a vector, which is now reported as such.

To minimise a multivariable function along a line with a 1-D method, pass a
function of the step length:

```python
gradientsearch.goldensection(lambda t: f(x0 + t*d), 0, 1)
```

## Notes and limits

**KKT enumeration is exponential.** `kkt` tries all 2^m active sets, exactly as
the VBA form's case branches did. That is fine for the teaching problems the
workbook targeted, but past roughly ten inequality constraints you want an
active-set or interior-point method instead.

**SQP takes full Newton steps.** No line search, no BFGS update, no active-set
logic — so it converges quickly near a solution and needs a reasonable starting
guess. This matches the original.

**Eigenvalues use random sample points.** The characteristic polynomial is
recovered by interpolation through randomly chosen values of lambda, so the
VBA result varied between runs; here a `seed` argument makes it reproducible.

### Fixes to the original behaviour

Each is documented in the source where it occurs.

- `neldermead`'s constrained starting simplex used a fixed 0.1 perturbation
  regardless of box width, which produced a degenerate simplex and premature
  convergence. The perturbation now scales with the box.
- `gauss_pivot` checked pivots 1..n−1 but not the last, dividing by zero on a
  rank-deficient system — which arises routinely inside the KKT enumeration.
- `intervalhalving` uses half the sample spacing when the centre point wins,
  which the VBA did but which is easy to miss when reading the branches.
- `gradient` and `hessian` always built a `Result` with a pandas DataFrame,
  even when called from inside a solver that discards everything but the
  numbers. That construction was about 70% of `kkt`'s runtime. The solvers now
  use `gradient_vector` and `hessian_matrix`, which return plain arrays —
  identical values, 12x faster, and the difference between a usable and an
  unusable KKT panel in the browser.
- `neldermead` judged convergence purely by the spread of function values.
  Vertices placed symmetrically about the minimum have equal values, so that
  test collapses to zero while the simplex is still wide: on `(x - 3)**2` from
  `x0 = 0` the original stops at 2.9. A second test on the simplex's geometric
  size (`xtol`, default 1e-8) is now required as well. Pass `xtol=0` to
  reproduce the original behaviour.

## Provenance

Ported from the 30 VBA source files in `vba_source/` (`.bas` modules and
`.frm` user forms). The `.frx` companions hold only form binaries — icons and
control layout — and contain no logic, so they are not included. Individual
function docstrings name the VBA routine each was derived from.

## Licence

MIT — see [LICENSE](LICENSE).

Maspolic Amo Yeboah
