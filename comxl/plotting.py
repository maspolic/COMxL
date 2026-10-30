"""
Plotting helpers.

Ports of the charting behaviour of:
  * `FunPLOT` (UserForm19.frm, "Grapher - String Function")
  * `QYXPLT`  (UserForm20.frm, "Grapher - Cell Function")
  * `graph`   (Module5.bas, UserForm3.frm,
               "Linear Programming 2D Constraints Grapher")
  * `AXISCALE` (Helper_Sub.bas) -- the axis-rounding routine

Uses matplotlib with seaborn styling; no plotting is done inside the solvers
themselves, so the package works headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:                                   # seaborn is optional
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:                    # pragma: no cover - environment-specific
    sns = None
    HAVE_SEABORN = False

from .core import make_function, make_vector_function
from .linprog import (constraint_line, feasible, feasible_vertices,
                      region_polygon)

__all__ = [
    "set_style",
    "save_figure",
    "palette",
    "plot_function",
    "plot_contours",
    "plot_constraints_2d",
    "plot_path",
    "axis_scale",
    "HAVE_SEABORN",
]


def set_style(style: str = "whitegrid", context: str = "notebook",
              palette: str = "deep"):
    """
    Apply a seaborn style to all subsequent plots.

    Falls back to a matplotlib style when seaborn is not installed, so the
    package still plots in a minimal environment such as a browser runtime.
    """
    if HAVE_SEABORN:
        sns.set_theme(style=style, context=context, palette=palette)
        return
    # Set the look explicitly rather than relying on a named style being
    # present: the bundled seaborn styles are missing from some minimal
    # matplotlib builds, and the next fallback (ggplot) has a grey axes
    # background that clashes with everything else.
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#b8c2c9",
        "axes.labelcolor": "#15232e",
        "axes.grid": True,
        "grid.color": "#dde3e7",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "text.color": "#15232e",
        "xtick.color": "#4a5c6a",
        "ytick.color": "#4a5c6a",
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#dde3e7",
    })


def palette(n: int):
    """
    A list of ``n`` distinguishable colours.

    Uses seaborn's 'husl' palette when available and matplotlib's tab20
    colormap otherwise, so constraint lines stay distinguishable either way.
    """
    if HAVE_SEABORN:
        return sns.color_palette("husl", n)
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(n)]


# ----------------------------------------------------------------------------
# Saving and displaying
# ----------------------------------------------------------------------------
def save_figure(fig, filename: str, directory=None, dpi: int = 200,
                verbose: bool = True, **savefig_kwargs) -> str:
    """
    Write a figure to disk and return the path it was written to.

    Parameters
    ----------
    fig : matplotlib Figure or Axes
        An Axes is accepted for convenience; its parent figure is used.
    filename : str
        Output file name. The extension sets the format -- '.png', '.pdf',
        '.svg' and so on. A bare name gets '.png'.
    directory : str or pathlib.Path, optional
        Where to write. Defaults to the current working directory. Created if
        it does not exist.
    dpi : int
        Resolution for raster formats.
    verbose : bool
        Print the path that was written.

    Returns
    -------
    str
        Absolute path of the saved file.

    Examples
    --------
    >>> import matplotlib; matplotlib.use('Agg')
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> path = save_figure(fig, 'demo.png', directory='/tmp', verbose=False)
    >>> path.endswith('demo.png')
    True
    """
    if hasattr(fig, "figure") and not hasattr(fig, "savefig"):
        fig = fig.figure                       # an Axes was passed

    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".png")

    folder = Path.cwd() if directory is None else Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    full = (folder / path.name).resolve()

    kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    kwargs.update(savefig_kwargs)
    fig.savefig(full, **kwargs)

    if verbose:
        print(f"Figure saved to {full}")
    return str(full)


def _finalise(ax, save, filename, directory, dpi, show, default_name):
    """
    Shared tail of every plotting function: optionally save, optionally show.

    ``save`` may be True (use ``default_name``), a filename string, or False.
    """
    if save:
        name = filename or (save if isinstance(save, str) else default_name)
        save_figure(ax.figure, name, directory=directory, dpi=dpi)
    if show:
        plt.show()
    return ax


# ----------------------------------------------------------------------------
# Axis scaling  (AXISCALE in Helper_Sub.bas)
# ----------------------------------------------------------------------------
def axis_scale(dmin: float, dmax: float):
    """
    Round axis bounds outward to tidy major/minor units.

    Returns
    -------
    (bmin, bmax, major, minor)
    """
    if dmax == dmin:
        bmax, bmin = dmax * 1.01, dmin * 0.99
    elif dmax < dmin:
        bmax, bmin = dmin, dmax
    else:
        bmax, bmin = dmax, dmin

    span = bmax - bmin
    if bmax > 0:
        bmax = bmax + span * 0.01
    elif bmax < 0:
        bmax = min(bmax + span * 0.01, 0.0)
    else:
        bmax = 0.0

    if bmin > 0:
        bmin = max(bmin - span * 0.01, 0.0)
    elif bmin < 0:
        bmin = bmin - span * 0.01
    else:
        bmin = 0.0

    if bmax == 0 and bmin == 0:
        bmax = 1.0

    m = np.log(bmax - bmin) / np.log(10.0)
    s = 10.0 ** (m - int(m))

    if s <= 2.5:
        s, umin = 0.2, 0.05
    elif s <= 5:
        s, umin = 0.5, 0.1
    elif s <= 7.5:
        s, umin = 1.0, 0.2
    else:
        s, umin = 2.0, 0.5

    t = 10.0 ** int(m)
    s, umin = s * t, umin * t
    bmin = s * int(bmin / s)
    bmax = s * (int(bmax / s) + 1)
    return bmin, bmax, s, umin


# ----------------------------------------------------------------------------
# Single-variable function plot  (UserForm19 / UserForm20)
# ----------------------------------------------------------------------------
def plot_function(f, a, b, n: int = 400, mark=None, ax=None,
                  label=None, title=None, save=False, filename=None,
                  directory=None, dpi: int = 200, show: bool = False,
                  **kwargs):
    """
    Plot y = f(x) over [a, b].

    Parameters
    ----------
    f : callable or str
    a, b : float
        Plot range.
    n : int
        Number of sample points.
    mark : float or sequence, optional
        x-values to highlight (e.g. the minimum found by a search method).
    save : bool or str
        ``True`` saves under an automatic name; a string is used as the file
        name. ``False`` (default) does not save.
    filename : str, optional
        Explicit output name, overriding ``save``.
    directory : str, optional
        Where to save. Defaults to the current working directory.
    dpi : int
        Resolution for raster formats.
    show : bool
        Call ``plt.show()`` before returning.
    """
    g = make_function(f)
    x = np.linspace(a, b, n)
    y = np.array([g(v) for v in x])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label=label or "f(x)", **kwargs)
    ax.axhline(0, color="black", lw=0.8)

    if mark is not None:
        marks = np.atleast_1d(np.asarray(mark, dtype=float))
        ax.scatter(marks, [g(v) for v in marks], color="crimson", zorder=5,
                   s=60, label="solution")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Objective function")
    ax.legend()
    ax.grid(alpha=0.3)
    return _finalise(ax, save, filename, directory, dpi, show, "function.png")


# ----------------------------------------------------------------------------
# Contour plot for two-variable objectives
# ----------------------------------------------------------------------------
def plot_contours(f, xlim, ylim, n: int = 120, levels: int = 25,
                  path=None, ax=None, title=None, log_levels: bool = False,
                  save=False, filename=None, directory=None, dpi: int = 200,
                  show: bool = False):
    """
    Contour map of a two-variable objective, optionally with a search path.

    Parameters
    ----------
    f : callable or str
        Objective of two variables.
    xlim, ylim : (lo, hi)
    path : array-like, shape (k, 2), optional
        Sequence of iterates to overlay -- pass ``result.steps[['X_1','X_2']]``.
    log_levels : bool
        Use logarithmically spaced levels, useful for steep valleys such as
        the Rosenbrock function.
    save : bool or str
        ``True`` saves under an automatic name; a string is used as the file
        name.
    directory : str, optional
        Where to save. Defaults to the current working directory.
    show : bool
        Call ``plt.show()`` before returning.
    """
    g = make_vector_function(f, n=2)
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    Z = np.array([[g([X[i, j], Y[i, j]]) for j in range(n)] for i in range(n)])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if log_levels:
        zmin = max(Z.min(), 1e-8)
        lv = np.logspace(np.log10(zmin), np.log10(Z.max()), levels)
    else:
        lv = levels

    cs = ax.contourf(X, Y, Z, levels=lv, cmap="viridis", alpha=0.85)
    ax.contour(X, Y, Z, levels=lv, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="f(x)")

    if path is not None:
        p = np.atleast_2d(np.asarray(path, dtype=float))
        ax.plot(p[:, 0], p[:, 1], "o-", color="crimson", ms=4, lw=1.5,
                label="search path")
        ax.scatter(p[-1, 0], p[-1, 1], color="white", edgecolor="black",
                   s=120, zorder=6, label="solution")
        ax.legend()

    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.set_title(title or "Objective contours")
    return _finalise(ax, save, filename, directory, dpi, show, "contours.png")


def plot_path(result, ax=None, save=False, filename=None, directory=None,
              dpi: int = 200, show: bool = False, **kwargs):
    """
    Overlay the iterate path stored in a ``Result`` onto an axis.

    Reads the ``X_1`` and ``X_2`` columns of ``result.steps``, so it works
    directly with anything returned by `directsearch` or `constrained`.
    """
    cols = [c for c in result.steps.columns if c.startswith("X_")]
    if len(cols) < 2:
        raise ValueError(
            "Result does not contain at least two X_ columns - "
            "plot_path needs a two-variable search history."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    style = {"marker": "o", "ms": 4, "lw": 1.5, "color": "crimson",
             "label": result.method}
    style.update(kwargs)
    ax.plot(result.steps[cols[0]], result.steps[cols[1]], **style)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    ax.legend()
    return _finalise(ax, save, filename, directory, dpi, show, "path.png")


# ----------------------------------------------------------------------------
# 2D LP constraint plot  (Module5.bas graph / UserForm3.frm)
# ----------------------------------------------------------------------------
def _coord_label(x, y, precision: int = 3) -> str:
    """
    Format a point as ``(x, y)``, trimming trailing zeros.

    An exact vertex reads as ``(3, 4)`` rather than ``(3.000, 4.000)``, while
    a genuinely fractional one keeps the digits it needs.
    """
    def fmt(v):
        v = v + 0.0                        # normalise -0.0 to 0.0
        text = f"{v:.{precision}f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-") else "0"

    return f"({fmt(x)}, {fmt(y)})"


def plot_constraints_2d(a, b, senses, xlim=(0, 6), ylim=(0, 8),
                        point=None, shade: bool = True,
                        nonnegative: bool = True, vertices: bool = True,
                        annotate: bool = True, hatch: str = "++",
                        precision: int = 3, ax=None, title=None,
                        save=False, filename=None, directory=None,
                        dpi: int = 200, show: bool = False):
    """
    Plot a two-variable linear programme's constraint set.

    The feasible region is filled and cross-hatched with a grid pattern. Its
    corners come from `linprog.region_polygon`, computed by exact line
    intersection, so the edges are crisp rather than the ragged outline a
    sampled mask would give. Constraint boundaries are dotted, following the
    original workbook chart, each with a fainter companion line offset toward
    its feasible side.

    Every corner of the region is marked and labelled with its coordinates in
    ``(x, y)`` form.

    Parameters
    ----------
    a : array-like, shape (m, 2)
        Constraint coefficients.
    b : array-like, shape (m,)
        Right-hand sides.
    senses : sequence of str
        '<=', '>=' or '=' per constraint.
    xlim, ylim : (lo, hi)
        Viewing box. An unbounded region is clipped to it for shading.
    point : (x1, x2), optional
        A point to test and annotate as FEASIBLE / INFEASIBLE.
    shade : bool
        Fill the feasible region.
    nonnegative : bool
        Impose x1 >= 0 and x2 >= 0, the usual LP convention.
    vertices : bool
        Mark the corner points of the feasible region.
    annotate : bool
        Label each corner with its ``(x, y)`` coordinates.
    hatch : str
        Matplotlib hatch pattern for the shaded region. ``'++'`` gives a grid;
        ``'xx'`` a diagonal mesh, ``'//'`` diagonal lines. Pass ``None`` for a
        plain fill.
    precision : int
        Decimal places in the coordinate labels. Trailing zeros are trimmed,
        so exact vertices read as ``(3, 4)`` rather than ``(3.000, 4.000)``.
    save : bool or str
        ``True`` saves under an automatic name; a string is used as the file
        name.
    filename : str, optional
        Explicit output name, overriding ``save``.
    directory : str, optional
        Where to save. Defaults to the current working directory.
    dpi : int
        Resolution for raster formats.
    show : bool
        Call ``plt.show()`` before returning.

    Returns
    -------
    matplotlib axis

    See Also
    --------
    comxl.linprog.feasible_vertices : the corner coordinates as a DataFrame.
    """
    A = np.atleast_2d(np.asarray(a, dtype=float))
    B = np.atleast_1d(np.asarray(b, dtype=float))

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.5))

    # ---- feasible region as an exact, grid-hatched polygon --------------
    poly = region_polygon(A, B, senses, xlim, ylim, nonnegative=nonnegative)
    if shade:
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], facecolor="#8ecae6", alpha=0.30,
                    edgecolor="#1d3557", linewidth=1.8, hatch=hatch,
                    zorder=1, label="feasible region")
        elif len(poly) == 2:
            # an equality constraint collapses the region to a segment, which
            # has no area to hatch but is still perfectly feasible
            ax.plot(poly[:, 0], poly[:, 1], color="#1d3557", lw=3.0,
                    solid_capstyle="round", zorder=1,
                    label="feasible region (segment)")
        elif len(poly) == 1:
            ax.scatter(poly[0, 0], poly[0, 1], s=90, color="#1d3557",
                       zorder=1, label="feasible region (single point)")
        else:
            ax.text(0.5, 0.5, "no feasible region", transform=ax.transAxes,
                    ha="center", fontsize=12, color="crimson")

    # ---- constraint boundaries, dotted as in the workbook ---------------
    colors = palette(A.shape[0])
    for i in range(A.shape[0]):
        line = constraint_line(xlim[0], xlim[1], A[i, 0], A[i, 1], B[i],
                               senses[i])
        ax.plot(line["x"], line["y"], color=colors[i], lw=1.8, ls=":",
                zorder=3,
                label=f"C{i + 1}: {A[i,0]:g}x\u2081 + {A[i,1]:g}x\u2082 "
                      f"{senses[i]} {B[i]:g}")
        # companion line offset toward the feasible side (drawLine's second
        # series), showing which way the inequality points
        ax.plot(line["x"], line["y_offset"], color=colors[i], lw=0.8, ls=":",
                alpha=0.45, zorder=2)

    # ---- corner points, labelled (x, y) ---------------------------------
    corners = feasible_vertices(A, B, senses, nonnegative=nonnegative,
                                show_steps=False)
    if vertices and not corners.empty:
        ax.scatter(corners["x1"], corners["x2"], s=80, zorder=6,
                   facecolor="white", edgecolor="#1d3557", linewidth=1.8,
                   label="corner points")
        if annotate:
            centre = corners[["x1", "x2"]].values.mean(axis=0)
            xpad = 0.10 * (xlim[1] - xlim[0])
            ypad = 0.10 * (ylim[1] - ylim[0])
            for row in corners.itertuples():
                # push the label away from the region so it does not sit on
                # the hatching, then flip it back inward for corners close to
                # an axis limit, which would otherwise be labelled off-figure
                dx = 10 if row.x1 >= centre[0] else -10
                dy = 10 if row.x2 >= centre[1] else -16
                if row.x1 - xlim[0] < xpad:
                    dx = 10
                elif xlim[1] - row.x1 < xpad:
                    dx = -10
                if row.x2 - ylim[0] < ypad:
                    dy = 12
                elif ylim[1] - row.x2 < ypad:
                    dy = -16
                ax.annotate(_coord_label(row.x1, row.x2, precision),
                            (row.x1, row.x2), textcoords="offset points",
                            xytext=(dx, dy), fontsize=9, zorder=7,
                            ha="left" if dx > 0 else "right",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                      ec="#1d3557", alpha=0.85, lw=0.6))

    if point is not None:
        ok = feasible(A, B, senses, point)
        ax.scatter(*point, s=170, zorder=8, marker="*",
                   color="green" if ok else "red", edgecolor="black")
        ax.annotate("FEASIBLE" if ok else "INFEASIBLE",
                    point, textcoords="offset points", xytext=(12, -18),
                    fontweight="bold", zorder=8,
                    color="green" if ok else "red")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title or "2D linear constraints")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return _finalise(ax, save, filename, directory, dpi, show,
                     "constraints.png")
