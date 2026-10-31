# Contributing

## Setup

```bash
git clone https://github.com/maspolic/COMXL.git
cd COMXL
pip install -e ".[dev,plots]"
pytest tests/ -q
```

## Before opening a pull request

```bash
pytest tests/ -q                      # the suite
pytest --doctest-modules comxl/ -q    # docstring examples
```

If you touched anything in `comxl/`, rebuild the wheel the website installs:

```bash
python tools/build_web_wheel.py
git add docs/wheels
```

CI fails if the committed wheel is a different version from the one it builds,
so a stale wheel cannot reach the published site unnoticed.

If you touched `docs/`, run the end-to-end check too:

```bash
pip install playwright && python -m playwright install chromium
python tools/test_site.py
```

## What a change to a solver needs

Every solver is tested against a problem with a **known analytical answer**, not
against its own previous output. A new method, or a change to an existing one,
should come with a test of that kind — otherwise a regression only shows up as
a slightly different number that nobody can adjudicate.

Where this port deliberately departs from the VBA, the source says so at the
point of departure and `README.md` lists it under *Fixes to the original
behaviour*. Keep that up to date: the value of the port is partly in knowing
exactly where it disagrees with the workbook.

Three conventions worth keeping:

- **No optimisation libraries.** Gauss elimination, the simplex pivots,
  Bairstow root-finding and every search method are written out, because
  providing them is the point of the package. NumPy is for array storage and
  arithmetic.
- **`show_steps` stays.** Every solver prints its iteration table by default and
  always records it on `result.steps`. That is the feature the workbook was
  valued for.
- **seaborn stays optional.** `comxl.plotting` must keep working when it is
  absent, or the browser build breaks. CI checks this explicitly.

## Style

Follow what is already there: British spelling in prose, NumPy-style
docstrings naming the VBA routine each function came from, and comments that
explain *why* rather than restating the code.
