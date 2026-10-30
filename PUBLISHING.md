# Publishing COMXL to GitHub

A one-time checklist for getting the repository up and the web app live.

## 1. Create the repository

On GitHub, create a new **public** repository named `COMXL` — no README,
`.gitignore` or licence, since this folder already has them.

## 2. Push this folder

```bash
cd COMXL
git init -b main
git add .
git commit -m "COMXL: Python port of the COMxL optimisation workbook"
git remote add origin https://github.com/<your-username>/COMXL.git
git push -u origin main
```

## 3. Turn on GitHub Pages

**Settings → Pages → Build and deployment → Source: GitHub Actions.**

Select GitHub Actions, not "Deploy from a branch". The `deploy pages`
workflow is already in `.github/workflows/pages.yml` and publishes `docs/`
on every push to `main` that touches it.

The first deployment takes a minute or two. The site then lives at:

```
https://<your-username>.github.io/COMXL/
```

## 4. Fix the links

Two places name the repository owner. Update them to your username:

- `README.md` — the "Run it in your browser" link near the top
- `docs/index.html` — the two `masthead-links` anchors

```bash
grep -rn "Maspolic-Amo-Yeboah" README.md docs/index.html
```

## 5. Check it worked

- **Actions** tab: both `tests` and `deploy pages` should be green.
- Open the site. The first load fetches about 15 MB of Python runtime from a
  CDN and takes 10–30 seconds; afterwards it is cached. If it sticks on
  "Starting the Python runtime", the browser console will say why.
- Click **Solve** on the 1-D panel. It should report x = 2.25 with an
  iteration table and a plot.

---

## Keeping it working

### After changing anything in `comxl/`

The site installs a committed wheel, so rebuild it or the published page keeps
running the old code:

```bash
python tools/build_web_wheel.py
git add docs/wheels && git commit -m "Rebuild Pages wheel"
```

The `tests` workflow fails if the committed wheel is a different version from
the one it builds, so a stale wheel cannot reach the site unnoticed.

### Before pushing a change to `docs/`

```bash
pip install playwright && python -m playwright install chromium
python tools/test_site.py
```

This serves `docs/` locally, loads it in headless Chromium, drives all six
panels and asserts the results — 27 checks. Catching a wiring mistake here is
much cheaper than catching it after publication.

### If the site cannot reach the CDN

On a restricted network, point the suite at a local Pyodide distribution
instead:

```bash
# download pyodide-0.26.2.tar.bz2 from github.com/pyodide/pyodide/releases
tar xjf pyodide-0.26.2.tar.bz2
COMXL_PYODIDE_DIR=./pyodide python tools/test_site.py
```

The published page still uses the CDN; this only affects local testing.

## Optional

**Publish to PyPI** so the package installs by name:

```bash
python -m build
python -m twine upload dist/*
```

Check that the name `comxl` is free first, and bump `version` in
`pyproject.toml` for each release.

**Add repository topics** — `optimization`, `numerical-methods`, `vba-port`,
`simplex`, `nelder-mead`, `teaching` — and set the description to the README's
first line so the repository is findable.
