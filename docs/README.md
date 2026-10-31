# docs/ — the GitHub Pages site

This folder is published as the COMXL web app. It runs the real `comxl`
package in the browser; no algorithm is reimplemented in JavaScript.

```
docs/
├── index.html          markup for the six solver panels
├── css/style.css       stylesheet
├── js/app.js           boots Pyodide, wires forms, renders results
├── js/runner.py        the Python bridge: form values in, JSON-safe out
├── wheels/             the comxl wheel the page installs, plus manifest.json
└── .nojekyll           serve files starting with _ and skip Jekyll
```

## How a click becomes a result

1. `app.js` boots Pyodide, loads NumPy, pandas and Matplotlib, then
   `micropip.install`s the wheel named in `wheels/manifest.json`.
2. `runner.py` is fetched and executed, exposing one function per panel.
3. A form submit sends its values as JSON to the matching function, which
   calls `comxl` and returns a dict: the solution, the iteration table as
   columns and rows, and any figure as a base64 PNG.
4. `app.js` renders that dict.

So the browser and the test suite exercise the same code. The only
browser-specific piece is turning a Matplotlib figure into a data URI.

## Notes

- First load fetches roughly 15 MB of runtime from the jsDelivr CDN and is
  cached afterwards. There is no offline mode.
- Asset paths resolve against the page URL, so the site works from a project
  subpath (`/COMXL/`) as well as from a user-site root.
- seaborn is not installed in the browser; `comxl.plotting` falls back to
  Matplotlib styling, which is why seaborn is an optional extra.
