#!/usr/bin/env python3
"""
End-to-end check of the GitHub Pages site.

Serves docs/ locally, loads it in headless Chromium, waits for Pyodide to boot
and the comxl wheel to install, then drives each panel and asserts the numbers
that come back. Catching a broken wheel or a wiring mistake here is much
cheaper than catching it after it is published.

    python tools/test_site.py

Requires playwright and a Chromium build:
    pip install playwright && python -m playwright install chromium

Pyodide itself is fetched from a CDN, so this normally needs outbound network
access. On an air-gapped or proxied machine, point COMXL_PYODIDE_DIR at an
unpacked Pyodide distribution of the matching version and requests to the CDN
are served from there instead:

    export COMXL_PYODIDE_DIR=/path/to/pyodide
    python tools/test_site.py

Download that distribution from
https://github.com/pyodide/pyodide/releases (pyodide-<version>.tar.bz2).
"""

from __future__ import annotations

import functools
import http.server
import mimetypes
import os
import socket
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BOOT_TIMEOUT_MS = 600_000
RUN_TIMEOUT_MS = 180_000

# Set to an unpacked Pyodide distribution to run without reaching the CDN.
LOCAL_PYODIDE = os.environ.get("COMXL_PYODIDE_DIR", "")


def install_pyodide_route(page, local_dir: Path):
    """
    Serve cdn.jsdelivr.net/pyodide/** from a local distribution.

    Lets the suite run where the CDN is unreachable, without changing what the
    published page requests.
    """
    served, missing = {"n": 0}, set()

    def handler(route, request):
        name = request.url.split("/full/")[-1].split("?")[0]
        candidate = local_dir / name
        if candidate.is_file():
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            if name.endswith(".wasm"):
                ctype = "application/wasm"
            elif name.endswith((".js", ".mjs")):
                ctype = "text/javascript"
            elif name.endswith(".json"):
                ctype = "application/json"
            served["n"] += 1
            route.fulfill(status=200, body=candidate.read_bytes(),
                          headers={"Content-Type": ctype,
                                   "Access-Control-Allow-Origin": "*"})
        else:
            missing.add(name)
            route.fulfill(status=404, body=b"not in local distribution")

    page.route("**/cdn.jsdelivr.net/pyodide/**", handler)
    return served, missing


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(directory: Path, port: int):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class Checks:
    def __init__(self):
        self.passed = 0
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failed.append(label)
            print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))


def panel_text(page, panel_id: str) -> str:
    return page.inner_text(f"#{panel_id}")


def contains(haystack: str, needle: str) -> bool:
    """Case-insensitive containment -- CSS uppercases the section headings."""
    return needle.lower() in haystack.lower()


def run_panel(page, tab_label, form_id, output_id, fields=None):
    """Switch to a tab, fill fields, submit, and return the output text."""
    page.click(f".tab:has-text('{tab_label}')")
    for name, value in (fields or {}).items():
        selector = f"#{form_id} [name='{name}']"
        if page.get_attribute(selector, "type") == "checkbox":
            page.set_checked(selector, bool(value))
        elif page.eval_on_selector(selector, "e => e.tagName") == "SELECT":
            page.select_option(selector, str(value))
        else:
            page.fill(selector, str(value))
    page.click(f"#{form_id} .go")
    page.wait_for_function(
        f"() => {{ const o = document.querySelector('#{output_id}');"
        f" return o && !o.querySelector('[data-busy]'); }}",
        timeout=RUN_TIMEOUT_MS,
    )
    return page.inner_text(f"#{output_id}")


def main() -> int:
    if not (DOCS / "wheels" / "manifest.json").exists():
        print("docs/wheels/manifest.json missing - run tools/build_web_wheel.py first.")
        return 1

    port = free_port()
    serve(DOCS, port)
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"Serving docs/ at {url}\n")

    checks = Checks()
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        served = missing = None
        if LOCAL_PYODIDE:
            local = Path(LOCAL_PYODIDE)
            if not local.is_dir():
                print(f"COMXL_PYODIDE_DIR is not a directory: {local}")
                return 1
            served, missing = install_pyodide_route(page, local)
            print(f"Serving Pyodide from {local} instead of the CDN.")

        page.goto(url)
        print("Booting Pyodide and installing the wheel (this takes a while)…")
        try:
            page.wait_for_selector("#app:not([hidden])", timeout=BOOT_TIMEOUT_MS)
        except Exception:
            print("\nBoot failed. Status on the page was:")
            print(" ", page.inner_text("#boot"))
            for e in console_errors[:10]:
                print("  console:", e[:200])
            if missing:
                print("  not found locally:", ", ".join(sorted(missing)[:12]))
            browser.close()
            return 1
        print("Runtime ready.\n")
        checks.check("app becomes visible after boot", True)

        # ── 1-D ───────────────────────────────────────────────────────
        print("1-D search")
        out = run_panel(page, "1-D search", "form-1d", "out-1d", {
            "method": "goldensection", "expression": "x^2 - 4*x + 3",
            "a": "0", "b": "5", "tol": "0.000001", "max_iter": "200",
        })
        checks.check("golden section finds x = 2", "2.0000" in out, out[:160])
        checks.check("golden section reports convergence", "converged" in out)
        checks.check("iteration table rendered", contains(out, "Iterations"), out[:200])
        checks.check("objective plot rendered",
                     page.locator("#out-1d .plot img").count() == 1)

        out = run_panel(page, "1-D search", "form-1d", "out-1d", {
            "method": "newtonraphson", "expression": "x**4 - 3*x**3 + 2",
            "x0": "3", "tol": "0.0000000001", "max_iter": "50",
        })
        checks.check("Newton-Raphson finds x = 2.25", "2.25" in out, out[:160])

        out = run_panel(page, "1-D search", "form-1d", "out-1d", {
            "method": "fibonacci", "expression": "(x-2)^2",
            "a": "0", "b": "5", "n": "25",
        })
        checks.check("Fibonacci search runs", "1.99999" in out or "2.0000" in out,
                     out[:160])

        # a deliberately wrong input must produce a readable error
        out = run_panel(page, "1-D search", "form-1d", "out-1d", {
            "method": "goldensection", "expression": "x1 + x2", "a": "0", "b": "5",
        })
        checks.check("bad objective reports an error",
                     page.locator("#out-1d .result-error").count() == 1, out[:160])

        # ── multivariable ─────────────────────────────────────────────
        print("\nMultivariable")
        out = run_panel(page, "Multivariable", "form-multi", "out-multi", {
            "method": "neldermead",
            "expression": "(1-x1)^2 + 100*(x2-x1^2)^2",
            "x0": "-1.2 1.0", "lower": "", "upper": "", "max_iter": "2000",
        })
        checks.check("Nelder-Mead solves Rosenbrock",
                     "1," in out and "converged" in out, out[:200])
        checks.check("contour plot rendered",
                     page.locator("#out-multi .plot img").count() == 1)

        out = run_panel(page, "Multivariable", "form-multi", "out-multi", {
            "method": "powell", "expression": "(1-x1)^2 + 100*(x2-x1^2)^2",
            "x0": "-1.2 1.0", "tol": "0.0000001",
        })
        checks.check("Powell solves Rosenbrock", "converged" in out, out[:200])

        out = run_panel(page, "Multivariable", "form-multi", "out-multi", {
            "method": "neldermead", "expression": "(1-x1)^2 + 100*(x2-x1^2)^2",
            "x0": "0.1 0.1", "lower": "0 0", "upper": "0.5 0.5",
            "max_iter": "3000",
        })
        checks.check("box-constrained run hits the 0.5 face", "0.5" in out, out[:200])

        # ── linear programming ────────────────────────────────────────
        print("\nLinear programming")
        out = run_panel(page, "Linear programming", "form-lp", "out-lp", {
            "matrix": "2 1 10\n1 3 15\n4 3 0", "mode": "max", "hatch": "++",
        })
        checks.check("simplex optimum z = 24", "24" in out, out[:200])
        checks.check("corner points listed", contains(out, "Corner points"), out[:200])
        checks.check("corner (3, 4) present", "3" in out and "4" in out)
        checks.check("feasible region plotted",
                     page.locator("#out-lp .plot img").count() == 1)
        checks.check("tableaux available",
                     page.locator("#out-lp details.tableaux").count() == 1)

        shots = ROOT / "docs" / "assets"
        shots.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(shots / "screenshot.png"), full_page=True)

        out = run_panel(page, "Linear programming", "form-lp", "out-lp", {
            "matrix": "1 2 4\n3 1 6\n6 3 0", "mode": "min",
        })
        checks.check("minimisation gives z = 13.2", "13.2" in out, out[:200])

        # ── KKT ───────────────────────────────────────────────────────
        print("\nKKT")
        out = run_panel(page, "Constrained (KKT)", "form-kkt", "out-kkt", {
            "expression": "(x1-2)^2 + (x2-1)^2", "n_vars": "2", "x0": "0.5 0.5",
            "inequality": "x1 + x2 - 2\n-x1\n-x2", "equality": "",
        })
        checks.check("KKT finds (1.5, 0.5)", "1.5" in out and "0.5" in out, out[:200])
        checks.check("active set reported", "g1" in out, out[:200])

        # ── matrices ──────────────────────────────────────────────────
        print("\nMatrices")
        matrix = "4 -2 1\n-2 4 -2\n1 -2 4"
        out = run_panel(page, "Matrices", "form-matrix", "out-matrix", {
            "operation": "determinant", "matrix": matrix,
        })
        checks.check("determinant = 36", "36" in out, out[:160])

        out = run_panel(page, "Matrices", "form-matrix", "out-matrix", {
            "operation": "eigenvalues", "matrix": matrix,
        })
        checks.check("eigenvalues include 7.37", "7.37" in out, out[:200])

        out = run_panel(page, "Matrices", "form-matrix", "out-matrix", {
            "operation": "solve", "matrix": matrix, "rhs": "11 -16 17",
        })
        checks.check("linear solve gives x = (1, -2, 3)",
                     "-2" in out and "3" in out, out[:200])

        out = run_panel(page, "Matrices", "form-matrix", "out-matrix", {
            "operation": "inverse", "matrix": matrix,
        })
        checks.check("inverse computed", contains(out, "Inverse"), out[:160])

        # ── derivatives ───────────────────────────────────────────────
        print("\nDerivatives")
        out = run_panel(page, "Derivatives", "form-calc", "out-calc", {
            "expression": "x1^2 + 3*x1*x2 + 2*x2^2", "point": "1 2",
            "what": "gradient",
        })
        checks.check("gradient has 8 and 11", "8" in out and "11" in out, out[:200])

        out = run_panel(page, "Derivatives", "form-calc", "out-calc", {
            "expression": "x1^2 - x2^2", "point": "0 0", "what": "hessian",
        })
        checks.check("Hessian detects a saddle", "SADDLE" in out, out[:200])

        browser.close()

    real_errors = [e for e in console_errors
                   if "favicon" not in e.lower() and "404" not in e]
    checks.check("no uncaught console errors", not real_errors,
                 "; ".join(real_errors[:3])[:200])

    print(f"\n{checks.passed} passed, {len(checks.failed)} failed")
    for name in checks.failed:
        print("  failed:", name)
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
