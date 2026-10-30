#!/usr/bin/env python3
"""
Build the wheel that the GitHub Pages site installs in the browser.

The web app does not reimplement any algorithm: it loads this wheel into
Pyodide and calls the same `comxl` package the test suite exercises. Rebuild
and commit the wheel whenever the package changes, or the published site will
keep running the previous version.

    python tools/build_web_wheel.py

Writes to docs/wheels/ and refreshes docs/wheels/manifest.json, which the
front end reads to discover the wheel filename and version.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEEL_DIR = ROOT / "docs" / "wheels"


def main() -> int:
    print("Building wheel from", ROOT)
    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("Wheel build failed.", file=sys.stderr)
        return result.returncode

    wheels = sorted(dist.glob("comxl-*.whl"))
    if not wheels:
        print("No wheel produced.", file=sys.stderr)
        return 1
    wheel = wheels[-1]

    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    for old in WHEEL_DIR.glob("*.whl"):
        old.unlink()
    target = WHEEL_DIR / wheel.name
    shutil.copy2(wheel, target)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    version = wheel.name.split("-")[1]

    manifest = {
        "wheel": wheel.name,
        "version": version,
        "sha256": digest,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (WHEEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size_kb = target.stat().st_size / 1024
    print(f"\n  {target.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    print(f"  version {version}")
    print(f"  sha256  {digest[:16]}...")
    print("\nCommit docs/wheels/ to publish this build to GitHub Pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
