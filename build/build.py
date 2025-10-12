#!/usr/bin/env python3
"""
Convenience builder for the DiffMPM Python bindings (pip + release flags).

Usage:
  python build.py             # build & test import
  NO_LTO=1 python build.py    # disable LTO
  DIFFMPM_NATIVE=1 python build.py    # enable -march=native (POSIX)
  DIFFMPM_FASTMATH=1 python build.py  # enable -ffast-math   (POSIX)
"""

import os, subprocess, sys, textwrap

def run(cmd: str):
    print(f"[build] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def main():
    # Clean previous in-place builds if any
    for pat in ("build", "dist", "*.egg-info", "diffmpm_bindings.*.so", "diffmpm_bindings.*.pyd"):
        run(f"rm -rf {pat} 2>/dev/null || true")

    # Build & install via pip to ensure a proper wheel installation
    run("python -m pip install -v .")

    # Import test
    code = "import diffmpm_bindings as m; print('OK:', hasattr(m,'CompGraph'))"
    run(f"python - <<'PY'\n{code}\nPY")

if __name__ == "__main__":
    main()