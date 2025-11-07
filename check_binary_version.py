#!/usr/bin/env python
"""Check which version of the binary is loaded."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import diffmpm_bindings

    print("=" * 70)
    print("Binary Version Check")
    print("=" * 70)

    # Check if binary file exists
    import inspect
    binary_path = inspect.getfile(diffmpm_bindings)
    print(f"\nLoaded from: {binary_path}")

    # Check file modification time
    import datetime
    mtime = os.path.getmtime(binary_path)
    mod_time = datetime.datetime.fromtimestamp(mtime)
    print(f"Last modified: {mod_time}")

    # Check file size
    size_mb = os.path.getsize(binary_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    # Try to create a simple graph to test behavior
    print("\n" + "=" * 70)
    print("Testing C++ Behavior")
    print("=" * 70)

    # This would require actual setup, so just print what to look for
    print("\nWhen you run the actual simulation, look for:")
    print("  ✅ 'Number of Iteration Passes: 1' (NEW binary)")
    print("  ❌ 'Number of Iteration Passes: 3' (OLD binary)")
    print("\nAnd in the config:")
    print("  ✅ 'Max GD iters: 10' (NEW config)")
    print("  ❌ 'Max GD iters: 1' (OLD config)")

except Exception as e:
    print(f"Error loading binary: {e}")
    print("\nThe binary may not be properly installed.")
