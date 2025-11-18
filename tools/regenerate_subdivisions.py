#!/usr/bin/env python3
"""Regenerate subdivision visualizations for all episodes."""

import subprocess
import sys
from pathlib import Path

def regenerate_subdivision(ep_dir: Path):
    """Regenerate subdivision visualization for one episode."""
    # Check if PLY file exists
    ply_files = list(ep_dir.glob("*_surface_*.ply"))
    if not ply_files:
        return False

    # Run visualize_subdivision.py
    result = subprocess.run(
        [sys.executable, "visualize_subdivision.py",
         "--episode", str(ep_dir),
         "--downsample", "0.1"],  # 10% for speed
        capture_output=True,
        text=True,
        timeout=60
    )

    return result.returncode == 0

def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("output/bunny_v2")

    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        return

    # Find all episode directories with PLY files
    ep_dirs = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ep"):
            ply_files = list(d.glob("*_surface_*.ply"))
            if ply_files:
                ep_dirs.append(d)

    print(f"Regenerating subdivision visualizations for {len(ep_dirs)} episodes in {output_dir}")
    print("=" * 70)

    success_count = 0
    for ep_dir in ep_dirs:
        print(f"Processing {ep_dir.name}...", end=" ", flush=True)
        try:
            if regenerate_subdivision(ep_dir):
                print(f"✅")
                success_count += 1
            else:
                print(f"❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("=" * 70)
    print(f"✅ Successfully regenerated {success_count}/{len(ep_dirs)} subdivision visualizations")

if __name__ == "__main__":
    main()
