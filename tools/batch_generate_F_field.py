#!/usr/bin/env python3
"""Batch generate F-field visualizations for all episodes."""

import subprocess
import sys
from pathlib import Path

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, nargs='?', default='output/bunny_v2')
    parser.add_argument('--max_particles', type=int, default=100000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        return

    # Find all episode directories with gaussians.npz
    ep_dirs = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and d.name.startswith("ep"):
            gaussians_file = d / f"{d.name}_gaussians.npz"
            if gaussians_file.exists():
                ep_dirs.append(d)

    print(f"Generating F-field visualizations for {len(ep_dirs)} episodes in {output_dir}")
    print(f"Max particles per episode: {args.max_particles:,}")
    print("=" * 70)

    success_count = 0
    for ep_dir in ep_dirs:
        print(f"Processing {ep_dir.name}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "visualize_F_field_from_gaussians.py",
                 "--episode_dir", str(ep_dir),
                 "--max_particles", str(args.max_particles)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print(f"✅")
                success_count += 1
            else:
                print(f"❌ Failed")
                print(f"    Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("=" * 70)
    print(f"✅ Successfully generated {success_count}/{len(ep_dirs)} F-field visualizations")

if __name__ == "__main__":
    main()
