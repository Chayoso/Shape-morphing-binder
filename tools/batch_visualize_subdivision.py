#!/usr/bin/env python3
"""Batch generate subdivision visualizations for all episodes in an experiment."""

import subprocess
import sys
from pathlib import Path

def main():
    experiment_dir = Path("output/bunny_v2")

    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])

    episodes = sorted(experiment_dir.glob("ep*"))

    print(f"Found {len(episodes)} episodes in {experiment_dir}")
    print("=" * 70)

    success_count = 0
    failed = []

    for ep in episodes:
        if not ep.is_dir():
            continue

        print(f"\nProcessing {ep.name}...")

        try:
            result = subprocess.run(
                [sys.executable, "visualize_subdivision.py",
                 "--episode", str(ep),
                 "--downsample", "0.1"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Extract key info from output
                for line in result.stdout.split('\n'):
                    if 'Loaded' in line or 'Generated' in line or 'Saved' in line:
                        print(f"  {line}")
                success_count += 1
            else:
                print(f"  ❌ Failed: {result.stderr.split(chr(10))[0]}")
                failed.append(ep.name)

        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout")
            failed.append(ep.name)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed.append(ep.name)

    print("\n" + "=" * 70)
    print(f"✅ Successfully generated: {success_count}/{len(episodes)}")

    if failed:
        print(f"❌ Failed episodes: {', '.join(failed)}")

    print(f"\nVisualization files saved in each episode directory:")
    print(f"  {experiment_dir}/ep*/subdivision_visualization.png")

if __name__ == "__main__":
    main()
