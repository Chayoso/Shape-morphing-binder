#!/usr/bin/env python3
"""Quick check for episode progress"""

import json
from pathlib import Path
import sys

def check_progress(output_dir='output/sphere/spot'):
    """Check all episodes and show trend."""
    episodes = []

    base_path = Path(output_dir)
    if not base_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return

    # Find all episode directories
    for ep_dir in sorted(base_path.glob('ep*')):
        if ep_dir.is_dir():
            summary_file = ep_dir / f"{ep_dir.name}_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    episodes.append({
                        'ep': ep_dir.name,
                        'physics_loss': data.get('loss_physics_final', 0),
                        'num_points': data.get('num_surface_points', 0)
                    })

    if not episodes:
        print(f"❌ No episodes found in {output_dir}")
        return

    print("\n" + "="*60)
    print(f"  EPISODE PROGRESS ({len(episodes)} episodes)")
    print("="*60)

    print(f"\n{'Episode':<10} {'Physics Loss':<15} {'Change':<15} {'Points':<10}")
    print("-"*60)

    prev_loss = None
    for ep_data in episodes:
        ep = ep_data['ep']
        loss = ep_data['physics_loss']
        points = ep_data['num_points']

        change = ""
        if prev_loss is not None:
            delta = ((loss - prev_loss) / prev_loss) * 100
            change = f"{delta:+.1f}%"
            if delta < -10:
                change += " ✅"
            elif delta > 5:
                change += " ❌"

        print(f"{ep:<10} {loss:<15.2f} {change:<15} {points:<10}")
        prev_loss = loss

    print("-"*60)

    # Summary
    if len(episodes) >= 2:
        initial = episodes[0]['physics_loss']
        final = episodes[-1]['physics_loss']
        total_reduction = ((final - initial) / initial) * 100

        print(f"\n📊 Total reduction: {total_reduction:.1f}%")

        if total_reduction < -20:
            print("✅ Status: CONVERGING WELL!")
        elif total_reduction < -5:
            print("⚠️  Status: Slow convergence")
        elif total_reduction > 5:
            print("❌ Status: DIVERGING - check settings!")
        else:
            print("⏸️  Status: Plateau")

    print("="*60 + "\n")

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output/sphere/spot'
    check_progress(output_dir)
