#!/usr/bin/env python3
"""Check which episodes are missing render files."""

from pathlib import Path
import sys

def check_episodes(base_dir):
    """Check which episodes have render files."""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    # Find all episode directories
    episode_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('ep')])

    if not episode_dirs:
        print(f"No episode directories found in {base_dir}")
        return

    print(f"Checking {len(episode_dirs)} episodes in {base_dir}\n")

    missing_render = []
    missing_any = []
    present = []

    for ep_dir in episode_dirs:
        render_png = ep_dir / "render.png"
        alpha_png = ep_dir / "alpha.png"
        depth_png = ep_dir / "depth.png"
        normal_png = ep_dir / "normal.png"

        files = list(ep_dir.glob("*"))

        if render_png.exists():
            present.append(ep_dir.name)
        else:
            missing_render.append(ep_dir.name)

        # Check if directory is completely empty
        if not files:
            missing_any.append((ep_dir.name, "EMPTY"))
        elif not any([render_png.exists(), alpha_png.exists(), depth_png.exists(), normal_png.exists()]):
            missing_any.append((ep_dir.name, "NO_IMAGES"))

    # Print summary
    print(f"Summary:")
    print(f"  Total episodes: {len(episode_dirs)}")
    print(f"  With render.png: {len(present)}")
    print(f"  Missing render.png: {len(missing_render)}")

    if missing_render:
        print(f"\nMissing render.png:")
        for ep in missing_render[:20]:  # Show first 20
            print(f"  - {ep}")
        if len(missing_render) > 20:
            print(f"  ... and {len(missing_render) - 20} more")

    if missing_any:
        print(f"\nEpisodes with issues:")
        for ep, issue in missing_any[:20]:
            print(f"  - {ep}: {issue}")
        if len(missing_any) > 20:
            print(f"  ... and {len(missing_any) - 20} more")

    # Check a few episodes in detail
    if missing_render:
        print(f"\nDetailed check of first 3 missing episodes:")
        for ep_name in missing_render[:3]:
            ep_dir = base_path / ep_name
            files = list(ep_dir.glob("*"))
            print(f"\n  {ep_name}/")
            if files:
                for f in files:
                    size = f.stat().st_size
                    print(f"    - {f.name} ({size:,} bytes)")
            else:
                print(f"    (empty)")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        check_episodes(sys.argv[1])
    else:
        check_episodes('output/bob/sphere')
