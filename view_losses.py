#!/usr/bin/env python3
"""
View training losses from log files.

Usage:
    python view_losses.py output/bob/sphere/
    python view_losses.py output/bob/sphere/ --episode 16
    python view_losses.py output/bob/sphere/ --plot
"""

import sys
import argparse
from pathlib import Path


def read_losses(loss_file: Path) -> dict:
    """Parse training_losses.txt file."""
    losses = {}

    if not loss_file.exists():
        return None

    with open(loss_file, 'r') as f:
        current_ep = None
        for line in f:
            line = line.strip()

            # Episode header
            if line.startswith("Episode"):
                ep_str = line.split()[1].rstrip(':')
                current_ep = int(ep_str)
                losses[current_ep] = {}

            # Loss value
            elif line and current_ep is not None and ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                try:
                    val = float(val.strip())
                    losses[current_ep][key] = val
                except ValueError:
                    pass

    return losses


def print_episode_losses(losses: dict, episode: int):
    """Print losses for a specific episode."""
    if episode not in losses:
        print(f"Episode {episode} not found in loss log")
        print(f"Available episodes: {sorted(losses.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"Episode {episode:03d} Losses")
    print(f"{'='*60}")

    ep_losses = losses[episode]

    # Print physics loss
    if 'loss_physics' in ep_losses:
        print(f"\nPhysics Loss: {ep_losses['loss_physics']:.6f}")

    # Print render loss
    if 'loss_render_total' in ep_losses:
        print(f"Render Loss:  {ep_losses['loss_render_total']:.6f}")

    # Print components
    components = [k for k in ep_losses.keys()
                  if k not in ['loss_physics', 'loss_render_total']]

    if components:
        print(f"\nRender Loss Components:")
        for key in sorted(components):
            print(f"  {key:20s}: {ep_losses[key]:.6f}")

    print(f"{'='*60}\n")


def print_all_losses(losses: dict):
    """Print summary of all episodes."""
    if not losses:
        print("No losses found")
        return

    episodes = sorted(losses.keys())

    print(f"\n{'='*80}")
    print(f"Training Loss Summary ({len(episodes)} episodes)")
    print(f"{'='*80}\n")

    print(f"{'Episode':<10} {'Physics':>12} {'Render':>12} {'Total':>12}")
    print(f"{'-'*50}")

    for ep in episodes:
        physics = losses[ep].get('loss_physics', 0)
        render = losses[ep].get('loss_render_total', 0)
        total = physics + render

        print(f"ep{ep:03d}      {physics:12.2f} {render:12.2f} {total:12.2f}")

    print(f"{'='*80}\n")


def plot_losses(losses: dict):
    """Plot loss curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib not installed")
        print("Install with: pip install matplotlib")
        return

    episodes = sorted(losses.keys())
    physics_losses = [losses[ep].get('loss_physics', 0) for ep in episodes]
    render_losses = [losses[ep].get('loss_render_total', 0) for ep in episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Physics loss
    ax1.plot(episodes, physics_losses, 'b-o', label='Physics Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Physics Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Render loss
    ax2.plot(episodes, render_losses, 'r-o', label='Render Loss', linewidth=2, markersize=4)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Render Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='View training losses from log files'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory (e.g., output/bob/sphere/)'
    )

    parser.add_argument(
        '--episode', '-e',
        type=int,
        help='Show specific episode losses'
    )

    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Plot loss curves (requires matplotlib)'
    )

    args = parser.parse_args()

    # Find loss file
    output_dir = Path(args.output_dir)
    loss_file = output_dir / "training_losses.txt"

    if not loss_file.exists():
        print(f"Error: Loss file not found: {loss_file}")
        print(f"\nThis file is created automatically when you run:")
        print(f"  python run.py -c config.yaml --png")
        print(f"\nIf you ran training before this feature was added, you'll need to re-run.")
        return 1

    # Read losses
    losses = read_losses(loss_file)

    if not losses:
        print(f"Error: Could not parse losses from {loss_file}")
        return 1

    # Display results
    if args.episode is not None:
        print_episode_losses(losses, args.episode)
    elif args.plot:
        plot_losses(losses)
    else:
        print_all_losses(losses)

    return 0


if __name__ == '__main__':
    sys.exit(main())
