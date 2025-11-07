#!/usr/bin/env python3
"""
Quick convergence checker - extracts episode-level losses from log file
"""

import re
import sys
from pathlib import Path

def parse_log(log_file):
    """Extract episode losses from log file."""
    physics_losses = []
    render_losses = []
    current_episode = -1

    with open(log_file, 'r') as f:
        for line in f:
            # Episode start
            if 'Episode' in line and 'START' in line:
                match = re.search(r'Episode (\d+) START', line)
                if match:
                    current_episode = int(match.group(1))

            # Physics loss (initial)
            if 'Initial physics loss:' in line:
                match = re.search(r'loss: ([\d.]+)', line)
                if match:
                    loss = float(match.group(1))
                    physics_losses.append((current_episode, loss))

            # Render loss (final = Pass 3)
            if 'Render loss:' in line:
                match = re.search(r'loss: ([\d.]+)', line)
                if match:
                    loss = float(match.group(1))
                    # Store all render losses, we'll take the last one per episode
                    if current_episode >= 0:
                        # Update if same episode, append if new
                        if render_losses and render_losses[-1][0] == current_episode:
                            render_losses[-1] = (current_episode, loss)
                        else:
                            render_losses.append((current_episode, loss))

    return physics_losses, render_losses

def print_summary(physics_losses, render_losses):
    """Print formatted convergence summary."""
    print("\n" + "="*70)
    print("  CONVERGENCE SUMMARY")
    print("="*70)

    if not physics_losses:
        print("\n❌ No losses found in log file! Training may not have started.")
        return

    print(f"\n📊 Collected {len(physics_losses)} episodes\n")

    # Header
    print(f"{'Episode':<10} {'Physics Loss':<15} {'Δ Phys':<12} {'Render Loss':<15} {'Δ Render':<12}")
    print("-" * 70)

    prev_phys = None
    prev_render = None

    for i, (ep, phys_loss) in enumerate(physics_losses):
        # Find corresponding render loss
        render_loss = None
        for r_ep, r_loss in render_losses:
            if r_ep == ep:
                render_loss = r_loss
                break

        # Calculate deltas
        delta_phys = ""
        delta_render = ""

        if prev_phys is not None:
            change = ((phys_loss - prev_phys) / prev_phys) * 100
            delta_phys = f"{change:+.1f}%"

        if prev_render is not None and render_loss is not None:
            change = ((render_loss - prev_render) / prev_render) * 100
            delta_render = f"{change:+.1f}%"

        # Format
        phys_str = f"{phys_loss:.2f}"
        render_str = f"{render_loss:.2f}" if render_loss is not None else "N/A"

        print(f"{ep:<10} {phys_str:<15} {delta_phys:<12} {render_str:<15} {delta_render:<12}")

        prev_phys = phys_loss
        if render_loss is not None:
            prev_render = render_loss

    print("-" * 70)

    # Final assessment
    if len(physics_losses) >= 2:
        initial_phys = physics_losses[0][1]
        final_phys = physics_losses[-1][1]
        total_reduction = ((final_phys - initial_phys) / initial_phys) * 100

        print(f"\n🎯 Total Physics Reduction: {total_reduction:.1f}%")

        if render_losses:
            initial_render = render_losses[0][1]
            final_render = render_losses[-1][1]
            render_reduction = ((final_render - initial_render) / initial_render) * 100
            print(f"🎯 Total Render Reduction:  {render_reduction:.1f}%")

        print()

        # Status assessment
        if total_reduction < -20:
            print("✅ Status: CONVERGING WELL! (>20% reduction)")
        elif total_reduction < -5:
            print("⚠️  Status: Slow convergence (<20% reduction)")
        elif total_reduction > 5:
            print("❌ Status: DIVERGING! Loss is increasing!")
        else:
            print("⏸️  Status: Plateaued (no significant change)")

    print("="*70 + "\n")

def main():
    if len(sys.argv) < 2:
        # Try default log file
        log_files = [
            'logs/test_pcgrad_rebuild.log',
            'logs/test_pcgrad_fix.log',
            'logs/training.log'
        ]
        log_file = None
        for lf in log_files:
            if Path(lf).exists():
                log_file = lf
                break

        if not log_file:
            print("Usage: python check_convergence.py <logfile>")
            print("\nNo default log file found. Please specify log file path.")
            sys.exit(1)
    else:
        log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"❌ Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"📂 Reading log: {log_file}")

    physics_losses, render_losses = parse_log(log_file)
    print_summary(physics_losses, render_losses)

if __name__ == '__main__':
    main()
