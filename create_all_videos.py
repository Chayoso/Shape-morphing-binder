#!/usr/bin/env python3
"""
Create videos from all experiment output directories.
Handles the ep000_render.png naming convention.
"""

import subprocess
import os
from pathlib import Path
import re

# Find all experiment directories
output_dir = Path('/home/chayo/Desktop/Shape-morphing-binder/output')

# Find all directories containing episode subdirectories
exp_dirs = []
for root, dirs, files in os.walk(output_dir):
    root_path = Path(root)
    # Check if this directory has episode subdirectories
    has_episodes = any(re.match(r'ep\d+', d) for d in dirs)
    if has_episodes:
        exp_dirs.append(root_path)

print(f"Found {len(exp_dirs)} experiment directories:")
for exp_dir in exp_dirs:
    print(f"  - {exp_dir.relative_to(output_dir.parent)}")

# Create videos for each experiment
successes = 0
for exp_dir in exp_dirs:
    print(f"\n{'='*80}")
    print(f"Processing: {exp_dir.name}")
    print(f"{'='*80}")

    # Find all episode directories
    episode_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and re.match(r'ep\d+', d.name)],
                         key=lambda x: int(re.search(r'\d+', x.name).group()))

    if not episode_dirs:
        print("  No episodes found, skipping...")
        continue

    # Create ffmpeg input file list
    input_list_path = exp_dir / 'ffmpeg_input.txt'

    frame_files = []
    for ep_dir in episode_dirs:
        # Look for render image
        render_file = ep_dir / f"{ep_dir.name}_render.png"
        if render_file.exists():
            frame_files.append(render_file)

    if not frame_files:
        print(f"  No render files found, skipping...")
        continue

    print(f"  Found {len(frame_files)} frames")

    # Write input file list for ffmpeg concat
    with open(input_list_path, 'w') as f:
        for frame_file in frame_files:
            f.write(f"file '{frame_file.absolute()}'\n")
            f.write(f"duration 0.1\n")  # 10 FPS = 0.1s per frame
        # Add last frame without duration for proper ending
        if frame_files:
            f.write(f"file '{frame_files[-1].absolute()}'\n")

    # Output video path
    video_path = exp_dir / f"{exp_dir.name}_animation.mp4"

    # Create video with ffmpeg
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(input_list_path),
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(video_path)
    ]

    print(f"  Creating video: {video_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Video created: {size_mb:.2f} MB ({len(frame_files)} frames)")
        successes += 1
    else:
        print(f"  ❌ Failed to create video")
        if result.stderr:
            print(f"  Error: {result.stderr[:300]}")

    # Clean up input list
    if input_list_path.exists():
        input_list_path.unlink()

print(f"\n{'='*80}")
print(f"Video creation complete! {successes}/{len(exp_dirs)} successful")
print(f"{'='*80}")
