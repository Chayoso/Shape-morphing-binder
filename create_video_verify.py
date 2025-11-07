#!/usr/bin/env python3
"""Create video from render PNG files."""

import subprocess
import os
from pathlib import Path

# Setup paths
base_dir = Path("/home/chayo/Desktop/Shape-morphing-binder/output/verify/3_full_e2e_pcgrad")
frames_dir = base_dir / "video_frames"
frames_dir.mkdir(exist_ok=True)

# Create symlinks for all episodes
episodes = sorted(base_dir.glob("ep*/"))
print(f"Found {len(episodes)} episodes")

for ep_dir in episodes:
    ep_num = ep_dir.name.replace("ep", "")
    src = ep_dir / f"ep{ep_num}_render.png"
    dst = frames_dir / f"frame_{ep_num}.png"

    # Remove existing symlink if present
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    # Create relative symlink
    rel_src = os.path.relpath(src, frames_dir)
    dst.symlink_to(rel_src)
    print(f"  Linked: {ep_num} -> {src.name}")

print(f"\nCreated {len(list(frames_dir.glob('*.png')))} frame links")

# Create video with ffmpeg
output_video = base_dir / "morphing_evolution.mp4"
cmd = [
    "ffmpeg",
    "-framerate", "3",  # 3 fps (slow enough to see each episode)
    "-i", str(frames_dir / "frame_%03d.png"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",  # High quality (18 is visually lossless)
    "-y",  # Overwrite
    str(output_video)
]

print(f"\nCreating video: {output_video}")
print(f"Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✅ Video created successfully!")
    print(f"   Location: {output_video}")
    print(f"   Duration: ~{len(episodes)/3:.1f}s at 3 fps")
else:
    print(f"❌ Error creating video:")
    print(result.stderr)
