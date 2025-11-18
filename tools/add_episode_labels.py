#!/usr/bin/env python3
"""Add episode number labels to render images."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

def add_episode_label(input_path, output_path, episode_num):
    """Add episode number label to an image."""
    # Open image
    img = Image.open(input_path)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    except:
        font = ImageFont.load_default()

    # Create text
    text = f"Episode {episode_num}"

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position at top center
    x = (img.width - text_width) // 2
    y = 50

    # Draw background box
    padding = 20
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)
    )

    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Save
    img.save(output_path)

def main():
    if len(sys.argv) != 3:
        print("Usage: python add_episode_labels.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True)

    # Find all episode directories
    episode_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('ep')])

    print(f"Found {len(episode_dirs)} episodes")

    for ep_dir in episode_dirs:
        # Extract episode number
        ep_num = int(ep_dir.name[2:])

        # Find render image
        render_files = list(ep_dir.glob('*_render.png'))
        if not render_files:
            print(f"  Warning: No render image found in {ep_dir.name}")
            continue

        input_file = render_files[0]
        output_file = output_dir / f"{ep_num:03d}.png"

        print(f"  Processing {ep_dir.name}...")
        add_episode_label(input_file, output_file, ep_num)

    print(f"\n✅ Saved {len(episode_dirs)} labeled images to {output_dir}")

if __name__ == "__main__":
    main()
