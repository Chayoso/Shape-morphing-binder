#!/usr/bin/env python3
"""
create_video.py - Create videos from episode render sequences

Creates MP4 videos from render.png files in episode directories.

Usage:
    # Create video from a specific output directory
    python create_video.py output/bob/sphere/

    # Create video with custom framerate
    python create_video.py output/bob/sphere/ --fps 10

    # Create video with custom output name
    python create_video.py output/bob/sphere/ --output my_animation.mp4

    # Process all subdirectories
    python create_video.py output/ --recursive

Author: CHAYO
Version: 1.0
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import re


def find_episode_dirs(base_dir: Path) -> list:
    """
    Find all episode directories (ep000, ep001, etc.) in the base directory.

    Returns:
        List of episode directories sorted by episode number
    """
    episode_dirs = []

    for item in base_dir.iterdir():
        if item.is_dir() and re.match(r'ep\d+', item.name):
            episode_dirs.append(item)

    # Sort by episode number
    episode_dirs.sort(key=lambda x: int(re.search(r'\d+', x.name).group()))

    return episode_dirs


def check_ffmpeg():
    """Check if ffmpeg is available."""
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg is not installed or not in PATH")
        print("Install with: sudo apt install ffmpeg")
        return False
    return True


def create_video(base_dir: Path, output_file: Path, fps: int = 5,
                image_name: str = "render.png", quality: str = "high") -> bool:
    """
    Create video from episode images using ffmpeg.

    Args:
        base_dir: Directory containing episode folders
        output_file: Output video file path
        fps: Frames per second
        image_name: Name of image file to use (e.g., "render.png", "alpha.png")
        quality: Video quality ("high", "medium", "low")

    Returns:
        True if successful, False otherwise
    """
    episode_dirs = find_episode_dirs(base_dir)

    if not episode_dirs:
        print(f"Error: No episode directories found in {base_dir}")
        return False

    print(f"Found {len(episode_dirs)} episode directories")
    print(f"First episode: {episode_dirs[0].name}")
    print(f"Last episode: {episode_dirs[-1].name}")

    # Create temporary directory for symlinks
    temp_dir = base_dir / "temp_video_frames"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Create numbered symlinks
        print(f"\nCreating symlinks for {image_name} files...")
        frame_count = 0
        missing_frames = []

        for i, ep_dir in enumerate(episode_dirs):
            image_path = ep_dir / image_name

            if image_path.exists():
                # Create symlink with zero-padded number
                link_path = temp_dir / f"frame_{i:04d}.png"

                # Remove existing symlink if it exists
                if link_path.exists():
                    link_path.unlink()

                # Create symlink (or copy on Windows)
                try:
                    link_path.symlink_to(image_path.resolve())
                except OSError:
                    # Windows may not support symlinks, so copy instead
                    shutil.copy2(image_path, link_path)

                frame_count += 1
            else:
                missing_frames.append(ep_dir.name)

        if missing_frames:
            print(f"Warning: {len(missing_frames)} episodes missing {image_name}:")
            for ep in missing_frames[:5]:  # Show first 5
                print(f"  - {ep}")
            if len(missing_frames) > 5:
                print(f"  ... and {len(missing_frames) - 5} more")

        if frame_count == 0:
            print(f"Error: No {image_name} files found!")
            return False

        print(f"Found {frame_count} frames")

        # Set quality parameters
        if quality == "high":
            crf = 18  # Lower = better quality (18-23 is good)
            preset = "slow"
        elif quality == "medium":
            crf = 23
            preset = "medium"
        else:  # low
            crf = 28
            preset = "fast"

        # Create video with ffmpeg
        print(f"\nCreating video at {fps} FPS...")
        print(f"Quality: {quality} (crf={crf})")
        print(f"Output: {output_file}")

        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', str(temp_dir / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-pix_fmt', 'yuv420p',  # For compatibility
            '-y',  # Overwrite output file
            str(output_file)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"Error running ffmpeg:")
            print(result.stderr)
            return False

        print(f"✅ Video created successfully: {output_file}")

        # Show file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        print(f"   Frames: {frame_count}")
        print(f"   Duration: {frame_count/fps:.2f} seconds")

        return True

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary files")


def process_directory(base_dir: Path, args):
    """Process a single directory to create video."""
    print(f"\n{'='*80}")
    print(f"Processing: {base_dir}")
    print(f"{'='*80}")

    # Determine output filename
    if args.output:
        output_file = Path(args.output)
    else:
        # Auto-generate output name based on directory
        output_name = f"{base_dir.name}_animation.mp4"
        output_file = base_dir / output_name

    # Create video
    success = create_video(
        base_dir,
        output_file,
        fps=args.fps,
        image_name=args.image,
        quality=args.quality
    )

    return success


def find_all_episode_parent_dirs(root_dir: Path) -> list:
    """
    Recursively find all directories that contain episode subdirectories.

    Returns:
        List of directories containing episode folders
    """
    parent_dirs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)

        # Check if this directory has episode subdirectories
        has_episodes = any(re.match(r'ep\d+', d) for d in dirnames)

        if has_episodes:
            parent_dirs.append(dirpath)

    return parent_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Create videos from episode render sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from bob/sphere directory
  python create_video.py output/bob/sphere/

  # Create video at 10 FPS
  python create_video.py output/bob/sphere/ --fps 10

  # Create video from alpha channel
  python create_video.py output/bob/sphere/ --image alpha.png

  # Process all subdirectories recursively
  python create_video.py output/ --recursive

  # High quality video (larger file)
  python create_video.py output/bob/sphere/ --quality high

Available images:
  render.png - Main render
  alpha.png  - Alpha channel
  depth.png  - Depth map
  normal.png - Normal map
        """
    )

    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing episode folders (ep000, ep001, ...)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output video file path (default: auto-generated)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Frames per second (default: 5)'
    )

    parser.add_argument(
        '--image',
        type=str,
        default='render.png',
        choices=['render.png', 'alpha.png', 'depth.png', 'normal.png'],
        help='Image type to use (default: render.png)'
    )

    parser.add_argument(
        '--quality',
        type=str,
        default='high',
        choices=['high', 'medium', 'low'],
        help='Video quality (default: high)'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process all subdirectories containing episodes'
    )

    args = parser.parse_args()

    # Check ffmpeg
    if not check_ffmpeg():
        return 1

    base_dir = Path(args.directory)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    if not base_dir.is_dir():
        print(f"Error: Not a directory: {base_dir}")
        return 1

    # Process directories
    if args.recursive:
        # Find all directories with episodes
        dirs_to_process = find_all_episode_parent_dirs(base_dir)

        if not dirs_to_process:
            print(f"Error: No episode directories found in {base_dir}")
            return 1

        print(f"Found {len(dirs_to_process)} directories with episodes:")
        for d in dirs_to_process:
            print(f"  - {d.relative_to(base_dir) if d != base_dir else d.name}")

        print()

        # Process each directory
        successes = 0
        for d in dirs_to_process:
            # Override output for recursive mode
            args.output = None
            if process_directory(d, args):
                successes += 1

        print(f"\n{'='*80}")
        print(f"Summary: {successes}/{len(dirs_to_process)} videos created successfully")
        print(f"{'='*80}")

        return 0 if successes == len(dirs_to_process) else 1
    else:
        # Process single directory
        success = process_directory(base_dir, args)
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
