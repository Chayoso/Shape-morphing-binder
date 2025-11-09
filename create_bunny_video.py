#!/usr/bin/env python3
"""
Create training progression video for bunny experiment.
Shows render, alpha, and depth evolution across episodes.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def create_training_video(
    output_dir: str,
    output_video: str = "bunny_training.mp4",
    fps: int = 2,
    image_type: str = "render"
):
    """
    Create video from episode images.

    Args:
        output_dir: Directory containing ep### folders
        output_video: Output video filename
        fps: Frames per second
        image_type: Type of image to use (render, alpha, depth, axis_hist)
    """
    output_path = Path(output_dir)

    # Find all episode directories
    episode_dirs = sorted(output_path.glob("ep*"))
    episode_dirs = [d for d in episode_dirs if d.is_dir()]

    print(f"Found {len(episode_dirs)} episodes")

    # Collect image paths
    image_paths = []
    for ep_dir in episode_dirs:
        ep_num = int(ep_dir.name.replace("ep", ""))
        img_path = ep_dir / f"{ep_dir.name}_{image_type}.png"

        if img_path.exists():
            image_paths.append((ep_num, img_path))
        else:
            print(f"Warning: Missing {image_type} image for {ep_dir.name}")

    if not image_paths:
        print(f"No {image_type} images found!")
        return

    # Sort by episode number
    image_paths.sort(key=lambda x: x[0])

    print(f"Creating video from {len(image_paths)} images...")

    # Read first image to get dimensions
    first_img = cv2.imread(str(image_paths[0][1]))
    if first_img is None:
        print(f"Failed to read first image: {image_paths[0][1]}")
        return

    height, width = first_img.shape[:2]

    # Create video writer
    video_path = output_path / output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Write frames
    for ep_num, img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        # Add episode number text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Episode {ep_num}"
        font_scale = 2.0
        thickness = 4

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (30 + text_width, 30 + text_height + baseline),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # Draw text
        cv2.putText(
            img,
            text,
            (20, 30 + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        out.write(img)
        print(f"  Added frame for episode {ep_num}")

    out.release()
    print(f"\n✅ Video saved to: {video_path}")


def create_comparison_video(
    output_dir: str,
    output_video: str = "bunny_comparison.mp4",
    fps: int = 2
):
    """
    Create side-by-side comparison video (render + alpha + depth).
    """
    output_path = Path(output_dir)

    # Find all episode directories
    episode_dirs = sorted(output_path.glob("ep*"))
    episode_dirs = [d for d in episode_dirs if d.is_dir()]

    print(f"Found {len(episode_dirs)} episodes")

    # Collect image paths
    frames = []
    for ep_dir in episode_dirs:
        ep_num = int(ep_dir.name.replace("ep", ""))

        render_path = ep_dir / f"{ep_dir.name}_render.png"
        alpha_path = ep_dir / f"{ep_dir.name}_alpha.png"
        depth_path = ep_dir / f"{ep_dir.name}_depth.png"

        if render_path.exists() and alpha_path.exists() and depth_path.exists():
            frames.append((ep_num, render_path, alpha_path, depth_path))

    if not frames:
        print("No complete image sets found!")
        return

    frames.sort(key=lambda x: x[0])

    print(f"Creating comparison video from {len(frames)} frames...")

    # Read first images to get dimensions
    first_render = cv2.imread(str(frames[0][1]))
    if first_render is None:
        print(f"Failed to read first render image")
        return

    h, w = first_render.shape[:2]

    # Scale down if too large for video codec
    max_width = 1920
    if w > max_width:
        scale = max_width / w
        w = int(w * scale)
        h = int(h * scale)
        print(f"Scaling images to {w}x{h} for video codec compatibility")

    # Create 1x3 layout (render | alpha | depth)
    combined_width = w * 3
    combined_height = h

    # Create video writer
    video_path = output_path / output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (combined_width, combined_height))

    # Write frames
    for ep_num, render_path, alpha_path, depth_path in frames:
        render = cv2.imread(str(render_path))
        alpha = cv2.imread(str(alpha_path))
        depth = cv2.imread(str(depth_path))

        if render is None or alpha is None or depth is None:
            print(f"Failed to read images for episode {ep_num}")
            continue

        # Resize if needed
        if render.shape[1] != w or render.shape[0] != h:
            render = cv2.resize(render, (w, h))
            alpha = cv2.resize(alpha, (w, h))
            depth = cv2.resize(depth, (w, h))

        # Combine horizontally
        combined = np.hstack([render, alpha, depth])

        # Add episode number
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Episode {ep_num}"
        cv2.putText(
            combined,
            text,
            (20, 50),
            font,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        # Add labels
        cv2.putText(combined, "Render", (w//2 - 50, h - 20), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "Alpha", (w + w//2 - 40, h - 20), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, "Depth", (2*w + w//2 - 40, h - 20), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(combined)
        print(f"  Added comparison frame for episode {ep_num}")

    out.release()
    print(f"\n✅ Comparison video saved to: {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training progression videos")
    parser.add_argument("output_dir", type=str, help="Output directory (e.g., output/bunny)")
    parser.add_argument("--type", type=str, default="comparison",
                       choices=["render", "alpha", "depth", "axis_hist", "comparison"],
                       help="Type of video to create")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")

    args = parser.parse_args()

    if args.type == "comparison":
        create_comparison_video(args.output_dir, fps=args.fps)
    else:
        output_name = f"bunny_{args.type}.mp4"
        create_training_video(args.output_dir, output_name, fps=args.fps, image_type=args.type)
