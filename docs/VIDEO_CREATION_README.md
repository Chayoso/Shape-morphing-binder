# Video Creation from Episode Renders

The `create_video.py` script creates MP4 videos from episode render sequences.

## Quick Start

### Create video from a specific directory
```bash
python create_video.py output/bob/sphere/
```
**Output:** `output/bob/sphere/sphere_animation.mp4`

### Create videos for all subdirectories
```bash
python create_video.py output/ --recursive
```
This will find all directories with episodes and create a video for each.

## Options

### Framerate (FPS)
```bash
# Slower animation (5 FPS - default)
python create_video.py output/bob/sphere/ --fps 5

# Faster animation (10 FPS)
python create_video.py output/bob/sphere/ --fps 10

# Very fast animation (30 FPS)
python create_video.py output/bob/sphere/ --fps 30
```

### Image Type
```bash
# Main render (default)
python create_video.py output/bob/sphere/ --image render.png

# Alpha channel
python create_video.py output/bob/sphere/ --image alpha.png

# Depth map
python create_video.py output/bob/sphere/ --image depth.png

# Normal map
python create_video.py output/bob/sphere/ --image normal.png
```

### Video Quality
```bash
# High quality (larger file, default)
python create_video.py output/bob/sphere/ --quality high

# Medium quality
python create_video.py output/bob/sphere/ --quality medium

# Low quality (smaller file)
python create_video.py output/bob/sphere/ --quality low
```

### Custom Output Name
```bash
python create_video.py output/bob/sphere/ --output my_morphing.mp4
```

## Examples

### Create a side-by-side comparison video
```bash
# Create render video
python create_video.py output/bob/sphere/ --output render.mp4 --fps 10

# Create depth video
python create_video.py output/bob/sphere/ --image depth.png --output depth.mp4 --fps 10

# Combine with ffmpeg (requires ffmpeg-concat or similar)
```

### Create high-quality slow-motion video
```bash
python create_video.py output/bob/sphere/ --fps 3 --quality high --output slowmo.mp4
```

### Batch process all experiments
```bash
python create_video.py output/ --recursive --fps 10 --quality medium
```

## Output Details

The script will:
1. Find all episode directories (ep000, ep001, ep002, ...)
2. Extract the specified image from each episode
3. Create a video with frames in sequential order
4. Report any missing frames

Example output:
```
================================================================================
Processing: output/bob/sphere
================================================================================
Found 50 episode directories
First episode: ep000
Last episode: ep049

Creating symlinks for render.png files...
Found 32 frames

Creating video at 10 FPS...
Quality: high (crf=18)
Output: output/bob/sphere/sphere_animation.mp4
✅ Video created successfully: output/bob/sphere/sphere_animation.mp4
   File size: 0.03 MB
   Frames: 32
   Duration: 3.20 seconds
```

## Troubleshooting

### Missing ffmpeg
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Check installation
ffmpeg -version
```

### Missing frames warning
If you see warnings about missing frames, some episodes didn't save PNG files. This usually means:
- The `--png` flag wasn't used during training
- Some episodes failed during training
- PNG export was disabled for certain episodes

The video will still be created using available frames.

### Video not playing
If the video doesn't play in your media player, try:
- Using VLC media player (most compatible)
- Converting to a different format: `ffmpeg -i video.mp4 -c:v libx264 video_compat.mp4`

## Advanced Usage

### Create GIF instead of MP4
```bash
# First create MP4
python create_video.py output/bob/sphere/ --output temp.mp4

# Convert to GIF
ffmpeg -i temp.mp4 -vf "fps=10,scale=640:-1:flags=lanczos" output.gif
```

### Extract specific frame range
Modify the script or use ffmpeg directly:
```bash
# Extract frames 10-30
ffmpeg -i sphere_animation.mp4 -vf "select='between(n\,10\,30)'" output_%04d.png
```

### Create comparison video (multiple directories)
```bash
# Create individual videos
python create_video.py output/bob/sphere/ --output bob_sphere.mp4
python create_video.py output/spot/sphere/ --output spot_sphere.mp4

# Stack horizontally with ffmpeg
ffmpeg -i bob_sphere.mp4 -i spot_sphere.mp4 -filter_complex hstack comparison.mp4
```

## File Formats

- **Input:** PNG images (render.png, alpha.png, depth.png, normal.png)
- **Output:** MP4 video (H.264 codec, yuv420p pixel format)
- **Compatibility:** Works on all major platforms and media players

## Performance

- High quality (crf=18): ~30-50KB per second of video
- Medium quality (crf=23): ~20-30KB per second of video
- Low quality (crf=28): ~10-20KB per second of video

For 50 frames at 10 FPS (5 seconds):
- High: ~150-250KB
- Medium: ~100-150KB
- Low: ~50-100KB
