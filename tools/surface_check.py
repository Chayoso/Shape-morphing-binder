import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io  # Required for in-memory byte streams
import re  # Required for regular expression matching for the filename

def load_ply_points_ascii(path: Path) -> np.ndarray:
    """
    Loads vertex points from an ASCII PLY file.
    It parses the header to find the number of vertices and the start of the data.
    """
    with path.open("r") as f:
        lines = f.readlines()
        
    end_idx = None
    num_vertices = None
    
    # Parse the header to find where the vertex data starts
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            try:
                num_vertices = int(line.strip().split()[-1])
            except (ValueError, IndexError):
                num_vertices = None
        if line.strip() == "end_header":
            end_idx = i
            break
            
    if end_idx is None:
        raise ValueError("PLY file header is malformed or 'end_header' not found.")
        
    pts = []
    # Read the vertex data lines
    data_lines = lines[end_idx + 1 : end_idx + 1 + (num_vertices if num_vertices else len(lines))]
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x, y, z = map(float, parts[:3])
                pts.append((x, y, z))
            except ValueError:
                # Skip lines that cannot be converted to float triplets
                continue
                
    return np.array(pts, dtype=np.float64)

def voxelize(points: np.ndarray, res: int = 128) -> np.ndarray:
    """
    Converts a point cloud into a binary voxel grid.
    Points are normalized and scaled to fit within the specified grid resolution.
    """
    # Find the bounding box of the point cloud
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9) # Avoid division by zero
    
    # Normalize coordinates and scale to grid resolution
    coords = ((points - mins) / extent * (res - 1)).astype(int)
    
    # Clamp coordinates to be within the grid bounds
    coords = np.clip(coords, 0, res - 1)
    
    # Create an empty grid and mark the occupied voxels
    grid = np.zeros((res, res, res), dtype=np.uint8)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    
    return grid

def plot_to_image(img2d: np.ndarray, title: str, xlabel: str, ylabel: str) -> Image.Image:
    """
    Plots a 2D numpy array using Matplotlib and returns it as a PIL Image object
    without saving it to a file on disk.
    """
    # --- MODIFIED PART: Font sizes are reduced ---
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })

    plt.figure(figsize=(5.0, 4.6))
    plt.imshow(img2d.T, origin="lower", aspect="equal")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Occupied voxels (count)")
    plt.tight_layout()
    
    # Save the plot to an in-memory buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=180, bbox_inches="tight")
    plt.close()  # Close the figure to free up memory
    buf.seek(0)  # Rewind the buffer to the beginning
    
    # Create a PIL Image object from the buffer's content
    pil_img = Image.open(buf).convert("RGB")
    
    return pil_img

def stitch_grid(images: list, out_path: Path, cols: int = 3, rows: int = 2, pad: int = 10, bg: tuple = (255,255,255)):
    """
    Stitches a list of PIL Image objects into a single grid image.
    The input is a list of images, not file paths.
    """
    if not images:
        return
    
    # Ensure all images are consistently sized for a clean grid
    w = min(im.width for im in images)
    h = min(im.height for im in images)
    images = [im.resize((w, h), Image.Resampling.LANCZOS) for im in images[:rows*cols]]
    
    # Calculate the total dimensions of the canvas
    W = cols * w + (cols - 1) * pad
    H = rows * h + (rows - 1) * pad
    
    # Create a new blank canvas
    canvas = Image.new("RGB", (W, H), bg)
    
    # Paste each image into its correct position on the canvas
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * (w + pad)
        y = r * (h + pad)
        canvas.paste(im, (x, y))
        
    canvas.save(out_path, optimize=True, quality=92)

def process_file(ply_path: Path, out_dir: Path, res: int, thick: int):
    """
    Main processing function for a single PLY file.
    It loads, voxelizes, generates views, and creates a final stitched panel.
    """
    pts = load_ply_points_ascii(ply_path)
    if pts.size == 0:
        print(f"[WARN] {ply_path.name} contains no points.")
        return 0.0, None
        
    grid = voxelize(pts, res=res)
    occ = float(grid.mean())

    # Generate Maximum Intensity Projections (MIPs)
    mip_x = grid.sum(axis=0)  # Projection onto YZ-plane
    mip_y = grid.sum(axis=1)  # Projection onto XZ-plane
    mip_z = grid.sum(axis=2)  # Projection onto XY-plane
    
    # Generate thick slices from the center of the grid
    c = res // 2
    slc_x = grid[c - thick : c + thick + 1, :, :].sum(axis=0)
    slc_y = grid[:, c - thick : c + thick + 1, :].sum(axis=1)
    slc_z = grid[:, :, c - thick : c + thick + 1].sum(axis=2)

    base = ply_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate all 6 images in-memory and store them in a list
    # The title passed to the plot function is also simplified
    images = [
        plot_to_image(mip_x, "MIP along X", "Y index", "Z index"),
        plot_to_image(mip_y, "MIP along Y", "X index", "Z index"),
        plot_to_image(mip_z, "MIP along Z", "X index", "Y index"),
        plot_to_image(slc_x, f"Slice around X={c}", "Y index", "Z index"),
        plot_to_image(slc_y, f"Slice around Y={c}", "X index", "Z index"),
        plot_to_image(slc_z, f"Slice around Z={c}", "X index", "Y index"),
    ]

    # --- MODIFIED PART: New filename logic ---
    # Try to find a pattern like 'ep123' in the input filename
    match = re.search(r'ep\d+', base)
    if match:
        episode_str = match.group(0)  # e.g., 'ep001'
        panel_path = out_dir / f"{episode_str}_surface_check.png"
    else:
        # If the pattern is not found, use the original name as a fallback
        panel_path = out_dir / f"{base}_surface_check.png"

    # Stitch the in-memory images into a final panel file
    stitch_grid(images, panel_path, cols=3, rows=2, pad=8)
    
    return occ, str(panel_path)

def main():
    """
    Main entry point of the script.
    Parses arguments, finds PLY files, and processes each one.
    """
    ap = argparse.ArgumentParser(description="Voxelize PLY files and generate projection/slice panels.")
    ap.add_argument("--input_dir", type=str, default=".", help="Directory containing .ply files.")
    ap.add_argument("--output_dir", type=str, default="./output", help="Directory to save the output panels.")
    ap.add_argument("--res", type=int, default=128, help="Resolution of the voxel grid.")
    ap.add_argument("--thick", type=int, default=2, help="Half-thickness of the central slices.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ply_files = sorted(in_dir.rglob("*.ply"))

    if not ply_files:
        print(f"No .ply files found in {in_dir}")
        return

    print(f"Found {len(ply_files)} PLY files in {in_dir}")
    for p in ply_files:
        try:
            occ, panel = process_file(p, out_dir, args.res, args.thick)
            if panel:
                print(f"[OK] {p.name}  occupancy={occ:.6f}  panel={panel}")
        except Exception as e:
            print(f"[ERR] Failed to process {p.name}: {e}")

if __name__ == "__main__":
    main()