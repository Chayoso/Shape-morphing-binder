"""d116_altmap.py MP4 OUT_PNG [TAIL_FRAC=0.3] -- where the tail flickers: per pixel the mean over the last TAIL_FRAC
frames of the alternating part |I+1 - 2I + I-1| / 2 (half resolution), drawn beside the last frame, and the three
hottest 160-px squares cropped from the last three frames (one row each, oldest left)."""
import json, subprocess, sys, numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
path, out = sys.argv[1], sys.argv[2]; tail = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
w, h = int(info["width"]) // 2, int(info["height"]) // 2
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
g = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0
k0 = int(len(g) * (1 - tail))
alt = np.mean([np.abs(g[t + 1] - 2 * g[t] + g[t - 1]) / 2 for t in range(max(1, k0), len(g) - 1)], axis=0)
heat = (np.clip(alt / max(np.percentile(alt, 99.9), 1e-6), 0, 1) * 255).astype(np.uint8)
sm = uniform_filter(alt, 40); spots = []
for _ in range(3):
    y, x = np.unravel_index(np.argmax(sm), sm.shape); spots.append((int(y), int(x)))
    sm[max(0, y - 120):y + 120, max(0, x - 120):x + 120] = 0
rows = []
for (y, x) in spots:
    y0, x0 = min(max(0, y - 80), h - 160), min(max(0, x - 80), w - 160)
    rows.append(np.concatenate([g[t][y0:y0 + 160, x0:x0 + 160] for t in (len(g) - 7, len(g) - 5, len(g) - 3, len(g) - 1)], 1))
crops = np.concatenate(rows, 0)
crops = np.kron(crops, np.ones((2, 2)))
left = np.concatenate([g[-1], heat / 255.0], 0)
H = max(left.shape[0], crops.shape[0])
pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, 0)), constant_values=1.0)
Image.fromarray((np.concatenate([pad(left), pad(crops)], 1) * 255).astype(np.uint8)).save(out)
print(json.dumps(dict(spots=spots, alt_body_p50=float(np.median(alt[alt > 1e-4])), alt_p99=float(np.percentile(alt, 99)))))
