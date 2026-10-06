"""d116_flicker.py MP4 [TAIL_FRAC=0.3] -- video_flicker.py on the body only (pixels that differ from the background,
read on the first frame's left edge, by more than 0.03), six digits, at half resolution: per frame t the alternating
part A = |I+1 - 2I + I-1|/2 and the drift S = |I+1 - I-1|/2, medians over the body, whole run, first fifth, last TAIL_FRAC."""
import json, subprocess, sys, numpy as np
path = sys.argv[1]; tail = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
w, h = int(info["width"]) // 2, int(info["height"]) // 2
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
g = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0
bg = np.median(g[0][:, :8], axis=1)[:, None]
A, S = [], []
for t in range(1, len(g) - 1):
    body = np.abs(g[t] - bg) > 0.03
    if body.sum() < 100:
        continue
    A.append(float((np.abs(g[t + 1] - 2 * g[t] + g[t - 1]) / 2)[body].mean()))
    S.append(float((np.abs(g[t + 1] - g[t - 1]) / 2)[body].mean()))
A, S = np.array(A), np.array(S)
k = max(3, int(round(tail * len(A))))
f = max(3, len(A) // 5)
print("%s: frames %d | whole ALT %.6f DRIFT %.6f (%.2f) | first fifth ALT %.6f DRIFT %.6f (%.2f) | tail(last %d) ALT %.6f DRIFT %.6f (%.2f)" % (
    path.split("/")[-1], len(g), np.median(A), np.median(S), np.median(A) / max(np.median(S), 1e-12),
    np.median(A[:f]), np.median(S[:f]), np.median(A[:f]) / max(np.median(S[:f]), 1e-12),
    k, np.median(A[-k:]), np.median(S[-k:]), np.median(A[-k:]) / max(np.median(S[-k:]), 1e-12)))
