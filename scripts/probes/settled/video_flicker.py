"""Alternation vs drift in a plain video (the perceptual discriminator; Kelly 1979: a sign flip every
frame is a Nyquist-frequency modulation inside the eye's flicker peak, a slow drift of the same size
is not). Frames decoded by ffmpeg to grey (0..1); per frame t: the first difference D1 = mean|I_t -
I_{t-1}| (the existing tail measure), the ALTERNATING component A = mean|I_{t+1} - 2 I_t + I_{t-1}| / 2
(pure +-flicker of amplitude a gives A = a, drift gives 0) and the DRIFT component S = mean|I_{t+1} -
I_{t-1}| / 2 (drift a per frame gives S = a, pure flicker 0). Reported over the whole delivered range
and its last TAIL_FRAC, with the ratio A / S. usage: video_flicker.py MP4 [TAIL_FRAC=0.2] [HOLD=20]"""
import json, subprocess, sys, numpy as np
path = sys.argv[1]; tail = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2; hold = int(sys.argv[3]) if len(sys.argv) > 3 else 20
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
w, h = int(info["width"]), int(info["height"])
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
g = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0
n = len(g) - hold; g = g[:n]
d1 = np.abs(np.diff(g, axis=0)).mean((1, 2))
alt = (np.abs(g[2:] - 2 * g[1:-1] + g[:-2]) / 2).mean((1, 2))
drf = (np.abs(g[2:] - g[:-2]) / 2).mean((1, 2))
k = max(3, int(round(tail * n)))
def s(v): return "%.4f" % float(np.median(v))
print("%s: %d delivered frames; whole: D1 %s  ALT %s  DRIFT %s  (ALT/DRIFT %.2f) | tail(last %d): D1 %s  ALT %s  DRIFT %s  (ALT/DRIFT %.2f)" % (
    path.split("/")[-1], n, s(d1), s(alt), s(drf), float(np.median(alt) / max(np.median(drf), 1e-9)), k, s(d1[-k:]), s(alt[-k:]), s(drf[-k:]), float(np.median(alt[-k:]) / max(np.median(drf[-k:]), 1e-9))))
