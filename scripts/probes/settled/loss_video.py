"""loss_video.py ARCHIVE.npz RUN.json GRAD_DUMP_DIR OUT.mp4 — the morph coloured by how wrong each particle
still is and by how hard each channel pulls on it, red (large) to green (small), with the window losses below.

Panels: (1) shape error, the distance of each particle to the nearest target point in target spacings (green at
0.5, red at 3 and beyond); (2) the render channel's pull, |lambda (g_sil + w_pbr g_pbr)| on each particle's end
position at the window start (the gradient dump), log scale, red at the first window's 99th percentile, green two
decades below; (3) the physics channel's pull |g_phys|, the same scale rule. The gradients of window k colour the
frames of window k. Bottom: the selection merit, the silhouette loss and the transport energy per accepted window,
each relative to its first value, with the current window marked. Particles are isotropic Gaussians shaded by
their density normals. Needs a run made with --grad_dump.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import argparse
import json
import math
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from physmorph import gpu
from physmorph.render.knn_gpu import knn_self_torch
from physmorph.render.photoreal import render_3dgs_torch
from physmorph.sampling.orientation import orient_archive

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("json"); ap.add_argument("dumps"); ap.add_argument("out")
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--panel", type=int, default=640)
ap.add_argument("--azimuth", type=float, default=35.0); ap.add_argument("--elevation", type=float, default=18.0)
ap.add_argument("--w_pbr", type=float, default=1.0); ap.add_argument("--T", type=int, default=20)
a = ap.parse_args()
dev = "cuda"
z = np.load(a.npz, allow_pickle=True)
frames, tgt, _, _ = orient_archive(z, a.npz)
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
hist = json.load(open(a.json))["arms"]["render_full_dt_iso_nn"]["history"]
runs = [h for h in hist if "c2f_render_res" not in h]            # one record per window, in dump order
wins = [i for i, h in enumerate(runs) if h.get("frame_end") and not h.get("null_commit")]
dumps = sorted(f for f in os.listdir(a.dumps) if f.startswith("win_"))
steps = 2 * a.T


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT, SMALL = font(22), font(17)
tgt_t = gpu.tensor(np.asarray(tgt, np.float32))
tree = gpu.KNN(tgt_t)
sp_t = gpu.median(tree.query(tgt_t, 2)[0][:, 1])


def ramp(v):
    """0 -> green, 0.5 -> yellow, 1 -> red (clipped)."""
    v = v.clamp(0, 1)[:, None]
    green, yellow, red = (torch.tensor(c, device=dev) for c in ((0.16, 0.66, 0.30), (0.95, 0.80, 0.18), (0.86, 0.18, 0.15)))
    return torch.where(v < 0.5, green + (yellow - green) * (2 * v), yellow + (red - yellow) * (2 * v - 1))


def grad_mags(k):
    d = np.load(os.path.join(a.dumps, dumps[k]))
    lam = float(d["lam_r"])
    render = lam * (torch.as_tensor(d["gx_sil"], device=dev) + a.w_pbr * torch.as_tensor(d["gx_pbr"], device=dev))
    return render.norm(dim=1), torch.as_tensor(d["gx_phys"], device=dev).norm(dim=1), lam, float(d["g_share"])


# the colour scale of the gradient panels: the first window's 99th percentiles (fixed for the whole video)
r0, p0, _, _ = grad_mags(wins[0])
R_ref = float(torch.quantile(r0[r0 > 0][:1_000_000].float(), 0.99)); P_ref = float(torch.quantile(p0[p0 > 0][:1_000_000].float(), 0.99))
curves = {k: np.array([runs[i][k] for i in wins], np.float64) for k in ("selection_merit", "d_sil", "transport_energy")}
curves = {k: v / v[0] for k, v in curves.items()}


def curve_strip(j, W, H):
    """The loss curves (log scale) with window j marked, as an RGB array."""
    im = Image.new("RGB", (W, H), (250, 250, 247)); dr = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 90, W - 40, 40, H - 50
    lo = min(v.min() for v in curves.values()); lo = 10 ** math.floor(math.log10(max(lo, 1e-6)))
    def px(i, v):
        return (x0 + (x1 - x0) * i / max(len(wins) - 1, 1),
                y0 + (y1 - y0) * (0 - math.log10(max(v, lo))) / (0 - math.log10(lo)))
    for e in range(0, int(-math.log10(lo)) + 1):
        y = y0 + (y1 - y0) * e / (-math.log10(lo)); dr.line([(x0, y), (x1, y)], fill=(222, 222, 216))
        dr.text((20, y - 9), f"1e-{e}" if e else "1", fill=(90, 90, 90), font=SMALL)
    cols = {"selection_merit": (40, 40, 40), "d_sil": (200, 60, 50), "transport_energy": (40, 110, 190)}
    names = {"selection_merit": "selection merit", "d_sil": "silhouette loss", "transport_energy": "transport energy"}
    for n, (k, v) in enumerate(curves.items()):
        pts = [px(i, v[i]) for i in range(len(v))]
        dr.line(pts, fill=cols[k], width=3)
        dr.ellipse([pts[j][0] - 6, pts[j][1] - 6, pts[j][0] + 6, pts[j][1] + 6], fill=cols[k])
        dr.text((x0 + 10 + 420 * n, 8), names[k] + " (relative to window 1)", fill=cols[k], font=SMALL)
    cx = px(j, 1)[0]; dr.line([(cx, y0), (cx, y1)], fill=(120, 120, 120), width=1)
    dr.text((x0, y1 + 12), "window 1", fill=(90, 90, 90), font=SMALL); dr.text((x1 - 100, y1 + 12), f"window {len(wins)}", fill=(90, 90, 90), font=SMALL)
    return np.asarray(im)


light = torch.nn.functional.normalize(torch.tensor([0.4, 0.8, 0.45], device=dev), dim=0)
center = tgt_t.mean(0); rad = float((tgt_t - center).norm(dim=1).max())
tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
idx = list(range(0, dn, a.stride)) + [dn - 1]
cache = {}
for kf, f in enumerate(idx):
    j = 0 if f == 0 else min((f - 1) // steps, len(wins) - 1)            # the accepted window of this frame
    if j not in cache:
        cache.clear(); cache[j] = grad_mags(wins[j])
    rmag, pmag, lam, share = cache[j]
    x = torch.as_tensor(np.asarray(frames[f], np.float32), device=dev)
    err = tree.query(x, 1)[0][:, 0].float() / sp_t
    d8, nb = knn_self_torch(x, 17)
    nrm = torch.nn.functional.normalize(x - x[nb[:, 1:]].mean(1), dim=1)
    sig = 0.8 * float(d8[:, 1].median())
    values = [(err - 0.5) / 2.5, 1 + torch.log10(rmag / R_ref + 1e-12) / 2, 1 + torch.log10(pmag / P_ref + 1e-12) / 2]
    kw = dict(sigma0=sig, opacity=0.95, azimuth=math.radians(a.azimuth), elevation=math.radians(a.elevation),
              dist=rad * 3.6, res=a.panel, fovy_deg=30.0, center=center, bg=(0, 0, 0))
    N_img = render_3dgs_torch(x, 0.5 * (nrm + 1), **kw)
    A_img = render_3dgs_torch(x, torch.ones_like(nrm), **kw)[..., :1].clamp(0, 1)
    npx = torch.nn.functional.normalize(N_img * 2 - A_img, dim=-1)
    shade = 0.45 + 0.55 * (npx @ light).clamp(0, 1)[..., None]
    panels = []
    for v in values:
        C_img = render_3dgs_torch(x, ramp(v), **kw)
        panels.append(C_img / A_img.clamp_min(1e-6) * shade * A_img + 0.97 * (1 - A_img))
    img = (torch.cat(panels, 1).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    W = img.shape[1]
    canvas = np.full((a.panel + 60 + 380, W, 3), 250, np.uint8)
    canvas[60:60 + a.panel] = img
    canvas[60 + a.panel:] = curve_strip(j, W, 380)
    im = Image.fromarray(canvas); dr = ImageDraw.Draw(im)
    for p, t in enumerate(("shape error (distance to target)", "render pull  |lambda g_render|", "physics pull  |g_physics|")):
        dr.text((p * a.panel + 16, 6), t, fill=(30, 30, 30), font=FONT)
    for p, t in enumerate(("green 0.5 sp  ->  red 3 sp", "log; red = window-1 p99, green = 1/100", "log; red = window-1 p99, green = 1/100")):
        dr.text((p * a.panel + 16, 34), t, fill=(110, 110, 110), font=SMALL)
    dr.text((16, 64 + a.panel - 30), f"frame {f}/{dn - 1}   window {j + 1}/{len(wins)}   lambda {lam:.3g}   render share {share:.2f}"
            f"   median error {float(err.median()):.2f} sp   >1.5 sp: {100 * float((err > 1.5).float().mean()):.1f} %",
            fill=(60, 60, 60), font=SMALL)
    im.save(os.path.join(tmp, f"f{kf:04d}.png"))
    if kf % 25 == 0:
        print(f"frame {kf + 1}/{len(idx)}", flush=True)
for h in range(20):
    im.save(os.path.join(tmp, f"f{len(idx) + h:04d}.png"))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "20", "-i", os.path.join(tmp, "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], check=True)
print("saved", a.out)
