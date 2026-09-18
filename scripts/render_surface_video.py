"""Surface video of a morph: the OBJECT changing, not particles. GPU (torch).

Per frame: orthographic projection, per-particle disk splats sized by the local spacing
(z-buffer via scatter_reduce amin), coverage-normalised depth smoothing, normals from the
depth field, Cook-Torrance/GGX shading with three lights + hemispherical ambient + a depth
AO term — the same model as scripts/render_pbr.py, vectorised. Two azimuths side by side,
target silhouette drawn as a faint outline, frame counter. Output: GIF (PIL) and optionally
an MP4 (imageio-ffmpeg if present). Raw archived state is the only input.

Usage: python render_surface_video.py --npz run.npz --out morph.gif [--res 480] [--stride 4]
       [--views 35,215] [--elev 18] [--fps 20] [--hold 16] [--label "..."]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=480)
ap.add_argument("--stride", type=int, default=4, help="archived frames per video frame")
ap.add_argument("--views", default="35,215", help="azimuths in degrees")
ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--hold", type=int, default=16)
ap.add_argument("--label", default="")
ap.add_argument("--splat_k", type=float, default=1.35)
ap.add_argument("--smooth", type=float, default=0.7)
ap.add_argument("--albedo", default="0.80,0.78,0.74")
ap.add_argument("--rough", type=float, default=0.45)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--mp4", default="", help="also write an mp4 (needs imageio-ffmpeg)")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(a.npz)
import sys as _sys, os as _os  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from physmorph.sampling.orientation import orient_archive  # noqa: E402
frames_np, tgt_np, _sr, _orient = orient_archive(d, a.npz)     # y-up (assets/orientation.json)
dn = int(d["deliver_n"]) if "deliver_n" in d.files else len(frames_np)
idx = list(range(0, dn, a.stride))
if idx[-1] != dn - 1:
    idx.append(dn - 1)
if a.max_frames > 0:
    idx = idx[: a.max_frames]
tgt = torch.as_tensor(tgt_np, device=dev, dtype=torch.float32)
ctr = 0.5 * (tgt.max(0).values + tgt.min(0).values)
RES = a.res
albedo = torch.tensor([float(c) for c in a.albedo.split(",")], device=dev)


def camera(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = torch.tensor([-math.cos(el) * math.sin(az), -math.sin(el), -math.cos(el) * math.cos(az)], device=dev)
    fwd = fwd / fwd.norm()
    right = torch.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=dev), dim=0); right = right / right.norm()
    up = torch.cross(right, fwd, dim=0)
    return fwd, right, up


views = [camera(float(v), a.elev) for v in a.views.split(",")]
# a common extent from the target in every view
ext = max(float((((tgt - ctr) @ torch.stack([r, u]).T).abs()).max()) for _, r, u in views) * 1.15
pix_per_wu = RES / (2 * ext)

# disk offsets up to the max radius (px); per-particle radius masks them
RMAX = 12
oy, ox = torch.meshgrid(torch.arange(-RMAX, RMAX + 1, device=dev), torch.arange(-RMAX, RMAX + 1, device=dev), indexing="ij")
ox, oy = ox.reshape(-1).float(), oy.reshape(-1).float()
odist2 = ox * ox + oy * oy
K = len(ox)


def local_spacing(x, n_sub=24000, k=8, chunk=4096):
    """Per-particle local spacing (N,): mean distance to the k nearest points of a random
    subset, rescaled by (n_sub/N)^(1/3) to the full cloud's spacing. Chunked cdist on GPU."""
    n = x.shape[0]
    sub = x[torch.randperm(n, device=dev)[: min(n, n_sub)]]
    scale = (sub.shape[0] / n) ** (1.0 / 3.0)
    out = torch.empty(n, device=dev)
    for s in range(0, n, chunk):
        dd = torch.cdist(x[s:s + chunk], sub)
        out[s:s + chunk] = dd.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out * scale


gy1 = torch.tensor([1.0, 0.0, -1.0], device=dev).view(1, 1, 3, 1) / 2
gx1 = torch.tensor([1.0, 0.0, -1.0], device=dev).view(1, 1, 1, 3) / 2


def gauss_blur(img, sig):
    r = int(3 * sig)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sig * sig)); k = k / k.sum()
    img = Fn.conv2d(img, k.view(1, 1, 1, -1), padding=(0, r))
    return Fn.conv2d(img, k.view(1, 1, -1, 1), padding=(r, 0))


def target_outline(fwd, right, up):
    p = ((tgt - ctr) @ torch.stack([right, up]).T + ext) / (2 * ext) * RES
    ij = p.long()
    ok = (ij >= 0).all(1) & (ij < RES).all(1)
    cov = torch.zeros(RES * RES, device=dev)
    cov.index_put_((ij[ok, 1] * RES + ij[ok, 0],), torch.ones(int(ok.sum()), device=dev), accumulate=True)
    m = (cov.view(RES, RES) > 0).float()
    m = (Fn.max_pool2d(m[None, None], 5, 1, 2)[0, 0] > 0).float()      # close sampling holes
    er = -Fn.max_pool2d(-m[None, None], 5, 1, 2)[0, 0]
    return (m - er) > 0                                                  # ring = edge


def render(x, fwd, right, up, sp, outline):
    p = (x - ctr) @ torch.stack([right, up]).T
    z = (x - ctr) @ fwd
    px = (p + ext) / (2 * ext) * RES
    rad = (a.splat_k * sp * pix_per_wu).clamp(1.5, RMAX)
    cx, cy = px[:, 0].round().long(), px[:, 1].round().long()
    # all (particle, offset) pairs inside the particle's disk
    inside = odist2[None, :] <= (rad * rad)[:, None]                    # (N, K)
    n_idx, k_idx = inside.nonzero(as_tuple=True)
    xs = cx[n_idx] + ox[k_idx].long(); ys = cy[n_idx] + oy[k_idx].long()
    ok = (xs >= 0) & (xs < RES) & (ys >= 0) & (ys < RES)
    xs, ys, n_idx, k_idx = xs[ok], ys[ok], n_idx[ok], k_idx[ok]
    zc = z[n_idx] - torch.sqrt((rad[n_idx] ** 2 - odist2[k_idx]).clamp(min=0)) / pix_per_wu * 0.5
    depth = torch.full((RES * RES,), float("inf"), device=dev)
    depth = depth.scatter_reduce(0, ys * RES + xs, zc, reduce="amin", include_self=True).view(RES, RES)
    cover = torch.isfinite(depth)
    w = cover.float()[None, None]
    dz = torch.where(cover, depth, torch.zeros_like(depth))[None, None]
    sig = max(1.0, a.smooth * float(rad.median()))
    sm = gauss_blur(dz * w, sig) / gauss_blur(w, sig).clamp(min=1e-6)
    jump = (sm - dz).abs() > 0.6 * float(sp.median()) * 4
    sm = torch.where(jump & cover[None, None], dz, sm)
    sm = gauss_blur(sm * w, 0.5 * sig) / gauss_blur(w, 0.5 * sig).clamp(min=1e-6)
    gy = Fn.conv2d(sm, gy1, padding=(1, 0)) * pix_per_wu
    gx = Fn.conv2d(sm, gx1, padding=(0, 1)) * pix_per_wu
    n = torch.cat([-gx, gy, torch.ones_like(gx)], 1)
    n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)
    n = n[0].permute(1, 2, 0) * cover[..., None]
    # AO
    ao = torch.ones(RES, RES, device=dev)
    for rr in (3, 6, 12, 24):
        m = Fn.avg_pool2d(sm, rr * 2 + 1, 1, rr)[0, 0]
        ao = ao - 0.18 * ((sm[0, 0] - m) / (0.03 * ext)).clamp(0, 1)
    ao = ao.clamp(0.35, 1.0)
    # GGX
    V = torch.tensor([0.0, 0.0, 1.0], device=dev)
    NdV = (n @ V).clamp(1e-4, 1.0)
    F0 = torch.full((3,), 0.04, device=dev)
    col = torch.zeros(RES, RES, 3, device=dev)
    for L, c, inten in (((0.45, 0.65, 0.62), (1.0, 0.98, 0.94), 2.6), ((-0.7, 0.25, 0.65), (0.80, 0.86, 1.0), 1.1), ((0.2, 0.5, -0.85), (1.0, 1.0, 1.0), 1.6)):
        L = torch.tensor(L, device=dev); L = L / L.norm()
        H = L + V; H = H / H.norm()
        NdL = (n @ L).clamp(0, 1); NdH = (n @ H).clamp(0, 1); VdH = float((V @ H).clamp(0, 1))
        a2 = (a.rough ** 2) ** 2
        Dg = a2 / (math.pi * (NdH * NdH * (a2 - 1) + 1) ** 2 + 1e-9)
        k = (a.rough + 1) ** 2 / 8
        G = (NdV / (NdV * (1 - k) + k)) * (NdL / (NdL * (1 - k) + k + 1e-9))
        Fr = F0 + (1 - F0) * (1 - VdH) ** 5
        spec = (Dg * G)[..., None] * Fr / (4 * NdV * NdL + 1e-4)[..., None]
        col = col + ((1 - Fr) * albedo / math.pi + spec) * (NdL[..., None] * inten) * torch.tensor(c, device=dev)
    sky, ground = torch.tensor([0.62, 0.68, 0.78], device=dev), torch.tensor([0.30, 0.27, 0.24], device=dev)
    hemi = 0.5 * (1 + n[..., 1:2])
    col = col + albedo * (hemi * sky + (1 - hemi) * ground) * 0.55 * ao[..., None]
    col = (col / (1 + col)).clamp(0, 1) ** (1 / 2.2)
    bg_t = torch.linspace(0, 1, RES, device=dev)[:, None, None]
    bg = (0.97 - 0.06 * bg_t) * torch.tensor([1.0, 0.995, 0.985], device=dev)
    bg = bg.expand(RES, RES, 3).clone()
    shadow = gauss_blur(cover.float()[None, None], 10)[0, 0]
    shadow = torch.roll(torch.roll(shadow, 14, 0), 6, 1)
    bg = bg * (1 - 0.35 * shadow)[..., None]
    img = torch.where(cover[..., None], col, bg)
    # target outline in a faint warm tone where the morph does not cover
    img = torch.where((outline & ~cover)[..., None], torch.tensor([0.85, 0.42, 0.30], device=dev), img)
    return (img.clamp(0, 1).flip(0) * 255).byte().cpu().numpy()


outlines = [target_outline(f, r, u) for f, r, u in views]
out_frames = []
sp = None
for j, i in enumerate(idx):
    x = torch.as_tensor(frames_np[i], device=dev, dtype=torch.float32)
    if sp is None or j % 10 == 0:
        sp = local_spacing(x)
    panels = [render(x, f, r, u, sp, o) for (f, r, u), o in zip(views, outlines)]
    im = Image.fromarray(np.concatenate(panels, axis=1))
    ImageDraw.Draw(im).text((6, 6), f"{a.label}  frame {i}/{dn-1}", fill=(30, 30, 30))
    out_frames.append(im)
out_frames += [out_frames[-1]] * a.hold
out_frames[0].save(a.out, save_all=True, append_images=out_frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print(f"saved {a.out}  ({len(out_frames)} frames, {len(idx)} rendered, res {RES}, views {a.views})")
if a.mp4:
    try:
        import imageio
        imageio.mimwrite(a.mp4, [np.asarray(f) for f in out_frames], fps=a.fps, quality=8)
        print("saved", a.mp4)
    except Exception as e:
        print("mp4 skipped:", e)
