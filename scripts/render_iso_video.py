"""Isosurface video of a morph (H4, 2026-09-16): the object as ONE implicit surface.

Per frame the particle mass is rasterised to a 3-D density grid over the run's box, blurred with
a Gaussian of one particle spacing, and the iso-level rho_iso = 0.5 x rho_bulk (rho_bulk = the
median density of the SOURCE's occupied voxels, i.e. the material's own bulk density) is
ray-marched orthographically for two azimuths; normals come from the density gradient at the
hit, shading is the same GGX + hemisphere + AO as render_surface_video.py. Mass below the
resolvable density (a lone particle) is not a surface and does not render — the raw-state
metrics (fragments, census) are reported separately and never come from this image.
Literature: Yu & Turk 2013 (implicit particle surfaces), van der Laan 2009 (screen-space
smoothing); this is the isotropic-kernel special case on a fixed grid.
Usage: render_iso_video.py --npz run.npz --out out.gif [--res 440 --stride 3 --grid 128 --views 35,215]
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
ap.add_argument("--res", type=int, default=440)
ap.add_argument("--stride", type=int, default=3, help="archived frames per video frame")
ap.add_argument("--views", default="35,215", help="azimuths in degrees")
ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--hold", type=int, default=16)
ap.add_argument("--label", default="")
ap.add_argument("--grid", type=int, default=128, help="density grid resolution per axis")
ap.add_argument("--iso", type=float, default=0.5, help="iso-level as a fraction of the source bulk density")
ap.add_argument("--blur", type=float, default=1.0, help="Gaussian sigma in particle spacings")
ap.add_argument("--albedo", default="0.80,0.78,0.74")
ap.add_argument("--rough", type=float, default=0.45)
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--still", type=int, default=-1, help="render only this archived frame to --out (png)")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(a.npz, allow_pickle=True)
frames_np = z["frames"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames_np)
tgt_np = np.asarray(z["tgt"], np.float32)
RES, G = a.res, a.grid
albedo = torch.tensor([float(c) for c in a.albedo.split(",")], device=dev)
tgt = torch.as_tensor(tgt_np, device=dev)
x0 = torch.as_tensor(np.asarray(frames_np[0], np.float32), device=dev)
N = x0.shape[0]

# ---- the box: every archived frame + the target, cube centred on the target -----------
lo = torch.minimum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).min(0), device=dev), tgt.min(0).values)
hi = torch.maximum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).max(0), device=dev), tgt.max(0).values)
ctr = 0.5 * (lo + hi)
half = float((hi - lo).max()) * 0.55
vox = 2 * half / G                                       # voxel size (wu)
# particle spacing of the source (median 8-NN distance on a subset) -> blur sigma
n_sub = min(N, 20000)
sub = x0[torch.randperm(N, device=dev)[:n_sub]]
d8 = torch.cdist(sub, sub).topk(9, largest=False).values[:, -1]
spacing = float(d8.median()) * (n_sub / N) ** (1.0 / 3.0)   # subsample -> full-cloud spacing
sig_vox = max(0.6, a.blur * spacing / vox)


def density(x):
    """Blurred mass density on the G^3 grid (trilinear splat + separable Gaussian)."""
    p = (x - (ctr - half)) / vox - 0.5                   # voxel coordinates (cell centres)
    i0 = torch.floor(p).long()
    f = p - i0.float()
    rho = torch.zeros(G * G * G, device=dev)
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                w = ((f[:, 0] if dx_ else 1 - f[:, 0]) * (f[:, 1] if dy_ else 1 - f[:, 1]) * (f[:, 2] if dz_ else 1 - f[:, 2]))
                i = i0 + torch.tensor([dx_, dy_, dz_], device=dev)
                ok = ((i >= 0) & (i < G)).all(1)
                idx = (i[ok, 2] * G + i[ok, 1]) * G + i[ok, 0]
                rho.index_put_((idx,), w[ok], accumulate=True)
    rho = rho.view(1, 1, G, G, G)
    r = int(3 * sig_vox)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sig_vox * sig_vox)); k = k / k.sum()
    rho = Fn.conv3d(rho, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
    rho = Fn.conv3d(rho, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
    rho = Fn.conv3d(rho, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
    return rho                                            # (1,1,G,G,G) indexed [z,y,x]


rho0 = density(x0)
occ = rho0[rho0 > 0]
rho_bulk = float(occ.median()) if occ.numel() else 1.0
iso = a.iso * rho_bulk


def camera(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    fwd = torch.tensor([-math.cos(el) * math.sin(az), -math.sin(el), -math.cos(el) * math.cos(az)], device=dev)
    fwd = fwd / fwd.norm()
    right = torch.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=dev), dim=0); right = right / right.norm()
    up = torch.cross(right, fwd, dim=0)
    return fwd, right, up


views = [camera(float(v), a.elev) for v in a.views.split(",")]
ext = half * 1.05                                         # image half-width (wu)
pix_per_wu = RES / (2 * ext)
S = int(2 * math.sqrt(3) * half / vox) + 2                # ray samples
ys_, xs_ = torch.meshgrid(torch.arange(RES, device=dev), torch.arange(RES, device=dev), indexing="ij")
uv = torch.stack([(xs_.float() + 0.5) / RES * 2 * ext - ext, (ys_.float() + 0.5) / RES * 2 * ext - ext], -1)   # (RES,RES,2)


def sample(rho, pts):
    """trilinear density at world points (...,3) -> (...)"""
    q = (pts - ctr) / half                                # [-1,1] in x,y,z
    g = q.view(1, -1, 1, 1, 3)                            # grid_sample wants (x,y,z) order for the last dim
    return Fn.grid_sample(rho, g, mode="bilinear", padding_mode="zeros", align_corners=False).view(pts.shape[:-1])


def render(rho, fwd, right, up, outline):
    origin = ctr + uv[..., 0:1] * right + uv[..., 1:2] * up - fwd * (math.sqrt(3) * half)   # (RES,RES,3)
    t = torch.arange(S, device=dev).float() * vox                                             # (S,)
    pts = origin[None] + t[:, None, None, None] * fwd                                         # (S,RES,RES,3)
    d = sample(rho, pts)                                                                      # (S,RES,RES)
    inside = d >= iso
    hit = inside.any(0)
    first = torch.where(hit, inside.float().argmax(0), torch.zeros_like(hit, dtype=torch.long))
    # linear interpolation of the crossing between samples first-1 and first
    k1 = first.clamp(min=1)
    d1 = d.gather(0, k1[None])[0]; d0 = d.gather(0, (k1 - 1)[None])[0]
    frac = ((iso - d0) / (d1 - d0).clamp(min=1e-9)).clamp(0, 1)
    tz = (k1.float() - 1 + frac) * vox
    depth = torch.where(hit, tz, torch.full_like(tz, float("inf")))
    hitp = origin + tz[..., None] * fwd
    eps = vox
    ex, ey, ez = (torch.tensor(v, device=dev) * eps for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    grad = torch.stack([sample(rho, hitp + ex) - sample(rho, hitp - ex),
                        sample(rho, hitp + ey) - sample(rho, hitp - ey),
                        sample(rho, hitp + ez) - sample(rho, hitp - ez)], -1)
    n_world = -grad / grad.norm(dim=-1, keepdim=True).clamp(min=1e-9)                       # outward
    # normals in camera frame (right, up, -fwd)
    n = torch.stack([n_world @ right, n_world @ up, -(n_world @ fwd)], -1)
    n = n * hit[..., None]
    cover = hit
    # AO from the depth map
    dz = torch.where(cover, depth, torch.zeros_like(depth))[None, None]
    ao = torch.ones(RES, RES, device=dev)
    for rr in (3, 6, 12, 24):
        m = Fn.avg_pool2d(dz, rr * 2 + 1, 1, rr)[0, 0]
        ao = ao - 0.18 * ((dz[0, 0] - m) / (0.03 * ext)).clamp(0, 1)
    ao = ao.clamp(0.35, 1.0)
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
        Gg = (NdV / (NdV * (1 - k) + k)) * (NdL / (NdL * (1 - k) + k + 1e-9))
        Fr = F0 + (1 - F0) * (1 - VdH) ** 5
        spec = (Dg * Gg)[..., None] * Fr / (4 * NdV * NdL + 1e-4)[..., None]
        col = col + ((1 - Fr) * albedo / math.pi + spec) * (NdL[..., None] * inten) * torch.tensor(c, device=dev)
    sky, ground = torch.tensor([0.62, 0.68, 0.78], device=dev), torch.tensor([0.30, 0.27, 0.24], device=dev)
    hemi = 0.5 * (1 + n[..., 1:2])
    col = col + albedo * (hemi * sky + (1 - hemi) * ground) * 0.55 * ao[..., None]
    col = (col / (1 + col)).clamp(0, 1) ** (1 / 2.2)
    bg_t = torch.linspace(0, 1, RES, device=dev)[:, None, None]
    bg = ((0.97 - 0.06 * bg_t) * torch.tensor([1.0, 0.995, 0.985], device=dev)).expand(RES, RES, 3).clone()
    sh = cover.float()[None, None]
    r = 10; kk = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / 50); kk = kk / kk.sum()
    sh = Fn.conv2d(Fn.conv2d(sh, kk.view(1, 1, 1, -1), padding=(0, r)), kk.view(1, 1, -1, 1), padding=(r, 0))[0, 0]
    sh = torch.roll(torch.roll(sh, 14, 0), 6, 1)
    bg = bg * (1 - 0.35 * sh)[..., None]
    img = torch.where(cover[..., None], col, bg)
    img = torch.where((outline & ~cover)[..., None], torch.tensor([0.85, 0.42, 0.30], device=dev), img)
    return (img.clamp(0, 1).flip(0) * 255).byte().cpu().numpy()


def target_outline(fwd, right, up):
    p = ((tgt - ctr) @ torch.stack([right, up]).T + ext) / (2 * ext) * RES
    ij = p.long()
    ok = (ij >= 0).all(1) & (ij < RES).all(1)
    cov = torch.zeros(RES * RES, device=dev)
    cov.index_put_((ij[ok, 1] * RES + ij[ok, 0],), torch.ones(int(ok.sum()), device=dev), accumulate=True)
    m = (cov.view(RES, RES) > 0).float()
    m = (Fn.max_pool2d(m[None, None], 5, 1, 2)[0, 0] > 0).float()
    er = -Fn.max_pool2d(-m[None, None], 5, 1, 2)[0, 0]
    return (m - er) > 0


outlines = [target_outline(f, r, u) for f, r, u in views]
if a.still >= 0:
    idx = [a.still]
else:
    idx = list(range(0, dn, a.stride))
    if idx[-1] != dn - 1:
        idx.append(dn - 1)
    if a.max_frames > 0:
        idx = idx[: a.max_frames]
out_frames = []
for i in idx:
    x = torch.as_tensor(np.asarray(frames_np[i], np.float32), device=dev)
    rho = density(x)
    panels = [render(rho, f, r, u, o) for (f, r, u), o in zip(views, outlines)]
    im = Image.fromarray(np.concatenate(panels, axis=1))
    ImageDraw.Draw(im).text((6, 6), f"{a.label}  frame {i}/{dn-1}", fill=(30, 30, 30))
    out_frames.append(im)
if a.still >= 0:
    out_frames[0].save(a.out)
    print(f"saved {a.out}  (frame {a.still}, grid {G}^3 vox {vox:.4f} wu, spacing {spacing:.4f}, iso {a.iso} x bulk)")
else:
    out_frames += [out_frames[-1]] * a.hold
    out_frames[0].save(a.out, save_all=True, append_images=out_frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"saved {a.out}  ({len(out_frames)} frames, {len(idx)} rendered, res {RES}, grid {G}^3 vox {vox:.4f} wu, spacing {spacing:.4f}, iso {a.iso} x bulk)")
