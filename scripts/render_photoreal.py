"""Photoreal 3DGS render of a morph result: real diff_gauss rasterizer, studio lighting.

Usage: python render_photoreal.py --npz output/heroX_<arm>.npz --out output/photoreal.png
"""
import argparse, math, sys
import numpy as np, torch
sys.path.insert(0, ".")
from physmorph.render.covariance import cov_from_F, sigma0_from_nn
from physmorph.pipeline.render_loss import field_normals
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

def world2view(right, up, fwd, campos):
    import numpy as np
    A = np.stack([right, -up, fwd], 0).astype(np.float32)   # rows = cam axes, y-down (COLMAP)
    W = np.eye(4, dtype=np.float32)
    W[:3, :3] = A
    W[:3, 3] = -A @ campos
    return W

def proj_matrix(znear, zfar, fovx, fovy):
    tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 1 / tx; P[1, 1] = 1 / ty
    P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=1400)
ap.add_argument("--azim", type=float, default=35.0)
ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--sigma_k", type=float, default=1.6, help="display splat sigma in NN spacings")
ap.add_argument("--adaptive_k", type=int, default=0, help=">0: per-particle sigma = sigma_k x mean distance to its k nearest neighbours (fills clustering gaps)")
ap.add_argument("--surface_only", type=float, default=0.0, help=">0: render only the surface parents (fraction, kNN one-sidedness score) - required for SOLID bodies")
ap.add_argument("--surfel", type=float, default=0.0,
                help=">0: SURFEL display - each particle becomes a flat Gaussian in its PCA "
                     "tangent plane, sigma_n = this x local spacing along the normal (0.15 "
                     "is crisp), sigma_t = --sigma_t x local spacing tangentially. A solid "
                     "body rendered with isotropic splats of ~1.6 spacings is a cloud; the "
                     "surface layer rendered as oriented surfels (2DGS-style) is a surface.")
ap.add_argument("--sigma_t", type=float, default=1.1, help="surfel tangential sigma in local spacings")
ap.add_argument("--frame", type=int, default=None,
                help="frame index; default = the DELIVERED frame (deliver_n-1) if the "
                     "archive carries one, else the last frame")
a = ap.parse_args()

d = np.load(a.npz)
n_frames = len(d["frames"])
if a.frame is None:
    fi = int(d["deliver_n"]) - 1 if "deliver_n" in d.files else n_frames - 1
else:
    fi = a.frame if a.frame >= 0 else n_frames + a.frame
x = d["frames"][fi].astype(np.float32)
# F is sampled sparsely (F_sample_idx): take the latest sample at or before the frame
if "F_sample_idx" in d.files:
    sidx = np.asarray(d["F_sample_idx"])
    k = int(np.searchsorted(sidx, fi, side="right") - 1)
    F = d["F_samples"][max(k, 0)].astype(np.float32)
else:
    F = d["F_samples"][-1].astype(np.float32)
print(f"[photoreal] frame {fi} of {n_frames}")
if a.surface_only > 0:
    from physmorph.pipeline.runner import _surface_weights
    _sw = _surface_weights(x, 24, a.surface_only, 0.05) > 0.5
    x, F = x[_sw], F[_sw]
    print(f"[photoreal] surface-only: {int(_sw.sum())} of {len(_sw)} particles")
dev = "cuda"
xt = torch.tensor(x, device=dev)
sigma0 = sigma0_from_nn(x, a.sigma_k)
cov = torch.tensor(cov_from_F(F, sigma0), device=dev)
if a.adaptive_k > 0:
    from scipy.spatial import cKDTree as _KD
    _d, _ = _KD(x).query(x, k=a.adaptive_k + 1, workers=-1)
    _loc = _d[:, 1:].mean(1).astype(np.float32)            # local spacing per particle
    _scale = (_loc / np.median(_loc)).astype(np.float32)    # relative: the median particle keeps sigma0
    cov = cov * torch.tensor(_scale ** 2, device=dev)[:, None, None]
    print(f"[photoreal] adaptive sigma: k={a.adaptive_k} scale med {np.median(_scale):.2f} p90 {np.percentile(_scale,90):.2f}")
cov6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                    cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1).contiguous()

# ---- studio lighting baked per particle (normals from the density field) ----
lo = x.min(0); hi = x.max(0)
gmin = torch.tensor(lo - 0.5, device=dev)
dims = (72, 72, 72)
dxg = float((hi - lo).max() + 1.0) / 72
n_field, sw = field_normals(xt, gmin, dxg, dims)
# PCA normals (kNN plane fit) - far smoother than density-gradient normals; sign
# oriented by the field normal so non-convex regions stay outward
from scipy.spatial import cKDTree
import numpy as _np
_xn = x
_idx = cKDTree(_xn).query(_xn, k=25, workers=-1)[1]
_nb = _xn[_idx] - _xn[_idx].mean(1, keepdims=True)
_cov = _np.einsum('nki,nkj->nij', _nb, _nb)
_w, _v = _np.linalg.eigh(_cov)
_pn = _v[:, :, 0]
n_p = torch.tensor(_pn.astype(_np.float32), device=dev)
flip = (n_p * n_field).sum(1, keepdim=True) < 0
n_p = torch.where(flip, -n_p, n_p)
if a.surfel > 0:
    # oriented surfels: covariance R diag(st^2, st^2, sn^2) R^T in the PCA frame
    # (t1, t2 = the two larger-variance axes, n = the smallest); sizes from the LOCAL
    # spacing so clustered regions do not over-cover and sparse ones do not gap
    _loc_s = cKDTree(x).query(x, k=9, workers=-1)[0][:, 1:].mean(1).astype(_np.float32)
    st_ = (a.sigma_t * _loc_s)[:, None]; sn_ = (a.surfel * _loc_s)[:, None]
    t1 = _v[:, :, 2].astype(_np.float32); t2 = _v[:, :, 1].astype(_np.float32)
    nn_ = _pn.astype(_np.float32)
    _cov_s = (st_[:, :, None] ** 2 * (t1[:, :, None] * t1[:, None, :] + t2[:, :, None] * t2[:, None, :])
              + sn_[:, :, None] ** 2 * (nn_[:, :, None] * nn_[:, None, :]))
    cov = torch.tensor(_cov_s, device=dev)
    cov6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1).contiguous()
    print(f"[photoreal] surfels: sigma_t {a.sigma_t} sigma_n {a.surfel} x local spacing "
          f"(med {float(_np.median(_loc_s)):.4f})")
def lit(albedo, n, cam_fwd, cam_right, cam_up):
    # camera-relative studio rig: subject is lit from the camera's upper-left at any azimuth
    key  = (-cam_fwd + 1.1 * cam_up - 1.4 * cam_right); key = key / key.norm()
    fill = (-cam_fwd - 0.6 * cam_up + 0.7 * cam_right); fill = fill / fill.norm()
    rim  = (cam_fwd + 0.9 * cam_up); rim = rim / rim.norm()
    kd = (n @ key).clamp(min=0).pow(1.2)
    fd = (n @ fill).clamp(min=0)
    rd = (n @ rim).clamp(min=0).pow(3)
    c = (albedo[None, :] * (0.06 + 1.25 * kd[:, None] * torch.tensor([1.0, 0.98, 0.94], device=dev)
         + 0.13 * fd[:, None] * torch.tensor([0.82, 0.88, 1.0], device=dev))
         + 0.18 * rd[:, None] * torch.tensor([0.9, 0.95, 1.0], device=dev))
    return (0.72 * c).clamp(0, 1.0)
albedo = torch.tensor([0.93, 0.90, 0.86], device=dev)     # warm porcelain
opac = torch.full((len(x), 1), 0.97, device=dev)

# ---- camera ----
center = torch.tensor((lo + hi) / 2, device=dev).cpu().numpy()
radius = float(np.linalg.norm(hi - lo)) * 1.55
az, el = math.radians(a.azim), math.radians(a.elev)
campos = center + radius * np.array([math.sin(az) * math.cos(el), math.sin(el),
                                     math.cos(az) * math.cos(el)], np.float32)
fwd = center - campos; fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
W2V = world2view(right.astype(np.float32), up.astype(np.float32),
                 fwd.astype(np.float32), campos.astype(np.float32))
fov = math.radians(38.0)
P = proj_matrix(0.01, 100.0, fov, fov)
viewm = torch.tensor(W2V.T, device=dev)                    # 3DGS stores transposed
projm = torch.tensor((P @ W2V).T, device=dev)

cf = torch.tensor(fwd, device=dev, dtype=torch.float32)
cr = torch.tensor(right, device=dev, dtype=torch.float32)
cu = torch.tensor(up, device=dev, dtype=torch.float32)
lit_c = lit(albedo, n_p, cf, cr, cu)
# interior particles have unreliable normals (random PCA orientation) - they showed as
# dark dapples through the surface; blend them toward flat ambient albedo by surface weight
w_s = sw.clamp(0, 1).pow(0.4)[:, None]
colors = w_s * lit_c + (1 - w_s) * (albedo[None, :] * 0.92)

st = GaussianRasterizationSettings(
    image_height=a.res, image_width=a.res,
    tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2),
    bg=torch.tensor([0.94, 0.94, 0.95], device=dev),
    scale_modifier=1.0, viewmatrix=viewm, projmatrix=projm,
    sh_degree=0, campos=torch.tensor(campos, device=dev),
    prefiltered=False, debug=False)
rast = GaussianRasterizer(raster_settings=st)
means2D = torch.zeros_like(xt)
out = rast(means3D=xt, means2D=means2D, opacities=opac,
           colors_precomp=colors, cov3Ds_precomp=cov6,
           norm3Ds_precomp=n_p.contiguous())
img = out[0] if isinstance(out, tuple) else out
img = img.clamp(0, 1).pow(1 / 1.7).detach().cpu().numpy()  # gamma
img8 = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
try:
    from PIL import Image
    Image.fromarray(img8).save(a.out)
except ImportError:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(a.out, img8)
print("saved", a.out, "N =", len(x))
