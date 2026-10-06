"""child_detail_static.py FRAMES_NPZ DENSE_NPZ[,DENSE_NPZ...] DENSE_IND_NPZ TARGET_OBJ OUT_DIR [STEPS] [+smooth=1] — D111
stage 0: a display surface finer than the body, its relief written by the render term, on a still frame (no simulation).

Base: the exterior's discs of the run's last kept frame (the 300k particles' Zhu–Bridson field at the run's pitch a),
on a lattice of 0.92 of the dense sample's pitch a_c (a 1.5M sample: a_c = 0.585 a). Each disc has a height h along its
normal n; the drawn point is p + h n and its normal normalize(n - grad_s h), grad_s h the least-squares slope of h over
the disc's 16 nearest discs in its tangent plane. h is fitted by Adam (STEPS, default 300; |h| <= a) to the render term
(the exterior's silhouette and shading measures, d_exterior) against the pictures the same operators draw of the dense
sample's own exterior (its pitch a_c, the same lattice), from the pipeline's 18 views at a pixel of a_c. Read: the
render terms against those pictures and against an independent dense sample's (DENSE_IND_NPZ), before and after; the
discs (before, after) and the dense sample's own exterior kept in OUT_DIR in exterior_offset_probe.py's format (signed
offset from the mesh in pitches of a, its Gaussian means) for ag2_bands.py; 4K stills of before / after / the dense
sample (the field's own normals, the studio light, slate blue on white). Several dense samples: the target pictures are
the mean of theirs (D91). +smooth=1: h = S v with S the render's splat footprint (a tent of one pixel), v fitted."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
import trimesh                                                 # noqa: E402
from PIL import Image                                          # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.render.support import normal_filter_size        # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
z = np.load(argv[1], allow_pickle=True)
denses = [torch.as_tensor(np.asarray(np.load(p)["tgt"], np.float32), device=dev) for p in argv[2].split(",")]
dense = denses[0]
dense_ind = torch.as_tensor(np.asarray(np.load(argv[3])["tgt"], np.float32), device=dev)
obj, out = argv[4], Path(argv[5])
steps = int(argv[6]) if len(argv) > 6 else 300
SMOOTH = opts.get("smooth", "0") == "1"
out.mkdir(parents=True, exist_ok=True)
cfg = PipelineConfig()
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
x = torch.as_tensor(np.asarray(z["frames"][-1], np.float32), device=dev)
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731  (the volume sample's pitch)
a, a_c = pitch(tgt), pitch(dense)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
extent = float(tgt.abs().max()) * 1.25                        # build_target's
res = int(np.ceil(2.0 * extent / a_c))                         # a pixel of one dense pitch
h_lat = 0.92 * a_c
lat = Lattice(center, 2.8 * radius)
views = make_views(cfg.render_views, cfg.render_elevs)
print(f"pitch {a:.4f} wu, dense pitch {a_c:.4f} wu ({a_c / a:.3f}); render {res} px over 2 x {extent:.3f} wu "
      f"({2 * extent / res / a:.3f} pitches a pixel), {len(views)} views; lattice {h_lat / a:.3f} pitches", flush=True)


def exterior(s, a_field):
    with torch.no_grad():
        p, g, _, _ = lat.discs(ZhuBridson(s, a_field), h_lat)
    return p, nnf.normalize(g, dim=1)


p_t, n_t = exterior(dense, a_c)
p_i, n_i = exterior(dense_ind, a_c)
per = [exterior_targets(*exterior(s, a_c), views, res, extent, cfg.sil_k, cfg.pbr_ambient) for s in denses]
sils = [torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])]          # the mean over the dense draws' pictures
shade = [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])]
sils_i, shade_i = exterior_targets(p_i, n_i, views, res, extent, cfg.sil_k, cfg.pbr_ambient)
p0, n0 = exterior(x, a)                                        # the base: the body's surface
M = len(p0)
print(f"discs: base {M}, dense {len(p_t)}, independent {len(p_i)}", flush=True)

# the least-squares slope of h in each disc's tangent plane over its 16 nearest discs (fixed: the base does not move)
K = 16
_, nb = knn_self_torch(p0, K + 1)
nb = nb[:, 1:]
ref = torch.where(n0[:, :1].abs() < .9, n0.new_tensor((1., 0., 0.)), n0.new_tensor((0., 1., 0.))).expand_as(n0)
t1 = nnf.normalize(torch.linalg.cross(n0, ref), dim=1)
t2 = torch.linalg.cross(n0, t1)
d = p0[nb] - p0[:, None]                                       # (M, K, 3)
D = torch.stack(((d * t1[:, None]).sum(-1), (d * t2[:, None]).sum(-1)), -1)        # (M, K, 2)
w = torch.exp(-(d.norm(dim=-1) / (1.5 * h_lat)) ** 2)[..., None]                   # (M, K, 1)
A = torch.linalg.solve((D * w).transpose(1, 2) @ D + 1e-6 * h_lat ** 2 * torch.eye(2, device=dev),
                       (D * w).transpose(1, 2))                                     # (M, 2, K)


def drawn(hh):
    dh = hh[nb] - hh[:, None]                                  # (M, K)
    g2 = (A @ dh[..., None]).squeeze(-1)                       # (M, 2): the slope in (t1, t2)
    n = nnf.normalize(n0 - g2[:, :1] * t1 - g2[:, 1:] * t2, dim=1)
    return p0 + hh[:, None] * n0, n


def terms(hh, s_, sh_):
    q, n = drawn(hh)
    ls, lp = d_exterior(q, n, s_, sh_, views, res, extent, cfg.sil_k, ambient=cfg.pbr_ambient)
    return ls, lp


pix = 2.0 * extent / res
if SMOOTH:
    # +smooth=1: h = S v, S the render's own splat footprint (a tent of one pixel over the disc and its 16 nearest,
    # rows normalised): h holds nothing the pictures cannot see (Mip-Splatting), and the update is smoothed at that
    # scale (Nicolet's preconditioning by reparameterisation)
    dist = torch.cat((torch.zeros(M, 1, device=dev), d.norm(dim=-1)), 1)            # (M, K+1): self first
    idx = torch.cat((torch.arange(M, device=dev)[:, None], nb), 1)
    ws = (1.0 - dist / pix).clamp_min(0.0)
    ws = ws / ws.sum(1, keepdim=True)
    smooth = lambda v: (ws * v[idx]).sum(1)                    # noqa: E731
else:
    smooth = lambda v: v                                       # noqa: E731
v = torch.zeros(M, device=dev, requires_grad=True)
h = smooth(v)
with torch.no_grad():
    before = [float(q) for q in terms(h, sils, shade)] + [float(q) for q in terms(h, sils_i, shade_i)]
opt = torch.optim.Adam([v], lr=0.05 * a_c)
for it in range(steps):
    opt.zero_grad()
    h = smooth(v)
    ls, lp = terms(h, sils, shade)
    (ls + cfg.w_pbr * lp).backward()
    opt.step()
    with torch.no_grad():
        v.clamp_(-a, a)
        h = smooth(v)
    if it % 50 == 0 or it == steps - 1:
        print(f"step {it}: silhouette {float(ls):.6f} shading {float(lp):.6f}; |h| rms {float(h.square().mean().sqrt()) / a:.3f} "
              f"max {float(h.abs().max()) / a:.3f} pitches", flush=True)
with torch.no_grad():
    after = [float(q) for q in terms(h, sils, shade)] + [float(q) for q in terms(h, sils_i, shade_i)]
row = dict(before=dict(fit_sil=before[0], fit_pbr=before[1], ind_sil=before[2], ind_pbr=before[3]),
           after=dict(fit_sil=after[0], fit_pbr=after[1], ind_sil=after[2], ind_pbr=after[3]),
           h_rms_pitches=float(h.square().mean().sqrt()) / a, discs=M, res=res)
print(json.dumps(row), flush=True)
print("render terms against the independent sample's pictures: silhouette %+.1f %%, shading %+.1f %%" % (
    100 * (after[2] / before[2] - 1), 100 * (after[3] / before[3] - 1)), flush=True)

# the discs in exterior_offset_probe.py's format (offsets in pitches of a), for ag2_bands.py
mesh = load_mesh(obj)
mesh.merge_vertices()
o = orient_name(obj)
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
t_np = tgt.cpu().numpy().astype(np.float64)
vb = np.asarray(mesh.bounds, np.float64)
step = float(knn_self_torch(tgt, 7)[0][:, 6].median())
mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * float(np.mean((t_np.max(0) - t_np.min(0) + step) / (vb[1] - vb[0]))) \
    + 0.5 * (t_np.max(0) + t_np.min(0))
pts, face = trimesh.sample.sample_surface(mesh, 3000000, seed=0)
surf = torch.as_tensor(np.asarray(pts, np.float32), device=dev)
surf_n = torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev)
tree = gpu.KNN(surf)


def gmean(P, s, sigma, k):
    dd, ii = gpu.KNN(P).query(P, k)
    ww = torch.exp(-.5 * (dd.float() / sigma) ** 2)
    return (ww * s[ii]).sum(1) / ww.sum(1)


def keep(name, P, n):
    with torch.no_grad():
        at = tree.query(P.contiguous(), 1)[1].reshape(-1)
        near = (P - surf[at]).norm(dim=1) < 2. * a
        P, n, at = P[near], n[near], at[near]
        m = surf_n[at]
        s = ((P - surf[at]) * m).sum(1) / a
        np.savez(out / f"{name}.npz", points=P.cpu().numpy(), normals=n.cpu().numpy(), mesh_normals=m.cpu().numpy(),
                 s=s.cpu().numpy(), g1=gmean(P, s, a, 64).cpu().numpy(), g25=gmean(P, s, 2.5 * a, 256).cpu().numpy(), a=a)
        print(f"{name}: {int(near.sum())} discs within two pitches of the mesh; mean offset {float(s.mean()):+.3f}", flush=True)


with torch.no_grad():
    q1, n1 = drawn(h)
keep("children_before", p0, n0)
keep("children_after", q1, n1)
keep("dense_own", p_t, n_t)


def still(P, n, name):
    cam = StudioRaster(center, radius, 3840, 2160, 35., 18.)
    cam.background = torch.zeros_like(cam.background)
    cam.albedo = torch.tensor((.08, .17, .40), device=dev)
    rr = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    tt = nnf.normalize(torch.linalg.cross(n, rr), dim=1, eps=1e-9)
    rot = torch.stack((tt, torch.linalg.cross(n, tt), n), dim=2)
    sig = torch.full((len(P),), h_lat, device=dev)
    cov = (rot * torch.stack((sig ** 2, sig ** 2, (sig / 4) ** 2), 1)[:, None]) @ rot.transpose(1, 2)
    with torch.no_grad():
        img, cover, _ = cam(P, n, cov, torch.full((len(P),), .92, device=dev), normal_kernel=normal_filter_size(2160, False),
                            return_buffers=True)
    Image.fromarray(((img + (1. - cover[..., None])).clamp(0, 1) * 255).round().byte().cpu().numpy()).save(out / f"{name}.png")


with torch.no_grad():
    still(p0, n0, "still_before")
    still(q1, n1, "still_after")
    still(p_t, n_t, "still_dense")
torch.save(dict(h=h.detach().cpu(), p0=p0.cpu(), n0=n0.cpu()), out / "fit.pt")
print("wrote", out, flush=True)
