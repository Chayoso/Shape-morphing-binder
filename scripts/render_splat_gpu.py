"""render_splat_gpu.py NPZ OUT.mp4 — the v8 splat render on the GPU end to end (2026-09-25 19:20 CDT).

Same picture as render_splat_video8 --adaptive --blur 3 --soft --disc --sigma 1 --nsmooth 2 --deferred:
  * per particle: shell count n_i within the target's shell radius r (from the 9-NN on the GPU hash grid), support =
    min(1, n_i / (k/2)) (opacity), adaptive radius = sigma x spacing x clip(r_i / r, 1, 4) with r_i the particle's own
    8th-neighbour distance, the CIC density on a torch grid blurred 3 spacings (separable conv3d), its gradient sampled
    at the particles (trilinear), normals averaged over the 32 nearest (two passes), discs in the tangent plane;
  * deferred shading: the normal and coverage buffers from the 3DGS rasteriser, lit per pixel (hemispheric wrap light).
CPU work per frame: none but the PNG write. Two views (35 / 215 deg, elevation 18), stride 12, a fixed camera.
"""
import sys, os, math, argparse, subprocess, tempfile, time
import numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo")
from physmorph.render.photoreal import render_3dgs
from physmorph.render.knn_gpu import knn_self_torch
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("out")
ap.add_argument("--stride", type=int, default=12); ap.add_argument("--sigma", type=float, default=1.0)
ap.add_argument("--blur", type=float, default=3.0); ap.add_argument("--res", type=int, default=900)
ap.add_argument("--opacity", type=float, default=0.92); ap.add_argument("--hold", type=int, default=20)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--nsmooth", type=int, default=2)
ap.add_argument("--label", default=""); ap.add_argument("--stills", default="")
ap.add_argument("--views", default="35,215"); ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--material_size", action="store_true",
                help="P283 second stage (with --material_normals): an anchored particle's splat radius follows the in-plane "
                     "area change of its material neighbourhood and its support opacity stays at the anchor's value")
ap.add_argument("--material_normals", action="store_true",
                help="P283 (2026-09-26): normals anchored at a particle's first exposed frame and transported by the tangent-plane "
                     "deformation of its fixed 32-neighbourhood instead of being re-estimated every frame")
a = ap.parse_args()
dev = "cuda"
z = np.load(a.npz, allow_pickle=True)
F = z["frames"]; T = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
n = len(F); idx = list(range(0, n, a.stride))
ctr = T.mean(0); rad = float((T - ctr).norm(dim=1).max())
dT, _ = knn_self_torch(T, a.k + 1)
sp = float(dT[:, 1].median()); rcov = float(dT[:, a.k].median())
light = torch.tensor([0.35, -0.85, -0.4], device=dev); light = light / light.norm()
stills = set(int(s) for s in a.stills.split(",") if s.strip())
views = [float(v) for v in a.views.split(",")]

# the density grid (fixed over the run): cell = half the blur sigma
cell = 0.5 * a.blur * sp; lo = ctr - 1.15 * rad; dims = int(math.ceil(2.3 * rad / cell)) + 3
gk = int(math.ceil(3 * a.blur * sp / cell)); gx = torch.arange(-gk, gk + 1, device=dev, dtype=torch.float32)
gw = torch.exp(-0.5 * (gx * cell / (a.blur * sp)) ** 2); gw = (gw / gw.sum())

def density_normals(x):
    rel = (x - lo) / cell; base = torch.floor(rel).long(); f = rel - base
    g = torch.zeros(dims * dims * dims, device=dev)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                ix = (base[:, 0] + dx).clamp(0, dims - 1); iy = (base[:, 1] + dy).clamp(0, dims - 1); iz = (base[:, 2] + dz).clamp(0, dims - 1)
                g.index_add_(0, (ix * dims + iy) * dims + iz, w)
    g = g.reshape(1, 1, dims, dims, dims)
    for d in range(3):                                                   # separable Gaussian blur
        shape = [1, 1, 1, 1, 1]; shape[2 + d] = len(gw)
        g = Fn.conv3d(g, gw.reshape(shape), padding=[gk if i == d else 0 for i in range(3)])
    grad = torch.stack(torch.gradient(g[0, 0], spacing=cell), 0)            # (3, D, D, D)
    # trilinear sample at the particles: grid_sample wants (N,1,1,1,3) coords in [-1,1] ordered (z, y, x)
    coords = (rel / (dims - 1)) * 2 - 1
    sam = Fn.grid_sample(grad[None], coords[None, :, None, None, :].flip(-1), align_corners=True, mode="bilinear")
    nrm = -sam[0, :, :, 0, 0].T                                             # (N, 3)
    mag = nrm.norm(dim=1)
    return nrm / mag.clamp_min(1e-9)[:, None], mag

tmp = tempfile.mkdtemp(prefix="splatgpu_"); t0 = time.time(); img = None
mat = None                                                                   # P283 material-bound normal state
if a.material_normals:
    _N = len(np.asarray(F[0]))
    mat = dict(anch=torch.zeros(_N, dtype=torch.bool, device=dev), nb0=None, P0=None, G=None, n_prev=None,
               last=None, last_anch=None, last_refit=None,
               sig0=torch.zeros(_N, device=dev), sup0=torch.zeros(_N, device=dev))
    stat_lines = []
for kf, i in enumerate(idx):
    x = torch.as_tensor(np.asarray(F[i], np.float32), device=dev)
    d, j = knn_self_torch(x, 33)                                            # self first
    n_i = (d[:, 1:a.k + 1] <= rcov).sum(1).float()                          # neighbours within the target's shell radius
    support = (n_i / (0.5 * a.k)).clamp(0, 1)
    r_i = d[:, a.k]
    sig_i = a.sigma * sp * (r_i / rcov).clamp(1.0, 4.0)
    nrm, mag = density_normals(x)
    strong = mag >= torch.quantile(mag[::max(1, len(mag) // 100000)], 0.6)
    if strong.any() and (~strong).any():                                   # interior particles take a strong neighbour's normal
        js = j[:, 1:33]; ms = strong[js]; has = ms.any(1)
        first = torch.argmax(ms.float(), dim=1)
        nb = nrm[js[torch.arange(len(x), device=dev), first]]
        nrm = torch.where((~strong & has)[:, None], nb, nrm)
    if mat is None:
        for _ in range(a.nsmooth):
            nrm = nrm[j].mean(1); nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    else:
        # P283 (2026-09-26 00:30 CDT): MATERIAL-BOUND normals. A particle's normal is anchored at its first exposed
        # (strong) frame and afterwards transported by the tangent-plane deformation of its FIXED 32-neighbourhood: a
        # 2D -> 3D least-squares map of the rest in-plane coordinates to the current offsets, whose two columns are
        # the transported tangents (no thickness support needed, so a near-planar neighbourhood is well conditioned).
        # A relative fit residual above 0.5 (torn or dispersed neighbourhood) drops the anchor and re-anchors on the
        # refit normal. Smoothing runs over the fixed material graph for anchored particles (the refit smoothing over
        # the current k-NN is what mixes new neighbours in). The refit normal is still computed for the statistics.
        nr_s = nrm
        for _ in range(a.nsmooth):
            nr_s = nr_s[j].mean(1); nr_s = nr_s / nr_s.norm(dim=1, keepdim=True).clamp_min(1e-9)
        N_ = len(x); js = j[:, 1:33]
        if mat["nb0"] is None:
            mat.update(nb0=js.clone(), P0=torch.zeros(N_, 32, 2, device=dev), G=torch.zeros(N_, 2, 2, device=dev),
                       n_prev=nrm.clone())
        anch = mat["anch"]; n_trans = nrm.clone()
        if anch.any():
            ai = torch.nonzero(anch).squeeze(1)
            D = x[mat["nb0"][ai]] - x[ai][:, None]                                          # (M,32,3) current offsets
            Mmap = torch.einsum("mki,mkj->mij", D, mat["P0"][ai]) @ mat["G"][ai]           # (M,3,2) transported tangents
            nt = torch.cross(Mmap[:, :, 0], Mmap[:, :, 1], dim=1); nt_norm = nt.norm(dim=1)   # = the in-plane area ratio
            nt = nt / nt_norm.clamp_min(1e-9)[:, None]
            flip = (nt * mat["n_prev"][ai]).sum(1) < 0; nt[flip] = -nt[flip]
            pred = torch.einsum("mkj,mij->mki", mat["P0"][ai], Mmap)
            resid = (D - pred).norm(dim=(1, 2)) / D.norm(dim=(1, 2)).clamp_min(1e-9)
            bad = resid > 0.5
            n_trans[ai[~bad]] = nt[~bad]
            if a.material_size:
                # second stage: the splat radius follows the neighbourhood's in-plane area change (PhysGaussian's
                # covariance transport restricted to the tangent plane), the support opacity stays the anchor's
                area = nt_norm[~bad].clamp(0.25, 16.0)
                sig_i[ai[~bad]] = (mat["sig0"][ai[~bad]] * area.sqrt()).clamp(a.sigma * sp, 4.0 * a.sigma * sp)
                support[ai[~bad]] = mat["sup0"][ai[~bad]]
            anch[ai[bad]] = False
        new = strong & ~anch
        if new.any():
            ni = torch.nonzero(new).squeeze(1); n_a = nrm[ni]
            ref_ = torch.where(n_a[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device=dev), torch.tensor([[0, 1.0, 0]], device=dev)).expand(len(ni), 3)
            t1_ = torch.cross(n_a, ref_, dim=1); t1_ = t1_ / t1_.norm(dim=1, keepdim=True).clamp_min(1e-9); t2_ = torch.cross(n_a, t1_, dim=1)
            D0 = x[js[ni]] - x[ni][:, None]                                                    # (m,32,3) rest offsets at the anchor
            P0 = torch.stack([(D0 * t1_[:, None]).sum(2), (D0 * t2_[:, None]).sum(2)], 2)     # (m,32,2) in-plane rest coordinates
            A_ = torch.einsum("mki,mkj->mij", P0, P0) + 1e-9 * torch.eye(2, device=dev)
            mat["nb0"][ni] = js[ni]; mat["P0"][ni] = P0; mat["G"][ni] = torch.linalg.inv(A_)
            mat["sig0"][ni] = sig_i[ni]; mat["sup0"][ni] = support[ni]
            anch[ni] = True; n_trans[ni] = n_a
        mat["n_prev"] = n_trans.clone()
        n_s = n_trans
        for _ in range(a.nsmooth):
            n_s = torch.where(anch[:, None], n_s[mat["nb0"]].mean(1), n_s[j].mean(1))
            n_s = n_s / n_s.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if mat["last"] is not None:
            m_ = anch & mat["last_anch"]
            if m_.any():
                ang = torch.rad2deg(torch.arccos((n_s[m_] * mat["last"][m_]).sum(1).clamp(-1, 1)))
                ang_r = torch.rad2deg(torch.arccos((nr_s[m_] * mat["last_refit"][m_]).sum(1).clamp(-1, 1)))
                stat_lines.append((i / (n - 1), int(m_.sum()), float(ang.median()), float((ang > 5).float().mean()),
                                   float(ang_r.median()), float((ang_r > 5).float().mean()), float(anch.float().mean())))
        mat["last"], mat["last_anch"], mat["last_refit"] = n_s.clone(), anch.clone(), nr_s.clone()
        nrm = n_s
    # discs: R diag(s^2, s^2, (s/4)^2) R^T with the normal as the third axis
    ref = torch.where(nrm[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device=dev), torch.tensor([[0, 1.0, 0]], device=dev)).expand(len(x), 3)
    t1 = torch.cross(nrm, ref, dim=1); t1 = t1 / t1.norm(dim=1, keepdim=True).clamp_min(1e-9); t2 = torch.cross(nrm, t1, dim=1)
    R = torch.stack([t1, t2, nrm], 2)
    S = torch.zeros(len(x), 3, 3, device=dev); S[:, 0, 0] = sig_i ** 2; S[:, 1, 1] = sig_i ** 2; S[:, 2, 2] = (sig_i / 4) ** 2
    cov = (R @ S @ R.transpose(1, 2)).cpu().numpy().astype(np.float32)
    xn = x.cpu().numpy(); op = (a.opacity * support).cpu().numpy().astype(np.float32)
    ncol = (0.5 * (nrm + 1)).cpu().numpy().astype(np.float32)
    panels = []
    for az in views:
        kw = dict(F=None, sigma0=float(sp), cov=cov, opacity=op, azimuth=math.radians(az), elevation=math.radians(a.elev),
                  dist=rad * 3.6, res=a.res, fovy_deg=30.0, center=ctr.cpu().numpy())
        N_img = torch.as_tensor(render_3dgs(xn, ncol, bg=(0, 0, 0), **kw), device=dev)
        A_img = torch.as_tensor(render_3dgs(xn, np.ones_like(ncol), bg=(0, 0, 0), **kw), device=dev)[..., 0]
        blur = lambda im: Fn.avg_pool2d(im.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
        N_s = blur(N_img); A_s = blur(A_img[..., None])[..., 0]
        npx = 2 * N_s / A_s.clamp_min(1e-3)[..., None] - 1; npx = npx / npx.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ndl = (npx * (-light)).sum(-1)
        shade = (0.40 + 0.60 * (0.5 * (1 + ndl)).clamp(0, 1)) * 0.97
        alpha = A_img.clamp(0, 1)[..., None]
        panels.append((shade[..., None] * alpha + 0.92 * (1 - alpha)).expand(-1, -1, 3).cpu().numpy())
    img = (np.concatenate(panels, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img); ImageDraw.Draw(im).text((10, 8), f"{a.label} splat frame {i}/{n - 1}  under-half {100 * float((n_i < 0.5 * a.k).float().mean()):.1f}%", fill=(20, 20, 20))
    im.save(os.path.join(tmp, f"f{kf:04d}.png"))
    if kf in stills:
        im.save(os.path.splitext(a.out)[0] + f"_f{kf}.png")
    if kf % 16 == 0:
        print(f"frame {kf + 1}/{len(idx)}  {time.time() - t0:.0f} s", flush=True)
for h in range(a.hold):
    Image.fromarray(img).save(os.path.join(tmp, f"f{len(idx) + h:04d}.png"))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "20", "-i", os.path.join(tmp, "f%04d.png"),
                "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
print(f"saved {a.out} ({len(idx)} frames + {a.hold} hold; {time.time() - t0:.0f} s total; sp {sp:.4f} rcov {rcov:.4f})")
if mat is not None and stat_lines:
    S_ = np.array(stat_lines)
    print("P283 normal turn per rendered frame (deg, particles anchored at both frames): t band | n | anchored share | "
          "transported med (>5deg) | refit med (>5deg)")
    for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, .9), (.9, 1.01)]:
        m_ = (S_[:, 0] > lo) & (S_[:, 0] <= hi)
        if m_.any():
            r = S_[m_]
            print(f"   {lo:.1f}-{min(hi, 1):.1f} | {int(np.median(r[:, 1])):7d} | {np.median(r[:, 6]):.2f} | {np.median(r[:, 2]):5.2f} ({np.median(r[:, 3]):.2f}) | "
                  f"{np.median(r[:, 4]):5.2f} ({np.median(r[:, 5]):.2f})")
