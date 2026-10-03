"""disc_rule_probe.py TARGET_ARCHIVE OUT_DIR NAME=ARCHIVE:RAW:x0,y0,x1,y1 [...] — D35: the display's measure of sparsity.

The 4K display draws a particle as a disc of sigma = target spacing x clamp(S_i / S_ref, 1, 4). The rule until D35
(R0): S = the distance to the 8th neighbour in space, S_ref = its median over ALL target particles. Nine tenths of those
are interior, and a surface particle has half its neighbourhood empty, so R0 reads "on the surface" as "sparse" (D34:
12.4 % of a perfect sample's particles are inflated). Candidates, each with S_ref = the median of the same S over the
target's SURFACE particles (neighbourhood asymmetry of half a coverage radius or more):
  R1   S = the 8th-neighbour distance in space (the reference moved to the surface, nothing else);
  R2   S = the 8th smallest distance, among the 32 nearest neighbours, measured in the particle's tangent plane
       (|(I - n n^T)(x_j - x_i)| with the display's own normal n): the spacing on the surface;
  R2s  R2 over the neighbours on the same sheet only (n_i . n_j > 0): a thin feature's opposite face is left out.
Each object (NAME=ARCHIVE:RAW:box; the target sample is one of them) is drawn with R0, the candidates and no inflation.
Printed: the rendered particles inflated by 1.1 / 1.5 times or more, the covered pixels lost against R0 (whole frame,
box), the silhouette edge width, the shading detail against R0, the box's strong-gradient share. A sheet per object."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402,F401  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support              # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
load = lambda p: np.load(p, allow_pickle=True)  # noqa: E731
zt = load(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tgt = torch.as_tensor(np.asarray(zt["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
spacing, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
studio = StudioRaster(center, radius, W, H, AZ, EL)
density_normals = DensityNormals(center, radius, spacing, blur=3.0)


def prims(x):
    """(distances, neighbours, support, normals): the display renderer's primitives before the disc size."""
    d, nb = knn_self_torch(x, 33)
    normals, mag = density_normals(x)
    strong = mag >= torch.quantile(mag[::max(1, len(x) // 100000)], .6)
    near = nb[:, 1:33]
    sn = strong[near]
    chosen = near[torch.arange(len(x), device=dev), sn.float().argmax(1)]
    normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
    for _ in range(2):
        normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
    return d, nb, live_support(d, cov_r, spacing), normals


def in_plane(x, nb, normals, same_sheet):
    """The 8th smallest tangent-plane distance among the 32 nearest neighbours (same-sheet ones only if asked)."""
    off = x[nb[:, 1:]] - x[:, None, :]
    p = (off - (off * normals[:, None, :]).sum(-1, keepdim=True) * normals[:, None, :]).norm(dim=-1)
    if same_sheet:
        ok = (normals[nb[:, 1:]] * normals[:, None, :]).sum(-1) > 0
        p = torch.where(ok, p, torch.full_like(p, float("inf")))
    s = p.sort(dim=1).values
    k = (torch.isfinite(s).sum(1).clamp(1, 8) - 1)[:, None]
    v = s.gather(1, k)[:, 0]
    return torch.where(torch.isfinite(v), v, off.norm(dim=-1)[:, 7])


def measures_S(x, d, nb, normals):
    return {"R0": d[:, 8], "R1": d[:, 8], "R2": in_plane(x, nb, normals, False), "R2s": in_plane(x, nb, normals, True)}


with torch.inference_mode():
    dT, nbT, _, nT = prims(tgt)
    surface = (tgt - tgt[nbT[:, 1:]].mean(1)).norm(dim=1) >= 0.5 * cov_r
    ST = measures_S(tgt, dT, nbT, nT)
    REF = {"R0": cov_r, **{k: float(ST[k][surface].median()) for k in ("R1", "R2", "R2s")}}
print(f"target: {len(tgt)} particles, {100 * float(surface.float().mean()):.1f} % on the surface; spacing {spacing:.4f} wu; references in spacings: "
      + ", ".join(f"{k} {v / spacing:.2f}" for k, v in REF.items()))


def discs(normals, sigma):
    ref = torch.where(normals[:, :1].abs() < .9, normals.new_tensor((1., 0., 0.)), normals.new_tensor((0., 1., 0.))).expand_as(normals)
    tangent = nnf.normalize(torch.linalg.cross(normals, ref), dim=1, eps=1e-9)
    rot = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    var = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    return (rot * var[:, None]) @ rot.transpose(1, 2)


def grad_mag(f):
    gy, gx = torch.gradient(f)
    return (gx ** 2 + gy ** 2).sqrt()


lum = lambda img: img @ img.new_tensor((.2126, .7152, .0722))  # noqa: E731
for arg in sys.argv[3:]:
    name, rest = arg.split("=", 1)
    path, raw, b = rest.rsplit(":", 2)
    raw, (x0, y0, x1, y1) = int(raw), [int(v) for v in b.split(",")]
    with torch.inference_mode():
        x = torch.as_tensor(np.asarray(load(path)["frames"][raw], np.float32), device=dev)
        d, nb, support, normals = prims(x)
        S = measures_S(x, d, nb, normals)
        ren = support > 0
        print(f"\n== {name} (raw {raw}, {int(ren.sum())} rendered particles), box {x0},{y0},{x1},{y1}")
        print("   rule | inflated 1.1x / 1.5x or more % | covered pixels lost against R0 % (whole frame, box) | silhouette edge width px | shading detail against R0 | box: strong-gradient share %")
        base, tiles = None, []
        for rule in ("R0", "R1", "R2", "R2s", "none"):
            infl = torch.ones_like(d[:, 8]) if rule == "none" else (S[rule] / REF[rule]).clamp(1., 4.)
            img, cov, _ = studio(x, normals, discs(normals, spacing * infl), .92 * support, normal_kernel=3, return_buffers=True)
            gc = grad_mag(cov)
            edge = (cov > .4) & (cov < .6) & (gc > 1e-4)
            det = float(grad_mag(lum(img))[cov > .99].mean())
            if base is None:
                base = (cov >= .5, det)
            cb, bb = cov[y0:y1, x0:x1] >= .5, base[0][y0:y1, x0:x1]
            strong = float((grad_mag(255 * lum(img[y0:y1, x0:x1]))[cb] > 4).float().mean()) if bool(cb.any()) else 0.0
            print(f"   {rule:4s} | {100 * float((infl[ren] >= 1.1).float().mean()):5.1f} / {100 * float((infl[ren] >= 1.5).float().mean()):4.1f} | "
                  f"{100 * float((base[0] & ~(cov >= .5)).sum() / base[0].sum()):5.2f}, {100 * float((bb & ~cb).sum() / bb.sum().clamp_min(1)):5.2f} | "
                  f"{float((0.8 / gc[edge]).median()):.1f} | {det / base[1]:.2f} | {100 * strong:.1f}")
            t = Image.fromarray((img[y0:y1, x0:x1].clamp(0, 1) * 255).byte().cpu().numpy())
            ImageDraw.Draw(t).text((8, 6), f"{name}: {rule}", fill=(255, 255, 0))
            tiles.append(t)
        w, h = tiles[0].size
        sheet = Image.new("RGB", (len(tiles) * w + 6 * (len(tiles) - 1), h), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * (w + 6), 0))
        sheet.save(out / f"{name}.jpg", quality=92)
