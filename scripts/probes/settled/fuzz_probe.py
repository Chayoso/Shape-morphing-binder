"""fuzz_probe.py ARCHIVE_NPZ OUT_DIR x0,y0,x1,y1 [RENDER_STRIDE [CENSUS_STRIDE]] — D57: the tufts of loose material near
thin features: the particles or the display, on every frame. Measurement only; the display is the base renderer's
(scripts/render_splat_photoreal.py: live support, sigma = spacing x clamp(r8 / target r8, 1, 4), density normals).

The particles, every CENSUS_STRIDE-th simulated frame (default 1), census.csv: the rendered particles (live support
above zero); of them the detached ones (not linked to the body within one layer spacing, single linkage over the 32
nearest neighbours; the body is the largest set), their sets, the median and 90 % distance of a detached particle to
the nearest particle of the body in target spacings; the particles the display draws enlarged (factor r8 / target r8
above 1.5), how many of those are detached, the mean factor of the detached and of the body's surface particles; the
detached particles' distance to the target (median) and how many of them are within the berth of 1.97 target spacings.
The surface's own sampling: the body's surface particles (neighbourhood asymmetry of half a coverage radius, not
detached), their spacing ON the surface (the 8th smallest in-plane distance among the 32 nearest, against the same on
the target's surface: S), the share of them with S above 1.5 (too few particles to cover that piece of surface), and the
same counts on the thin part (nearest target point's feature thickness below 2 MPM cells of DX).

The display, every RENDER_STRIDE-th frame (default 12, the video's frames), render.csv and crops/: three drawings with
the renderer's coverage buffer, 4K, the video's camera: `base` as the base draws it; `flat` every disc at the target
spacing (no enlargement); `body` the detached particles left out. Per drawing the solid pixels (coverage >= 0.5) and
the soft pixels (0.02 <= coverage < 0.5) of the whole picture and of the crop x0,y0,x1,y1; the pixels that are solid
only by the enlargement (solid in base, not in flat: surface the particles do not cover at their own size) and only by
the detached particles (solid in base, not in body); the crop of each drawing is saved (crops/NNNN.jpg: base | flat |
body). The same three drawings of the target's own sample give the level of a finished surface (target.json,
crops/target.jpg). The MPM cell DX is read from the run's log beside the archive."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.pipeline.window.layer import layer_spacing      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support, normal_filter_size  # noqa: E402
from physmorph.thin import local_thickness                     # noqa: E402

dev = torch.device("cuda")
out = Path(sys.argv[2]); (out / "crops").mkdir(parents=True, exist_ok=True)
box = [int(v) for v in sys.argv[3].split(",")]
r_stride = int(sys.argv[4]) if len(sys.argv) > 4 else 12
c_stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
W, H = 3840, 2160
z = np.load(sys.argv[1], allow_pickle=True)
frames, count = z["frames"], int(z["deliver_n"])
target = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center = target.mean(0)
radius = float((target - center).norm(dim=1).max())
td = knn_self_torch(target, 9)[0]
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
density_normals = DensityNormals(center, radius, sp)
studio = StudioRaster(center, radius, W, H, 35., 18.)
kernel = normal_filter_size(H, False)
tknn = gpu.KNN(target)
suffix = "_render_full_dt_iso_nn.npz"
log = Path(sys.argv[1][:-len(suffix)] + ".log") if sys.argv[1].endswith(suffix) else None
dx = (float(sys.argv[6]) if len(sys.argv) > 6 else
      float(re.search(r"\| dx=([0-9.]+) dt=", log.read_text()).group(1)) if log is not None and log.exists() else None)
thin_t = (local_thickness(target, sp) / dx < 2.) if dx else torch.zeros(len(target), dtype=torch.bool, device=dev)


def normals_of(x, nb):
    """The display's normals: the smoothed density gradient, carried to weak-gradient particles by their neighbours."""
    normals, magnitude = density_normals(x)
    strong = magnitude >= torch.quantile(magnitude[::max(1, len(x) // 100000)], .6)
    nearest = nb[:, 1:33]
    sn = strong[nearest]
    chosen = nearest[torch.arange(len(x), device=dev), sn.float().argmax(1)]
    normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
    for _ in range(2):
        normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
    return normals


def surface_spacing(x, nb, normals, k=8):
    """The k-th smallest distance to the 32 nearest neighbours measured in the particle's tangent plane."""
    offsets = x[nb[:, 1:]] - x[:, None, :]
    planar = (offsets - (offsets * normals[:, None, :]).sum(-1, keepdim=True) * normals[:, None, :]).norm(dim=-1)
    return planar.kthvalue(k, dim=1).values


def surface_of(x, nb):
    return (x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r           # neighbourhood asymmetry


with torch.inference_mode():
    _d, _nb = knn_self_torch(target, 33)
    _surf = surface_of(target, _nb)
    S_ref = float(surface_spacing(target, _nb, normals_of(target, _nb))[_surf].median())
    n_surf_t, n_surf_thin_t = int(_surf.sum()), int((_surf & thin_t).sum())


def groups(d, nb, a):
    """(detached (N,) bool, number of detached sets): single linkage at distance a over the listed neighbours; the
    body is the largest set."""
    N = d.shape[0]
    ok = d[:, 1:] < a
    i = torch.arange(N, device=d.device)[:, None].expand(-1, d.shape[1] - 1)[ok]
    j = nb[:, 1:][ok]
    label = torch.arange(N, device=d.device)
    while True:
        new = label.clone()
        new.scatter_reduce_(0, i, label[j], "amin")
        new.scatter_reduce_(0, j, label[i], "amin")
        new = new[new]
        if bool((new == label).all()):
            break
        label = new
    ids, inv, cnt = label.unique(return_inverse=True, return_counts=True)
    return inv != cnt.argmax(), len(ids) - 1


def census(x):
    d, nb = knn_self_torch(x, 33)
    shown = live_support(d, cov_r, sp) > 0
    factor = (d[:, 8] / cov_r).clamp(1., 4.)
    det, n_sets = groups(d, nb, layer_spacing(x))
    normals = normals_of(x, nb)
    surf = surface_of(x, nb) & ~det & shown                               # the body's own surface particles
    S = surface_spacing(x, nb, normals) / S_ref
    thin = thin_t[tknn.query(x, 1)[1][:, 0]]                              # by the nearest target point's thickness
    r = det & shown
    row = dict(rendered=int(shown.sum()), detached=int(r.sum()), sets=n_sets, enlarged=int((shown & (factor > 1.5)).sum()),
               enlarged_detached=int((r & (factor > 1.5)).sum()), factor_detached=float(factor[r].mean()) if bool(r.any()) else 1.,
               factor_surface=float(factor[surf].mean()),
               surface=int(surf.sum()), surface_sparse=int((surf & (S > 1.5)).sum()), surface_S_median=float(S[surf].median()),
               surface_thin=int((surf & thin).sum()), surface_thin_sparse=int((surf & thin & (S > 1.5)).sum()),
               surface_thin_S_median=float(S[surf & thin].median()) if bool((surf & thin).any()) else float("nan"),
               enlarged_surface=int((surf & (factor > 1.5)).sum()), target_surface=n_surf_t, target_surface_thin=n_surf_thin_t)
    if bool(r.any()):
        gap = gpu.KNN(x[~det]).query(x[r], 1)[0][:, 0].float() / sp
        dt = tknn.query(x[r], 1)[0][:, 0].float() / sp
        row.update(gap_median=float(gap.median()), gap_p90=float(gap.quantile(.9)), to_target_median=float(dt.median()),
                   within_berth=int((dt <= 1.97).sum()))
    return row, (d, nb, det, normals)


def draw(x, known):
    """The three drawings (picture, coverage buffer) from a frame's census."""
    d, nb, det, normals = known
    support = live_support(d, cov_r, sp)
    reference = torch.where(normals[:, :1].abs() < .9, x.new_tensor((1., 0., 0.)), x.new_tensor((0., 1., 0.))).expand_as(x)
    tangent = nnf.normalize(torch.linalg.cross(normals, reference), dim=1, eps=1e-9)
    rotation = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)

    def picture(sigma, opacity):
        variance = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
        covariance = (rotation * variance[:, None]) @ rotation.transpose(1, 2)
        image, coverage, _ = studio(x, normals, covariance, opacity, normal_kernel=kernel, return_buffers=True)
        return image, coverage
    enlarged = sp * (d[:, 8] / cov_r).clamp(1., 4.)
    return {"base": picture(enlarged, .92 * support), "flat": picture(torch.full_like(enlarged, sp), .92 * support),
            "body": picture(enlarged, .92 * support * (~det).float())}


def pixels(pics):
    """Per drawing the solid and soft pixels, whole picture and crop; and the pixels solid in base only."""
    out_, crop = {}, lambda c: c[box[1]:box[3], box[0]:box[2]]          # noqa: E731
    for name in pics:
        c = pics[name][1]
        out_.update({f"{name}_solid": int((c >= .5).sum()), f"{name}_soft": int(((c >= .02) & (c < .5)).sum()),
                     f"{name}_crop_solid": int((crop(c) >= .5).sum()), f"{name}_crop_soft": int(((crop(c) >= .02) & (crop(c) < .5)).sum())})
    base = pics["base"][1] >= .5
    for key, name in (("only_enlarged", "flat"), ("only_detached", "body")):
        only = base & (pics[name][1] < .5)
        out_.update({key: int(only.sum()), f"{key}_crop": int(crop(only).sum())})
    return out_


def crop_sheet(pics, label, path):
    tiles = []
    for name in ("base", "flat", "body"):
        im = Image.fromarray((pics[name][0][box[1]:box[3], box[0]:box[2]] * 255 + .5).byte().cpu().numpy())
        ImageDraw.Draw(im).text((8, 6), f"{label}  {name}", fill=(255, 255, 0))
        tiles.append(im)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (3 * w + 8, h), (20, 20, 20))
    for k, t in enumerate(tiles):
        sheet.paste(t, (k * (w + 4), 0))
    sheet.save(path, quality=88)


with torch.inference_mode():
    row, known = census(target)
    pics = draw(target, known)
    ref = dict(pixels=pixels(pics), census=row, S_ref_sp=S_ref / sp, dx=dx)
    crop_sheet(pics, "target sample", out / "crops" / "target.jpg")
    (out / "target.json").write_text(json.dumps(ref, indent=1))
    print("target sample:", json.dumps(ref), flush=True)
    c_rows, r_rows = [], []
    for raw in range(count):
        on_c, on_r = raw % c_stride == 0 or raw == count - 1, raw % r_stride == 0 or raw == count - 1
        if not (on_c or on_r):
            continue
        x = torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)
        row, known = census(x)
        c_rows.append(dict(raw=raw, window=raw / 40., **row))
        if on_r:
            pics = draw(x, known)
            r_rows.append(dict(raw=raw, window=raw / 40., **pixels(pics)))
            crop_sheet(pics, f"frame {len(r_rows) - 1} (window {raw / 40.:.1f})", out / "crops" / f"{len(r_rows) - 1:04d}.jpg")
            if len(r_rows) % 40 == 1:
                print(f"raw {raw} of {count}: {json.dumps({**row, **r_rows[-1]})}", flush=True)
for name, rows in (("census", c_rows), ("render", r_rows)):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("raw", "window"), k))
    (out / f"{name}.csv").write_text(",".join(keys) + "\n" + "\n".join(",".join(str(r.get(k, "")) for k in keys) for r in rows) + "\n")
print("wrote", out, len(c_rows), "census rows,", len(r_rows), "render rows")
