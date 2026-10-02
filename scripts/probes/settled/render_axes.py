"""render_axes.py ARCHIVE_NPZ MESH_OBJ OUT_DIR [RAW,RAW,...] [x0,y0,x1,y1] — D12: which part of the 4K display renderer
blurs the picture, measured without a simulation.

The reference is the target mesh itself: three million surface samples with the mesh's own face normals, drawn as
small discs by the same rasteriser, camera and material. Against it, the display renderer's rule
(scripts/render_splat_photoreal.py: disc radius = target spacing x clamp(8th-neighbour distance / coverage radius, 1,
4); normals = gradient of a density field blurred over `blur` spacings, replaced where weak, averaged `passes` times
over 32 neighbours; a `kernel`-pixel image filter on the normal buffer) is drawn on
  the run's target sample (300k volume particles), with one part of the rule changed at a time;
  morph frames of the archive, with the same changes (a smaller disc may open the stretched sheets);
  denser volume samples of the same mesh (the sampling's share), with the rule unchanged.
Per picture: the angle between its pixel normals and the reference's (where both cover the pixel), the width of the
silhouette edge (0.8 / |grad coverage| on the half-coverage contour, pixels), the interior shading detail (mean
luminance gradient where the coverage is full, as a share of the reference's), the silhouette's IoU with the
reference, and the covered pixels lost against the unchanged rule on the same particles. A sheet of crops is saved
per object."""
import math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals, StudioRaster  # noqa: E402
from physmorph.render.support import live_support              # noqa: E402
from physmorph.sampling.mesh import load_mesh, sample_volume_stratified  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
archive, mesh_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
raws = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
box = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [1800, 60, 2700, 860]
out.mkdir(parents=True, exist_ok=True)
z = np.load(archive, allow_pickle=True)
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
n_del = int(z["deliver_n"]) if "deliver_n" in z.files else len(z["frames"])
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
studio = StudioRaster(center, radius, W, H, AZ, EL)


def scales(x):
    d = knn_self_torch(x, 9)[0]
    return float(d[:, 1].median()), float(d[:, 8].median())


def discs(normals, sigma):
    ref = torch.where(normals[:, :1].abs() < .9, normals.new_tensor((1., 0., 0.)), normals.new_tensor((0., 1., 0.))).expand_as(normals)
    tangent = nnf.normalize(torch.linalg.cross(normals, ref), dim=1, eps=1e-9)
    rot = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    var = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    return (rot * var[:, None]) @ rot.transpose(1, 2)


def draw(x, spacing, cov_r, sigma_scale=1.0, blur=3.0, passes=2, kernel=3):
    """The display renderer's primitives (render_splat_photoreal.py, the unpinned path), with one part changed."""
    with torch.inference_mode():
        d, nb = knn_self_torch(x, 33)
        support = live_support(d, cov_r, spacing)
        sigma = sigma_scale * spacing * (d[:, 8] / cov_r).clamp(1., 4.)
        normals, mag = DensityNormals(center, radius, spacing, blur=blur)(x)
        strong = mag >= torch.quantile(mag[::max(1, len(x) // 100000)], .6)
        near = nb[:, 1:33]
        sn = strong[near]
        chosen = near[torch.arange(len(x), device=dev), sn.float().argmax(1)]
        normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
        for _ in range(passes):
            normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
        return studio(x, normals, discs(normals, sigma), .92 * support, normal_kernel=kernel, return_buffers=True)


def luminance(img):
    return img @ img.new_tensor((.2126, .7152, .0722))


def grad_mag(f):
    gy, gx = torch.gradient(f)
    return (gx ** 2 + gy ** 2).sqrt()


def measures(img, cov, nrm, ref, base_cov=None):
    rimg, rcov, rnrm = ref
    both = (cov > .9) & (rcov > .9)
    ang = torch.rad2deg(torch.acos((nrm[both] * rnrm[both]).sum(-1).clamp(-1, 1)))
    gc = grad_mag(cov)
    edge = (cov > .4) & (cov < .6) & (gc > 1e-4)
    full, rfull = cov > .99, rcov > .99
    a, b = cov >= .5, rcov >= .5
    m = dict(normal_err=float(ang.mean()), normal_err_p90=float(torch.quantile(ang[::max(1, len(ang) // 500000)], .9)),
             edge_px=float((0.8 / gc[edge]).median()), detail=float(grad_mag(luminance(img))[full].mean() / grad_mag(luminance(rimg))[rfull].mean()),
             iou=float((a & b).sum() / (a | b).sum()))
    if base_cov is not None:
        bb = base_cov >= .5
        m["lost"] = 100 * float((bb & ~a).sum() / bb.sum())
    return m


def save(img, name):
    Image.fromarray((img * 255 + .5).byte().cpu().numpy()).save(out / f"{name}.png")


def tile(img, label):
    t = Image.fromarray((img * 255 + .5).byte().cpu().numpy()).crop(box)
    ImageDraw.Draw(t).text((8, 6), label, fill=(255, 255, 0))
    return t


def sheet(tiles, name, cols=3):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (cols * w + 6 * (cols - 1), rows * h + 6 * (rows - 1)), (20, 20, 20))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    s.save(out / f"{name}.jpg", quality=92)


# the mesh in the target sample's frame: the sample's bounding box is the mesh's, inset by half a nominal spacing a side
mesh = load_mesh(mesh_path)
o = orient_name(mesh_path)
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
t_np = tgt.cpu().numpy().astype(np.float64)
vb = np.asarray(mesh.bounds, np.float64)
ext_t = t_np.max(0) - t_np.min(0)
# nominal spacing from the sample itself: the median distance to the 6th neighbour of a jittered lattice is about one lattice step
sp_t, cov_t = scales(tgt)
lattice = float(knn_self_torch(tgt, 7)[0][:, 6].median())
s_fit = float(np.mean((ext_t + lattice) / (vb[1] - vb[0])))
mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * s_fit + 0.5 * (t_np.max(0) + t_np.min(0))
print(f"target sample: {len(tgt)} particles, nn spacing {sp_t:.4f} wu, coverage radius {cov_t:.4f} wu, lattice step about {lattice:.4f} wu; "
      f"mesh fitted by bounding box (scale {s_fit:.4f}; per-axis ratios {np.round((ext_t + lattice) / (vb[1] - vb[0]), 4).tolist()})")
px = 2 * 3.6 * radius * math.tan(math.radians(15)) / H
print(f"one 4K pixel at the object's centre is {px:.5f} wu: nn spacing {sp_t / px:.1f} px, lattice step {lattice / px:.1f} px, density blur of the normals {3 * sp_t / px:.0f} px (sigma)")

pts, face = mesh.sample(3_000_000, return_index=True)
rx = torch.as_tensor(np.asarray(pts, np.float32), device=dev)
rn = nnf.normalize(torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev), dim=1)
r_sp = float(knn_self_torch(rx[::10], 2)[0][:, 1].median()) / math.sqrt(10)
with torch.inference_mode():
    ref = studio(rx, rn, discs(rn, torch.full((len(rx),), 2.0 * r_sp, device=dev)), torch.full((len(rx),), .92, device=dev), normal_kernel=1, return_buffers=True)
save(ref[0], "reference_mesh")
print(f"reference: 3,000,000 surface samples, spacing {r_sp:.4f} wu ({r_sp / px:.1f} px), disc radius {2 * r_sp / px:.1f} px, the mesh's face normals")

VARIANTS = [("the rule as it is", {}), ("disc x 0.7", dict(sigma_scale=.7)), ("disc x 0.5", dict(sigma_scale=.5)),
            ("normal blur 1.5 sp", dict(blur=1.5)), ("normal blur 1.5 sp, no averaging", dict(blur=1.5, passes=0)),
            ("normal blur 0.75 sp, no averaging", dict(blur=.75, passes=0)), ("no averaging only", dict(passes=0)),
            ("no image filter", dict(kernel=1)), ("disc x 0.7, blur 1.5 sp, no averaging", dict(sigma_scale=.7, blur=1.5, passes=0))]
HEAD = "normal error deg (mean p90) | edge px | detail (share of the reference) | silhouette IoU with the reference | covered pixels lost %"


def study(x, name, spacing, cov_r, variants):
    print(f"\n== {name}: {len(x)} particles; disc radius {spacing / px:.1f} px, normal blur {3 * spacing / px:.0f} px\n   variant | {HEAD}")
    tiles, base = [tile(ref[0], "reference: the mesh")], None
    for label, kw in variants:
        img, cov, nrm = draw(x, spacing, cov_r, **kw)
        m = measures(img, cov, nrm, ref, base)
        if base is None:
            base = cov
        print(f"   {label:40s} | {m['normal_err']:5.1f} {m['normal_err_p90']:5.1f} | {m['edge_px']:5.1f} | {m['detail']:.2f} | {m['iou']:.4f} | " + (f"{m['lost']:.2f}" if 'lost' in m else "-"))
        tiles.append(tile(img, label))
        if not kw:
            save(img, name.replace(" ", "_"))
    sheet(tiles, "sheet_" + name.replace(" ", "_"))


study(tgt, "target 300k", sp_t, cov_t, VARIANTS)
frames = z["frames"]
for raw in raws:
    raw = min(raw, n_del - 1)
    study(torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev), f"morph frame {raw}", sp_t, cov_t, VARIANTS)
del frames, z
for n in (1_200_000, 2_400_000):
    xd = torch.as_tensor(sample_volume_stratified(mesh, n, seed=98).astype(np.float32), device=dev)
    sp_d, cov_d = scales(xd)
    study(xd, f"target {n // 1000}k", sp_d, cov_d, VARIANTS[:1] + [VARIANTS[4]])
