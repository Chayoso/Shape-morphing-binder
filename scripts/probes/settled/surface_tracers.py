"""surface_tracers.py ARCHIVE_NPZ SOURCE_OBJ TARGET_OBJ OUT_DIR [RAW,RAW,...] [x0,y0,x1,y1 ...] — D13: the source's
surface carried by the archived motion, with no new simulation.

The source mesh is fitted to the source sample, subdivided (164k and 655k vertices), and its vertices are moved
through the delivered frames as massless tracers: over each stretch of frames a tracer takes the displacement of the
particles around it (an affine least-squares fit over its 24 nearest particles, Gaussian weights of 1.5 lattice
steps). The particles' displacements are the simulation's own motion sampled at 300k points, so nothing in the
physics changes. The carried mesh is then drawn exactly as D12's reference (three million samples of the surface with
the faces' normals, the same rasteriser, camera and material) and measured against the target mesh drawn the same
way: pixel-normal angle, silhouette edge width, shading detail, silhouette IoU. Also: how far the tracers are from
the particles (and the particles' outer layer from the tracers), the area stretch of the tracer triangles, and the
folding of the carried mesh (the angle between neighbouring faces)."""
import math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
import trimesh                                                 # noqa: E402
from PIL import Image, ImageDraw                               # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import StudioRaster               # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

W, H, AZ, EL = 3840, 2160, 35.0, 18.0
dev = torch.device("cuda")
archive, src_path, tgt_path, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
raws = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] else []
boxes = [[int(v) for v in b.split(",")] for b in sys.argv[6:]] or [[1800, 60, 2700, 860]]
out.mkdir(parents=True, exist_ok=True)
z = np.load(archive, allow_pickle=True)
frames = z["frames"]
n_del = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
last = n_del - 1
raws = sorted({min(r, last) for r in raws} | {last})
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
studio = StudioRaster(center, radius, W, H, AZ, EL)
px = 2 * 3.6 * radius * math.tan(math.radians(15)) / H


def X(raw):
    return torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)


def fitted(path, cloud):
    """The mesh in the sample's frame: the sample's bounding box is the mesh's, inset by half a lattice step a side."""
    mesh = load_mesh(path)
    mesh.merge_vertices()                                      # one vertex per corner, so the subdivided mesh is closed
    o = orient_name(path)
    if o != "id":
        mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
    c = cloud.cpu().numpy().astype(np.float64)
    vb = np.asarray(mesh.bounds, np.float64)
    step = float(knn_self_torch(cloud, 7)[0][:, 6].median())
    s = float(np.mean((c.max(0) - c.min(0) + step) / (vb[1] - vb[0])))
    mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * s + 0.5 * (c.max(0) + c.min(0))
    return mesh, step


def discs(normals, sigma):
    ref = torch.where(normals[:, :1].abs() < .9, normals.new_tensor((1., 0., 0.)), normals.new_tensor((0., 1., 0.))).expand_as(normals)
    tangent = nnf.normalize(torch.linalg.cross(normals, ref), dim=1, eps=1e-9)
    rot = torch.stack((tangent, torch.linalg.cross(normals, tangent), normals), dim=2)
    var = torch.stack((sigma ** 2, sigma ** 2, (sigma / 4) ** 2), dim=1)
    return (rot * var[:, None]) @ rot.transpose(1, 2)


def draw_mesh(mesh, n=3_000_000):
    """A surface drawn as D12's reference: n samples of the surface, the faces' normals, discs of two sample spacings."""
    pts, face = mesh.sample(n, return_index=True)
    x = torch.as_tensor(np.asarray(pts, np.float32), device=dev)
    nr = nnf.normalize(torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev), dim=1, eps=1e-9)
    sp = float(knn_self_torch(x[::10], 2)[0][:, 1].median()) / math.sqrt(10)
    with torch.inference_mode():
        return studio(x, nr, discs(nr, torch.full((len(x),), 2.0 * sp, device=dev)), torch.full((len(x),), .92, device=dev),
                      normal_kernel=1, return_buffers=True)


def grad_mag(f):
    gy, gx = torch.gradient(f)
    return (gx ** 2 + gy ** 2).sqrt()


def measures(pic, ref):
    (img, cov, nrm), (rimg, rcov, rnrm) = pic, ref
    lum = lambda i: i @ i.new_tensor((.2126, .7152, .0722))
    both = (cov > .9) & (rcov > .9)
    ang = torch.rad2deg(torch.acos((nrm[both] * rnrm[both]).sum(-1).clamp(-1, 1)))
    gc = grad_mag(cov)
    edge = (cov > .4) & (cov < .6) & (gc > 1e-4)
    a, b = cov >= .5, rcov >= .5
    return (f"normal error {float(ang.mean()):5.1f} deg (p90 {float(torch.quantile(ang[::max(1, len(ang) // 500000)], .9)):5.1f}) | edge {float((0.8 / gc[edge]).median()):4.1f} px | "
            f"detail {float(grad_mag(lum(img))[cov > .99].mean() / grad_mag(lum(rimg))[rcov > .99].mean()):.2f} of the reference | silhouette IoU {float((a & b).sum() / (a | b).sum()):.4f}")


def tile(img, label, box):
    t = Image.fromarray((img * 255 + .5).byte().cpu().numpy()).crop(box)
    ImageDraw.Draw(t).text((8, 6), label, fill=(255, 255, 0))
    return t


def sheet(tiles, name):
    w, h = tiles[0].size
    s = Image.new("RGB", (len(tiles) * w + 6 * (len(tiles) - 1), h), (20, 20, 20))
    for i, t in enumerate(tiles):
        s.paste(t, (i * (w + 6), 0))
    s.save(out / f"{name}.jpg", quality=92)


x0 = X(0)
src_mesh, step = fitted(src_path, x0)
tgt_mesh, _ = fitted(tgt_path, tgt)
print(f"N {len(x0)}, delivered frames {n_del}; lattice step {step:.4f} wu = {step / px:.1f} px at 4K; source mesh {len(src_mesh.vertices)} vertices, {len(src_mesh.faces)} faces")
ref = draw_mesh(tgt_mesh)

# the tracer meshes: the source mesh subdivided (flat faces stay flat)
levels = []
for sub in (6, 7):
    v, f = np.asarray(src_mesh.vertices, np.float64), np.asarray(src_mesh.faces)
    for _ in range(sub):
        v, f = trimesh.remesh.subdivide(v, f)
    levels.append((torch.as_tensor(v, dtype=torch.float32, device=dev), f))
    print(f"tracer mesh, {sub} subdivisions: {len(v)} vertices, {len(f)} faces, edge about {float(np.sqrt(src_mesh.area / len(f) * 4 / math.sqrt(3))) / px:.1f} px at 4K")
T = torch.cat([v for v, _ in levels])
split = [len(v) for v, _ in levels]
area0 = [trimesh.Trimesh(v.cpu().numpy(), f, process=False).area_faces for v, f in levels]

# the stretches of frames: short while the body moves fast, long once it is settling
stops, f = [0], 0
while f < last:
    f = min(last, f + (4 if f < 800 else 8 if f < 2400 else 20))
    stops.append(f)
stops = sorted(set(stops) | set(raws))
print(f"{len(stops) - 1} stretches of frames; tracers {len(T)}")
K, HW = 24, 1.5 * step
t0 = time.time()
xa = x0
saved = {}
for i in range(1, len(stops)):
    xb = X(stops[i])
    d, idx = gpu.KNN(xa).query(T, K)
    P, D = xa[idx], (xb - xa)[idx]                              # (M, K, 3)
    d = d.float()
    h = torch.maximum(d[:, -1:], d.new_tensor(HW))             # never narrower than the neighbourhood itself
    w = torch.exp(-(d / h) ** 2)[..., None]
    w = w / w.sum(1, keepdim=True)
    pc, dc = (w * P).sum(1), (w * D).sum(1)
    Pc, Dc = P - pc[:, None], D - dc[:, None]
    M = (w * Pc).transpose(1, 2) @ Pc
    B = (w * Dc).transpose(1, 2) @ Pc
    M = M + 1e-3 * (M.diagonal(dim1=1, dim2=2).sum(1) / 3)[:, None, None] * torch.eye(3, device=dev)
    A = B @ torch.linalg.inv(M)
    # the fit holds inside the neighbourhood: a tracer beyond it takes the fit at the neighbourhood's edge
    off = T - pc
    reach = Pc.norm(dim=2).max(1).values
    off = off * (reach / off.norm(dim=1).clamp_min(1e-12)).clamp(max=1.)[:, None]
    T = T + dc + (A @ off[..., None])[..., 0]
    xa = xb
    if stops[i] in raws:
        saved[stops[i]] = T.clone()
    if i % 50 == 0 or i == len(stops) - 1:
        dn = gpu.KNN(xa).query(T, 1)[0][:, 0].float() / step
        print(f"   stretch {i:4d} -> raw {stops[i]:5d} ({time.time() - t0:5.0f} s): tracer to nearest particle, lattice steps: median {float(dn.median()):.2f}, p99 {float(torch.quantile(dn[::8], .99)):.2f}, max {float(dn.max()):.1f}", flush=True)
np.savez_compressed(out / "tracers.npz", **{f"raw_{r}": t.cpu().numpy() for r, t in saved.items()}, split=np.array(split),
                    faces_0=levels[0][1], faces_1=levels[1][1])


def outer(x):
    d, i = knn_self_torch(x, 34)
    return (x[i[:, 1:]].mean(1) - x).norm(dim=1) > 0.35 * d[:, -1]


for raw in raws:
    xr = X(raw)
    print(f"\n== raw frame {raw}")
    tiles = [[tile(ref[0], "reference: the target mesh", b)] for b in boxes]
    p = out.parent / "d12_dragon" / f"morph_frame_{raw}.png"
    if p.exists():
        for k, b in enumerate(boxes):
            tiles[k].append(Image.open(p).convert("RGB").crop(b))
            ImageDraw.Draw(tiles[k][-1]).text((8, 6), "the particles, the display rule as it is", fill=(255, 255, 0))
    lo = 0
    for (v0, f), n, a0 in zip(levels, split, area0):
        v = saved[raw][lo: lo + n]
        lo += n
        mesh = trimesh.Trimesh(v.cpu().numpy().astype(np.float64), f, process=False)
        dn = gpu.KNN(xr).query(v, 1)[0][:, 0].float() / step
        po = xr[outer(xr)]
        dp = gpu.KNN(v).query(po, 1)[0][:, 0].float() / step
        ratio = torch.as_tensor(mesh.area_faces / a0)
        fold = np.degrees(mesh.face_adjacency_angles)
        edge = np.linalg.norm(mesh.vertices[mesh.edges_unique[:, 0]] - mesh.vertices[mesh.edges_unique[:, 1]], axis=1) / px
        # a face with a vertex more than two lattice steps from every particle spans a gap the material has left:
        # it is not drawn (the particle display's support rule does the same to a particle without neighbours)
        has = (dn[torch.as_tensor(f, device=dev)] <= 2).all(1).cpu().numpy()
        pic = draw_mesh(trimesh.Trimesh(mesh.vertices, f[has], process=False))
        print(f"   faces not drawn (a vertex beyond two lattice steps of every particle): {100 * (1 - has.mean()):.2f} %, {100 * mesh.area_faces[~has].sum() / mesh.area:.1f} % of the carried area")
        print(f"   tracers {n}: to the nearest particle (lattice steps) median {float(dn.median()):.2f}, p99 {float(torch.quantile(dn[::4], .99)):.2f}, max {float(dn.max()):.1f} | "
              f"particles' outer layer to the nearest tracer median {float(dp.median()):.2f}, p99 {float(torch.quantile(dp, .99)):.2f}, beyond 2 steps {100 * float((dp > 2).float().mean()):.1f} %")
        print(f"      triangle area against the start: median {float(ratio.median()):.2f}, p90 {float(torch.quantile(ratio[::4].float(), .9)):.1f}, p99 {float(torch.quantile(ratio[::4].float(), .99)):.1f}, max {float(ratio.max()):.0f}; total area x {mesh.area / float(np.sum(a0)):.2f} | "
              f"edge length px: median {np.median(edge):.1f}, p99 {np.percentile(edge, 99):.1f} | angle between neighbouring faces: median {np.median(fold):.1f} deg, p99 {np.percentile(fold, 99):.0f}, above 90 deg {100 * np.mean(fold > 90):.2f} %")
        print("      picture: " + measures(pic, ref))
        Image.fromarray((pic[0] * 255 + .5).byte().cpu().numpy()).save(out / f"tracers_{n}_raw_{raw}.png")
        for k, b in enumerate(boxes):
            tiles[k].append(tile(pic[0], f"the source's surface carried by the motion, {n} tracers", b))
    for k in range(len(boxes)):
        sheet(tiles[k], f"sheet_raw_{raw}_box{k}")
