"""mesh_frames.py ARCHIVE OUT_DIR [+frames=a-b] [+stride=1] [+hm=0.5] [+worker=i/n] -- the demo's per-frame surfaces:
the zero set of the display's field (Zhu and Bridson's with Solenthaler et al.'s correction, "zbs", radius 3 and offset
0.8 pitches, render_deliverable.py's +field=zbs) of each body's particles, meshed by marching cubes at +hm pitches
(0.5 by default), with the field's own gradient as vertex normals. Nothing else is done to the geometry: no smoothing,
no removal of pieces.

ARCHIVE: a morph's kept frames (frames12: frames, raws, tgt) or a scene archive (frames, raws, rows, tgt0.., floor_y);
each body (row) is meshed from its own particles at its own volume sample's pitch (render_scene.py's rule), so two
bodies in contact stay two surfaces. Output per kept frame k and body b: OUT_DIR/kKKKK_bB.ply (binary little endian;
float x y z nx ny nz, int vertex_indices), world coordinates of the archive (y up), and OUT_DIR/meta.json (pitch,
floor, per-frame bounds). +worker=i/n meshes every n-th frame from i (several CPU processes on one card)."""
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_freeze1015")
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from numpy.lib import format as npf                            # noqa: E402
from skimage.measure import marching_cubes                     # noqa: E402
from physmorph.render.exterior import make_field              # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from scripts.render_scene import base_discs                    # noqa: E402  (the paper display's discs and normals)

torch.set_grad_enabled(False)
dev = torch.device("cuda")
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
src, out = argv[1], Path(argv[2])
out.mkdir(parents=True, exist_ok=True)
HM = float(opts.get("hm", 0.5))
STRIDE = int(opts.get("stride", 1))
wi, wn = (int(v) for v in opts.get("worker", "0/1").split("/"))
torch.set_num_threads(4)

z = np.load(src, allow_pickle=True)
raws = [int(r) for r in z["raws"]]
rows = [tuple(int(v) for v in r) for r in z["rows"]] if "rows" in z.files else None
floor_y = float(z["floor_y"]) if "floor_y" in z.files else None
with zipfile.ZipFile(src) as zf, zf.open("frames.npy") as f:
    version = npf.read_magic(f)
    shape, fortran, dtype = npf._read_array_header(f, version)
    header = f.tell()
    info = zf.getinfo("frames.npy")
row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
# the uncompressed member's data offset in the zip file: local header (30 bytes + name + extra) + the npy header
with open(src, "rb") as fh:
    fh.seek(info.header_offset)
    lh = fh.read(30)
    data0 = info.header_offset + 30 + int.from_bytes(lh[26:28], "little") + int.from_bytes(lh[28:30], "little")
assert info.compress_type == 0, "frames.npy must be stored uncompressed"


def frame(k):
    with open(src, "rb") as fh:
        fh.seek(data0 + header + k * row_bytes)
        return np.frombuffer(fh.read(row_bytes), dtype).reshape(shape[1:]).astype(np.float32)


pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731
if rows is None:
    rows = [(0, shape[1])]
    tgts = [z["tgt"]]
else:
    tgts = [z[f"tgt{b}"] for b in range(len(rows))]
pitches = [pitch(torch.as_tensor(np.asarray(t, np.float32), device=dev)) for t in tgts]

keep = list(range(0, len(raws), STRIDE))
if "frames" in opts:
    f0, f1 = (int(v) for v in opts["frames"].split("-"))
    keep = [k for k in keep if f0 <= k <= f1]
mine = keep[wi::wn]


def surface(x, a):
    """The zbs field's zero set of the particles x (cuda) at pitch a: the field at the nodes of a grid of HM pitches
    within the field's radius of a particle (+radius elsewhere: no particle there, outside), marching cubes, and the
    field's gradient at the vertices as normals."""
    field = make_field("zbs", x, a, radius=3.0, offset=0.8, device_field=True)
    R = field.radius
    h = HM * a
    lo = x.min(0).values - (R + 2 * h)
    n = ((x.max(0).values + (R + 2 * h) - lo) / h).ceil().long() + 1
    nx, ny, nz = (int(v) for v in n)
    occ = torch.zeros(nx, ny, nz, device=dev)
    ij = ((x - lo) / h).round().long()
    occ[ij[:, 0], ij[:, 1], ij[:, 2]] = 1.0
    r = int(np.ceil(R / h)) + 1
    near = nnf.max_pool3d(occ[None, None], 2 * r + 1, stride=1, padding=r)[0, 0] > 0
    idx = near.nonzero()
    vals = torch.full((nx, ny, nz), R, device=dev)
    for c in idx.split(4_000_000):
        f, _, _ = field(lo + h * c.float(), grad=False)
        vals[c[:, 0], c[:, 1], c[:, 2]] = torch.nan_to_num(f.float(), nan=R, posinf=R).clamp(max=R)
    V, F, _, _ = marching_cubes(vals.cpu().numpy(), level=0.0, spacing=(h, h, h), allow_degenerate=False)
    V = V.astype(np.float32) + lo.cpu().numpy()
    Vt = torch.as_tensor(V, device=dev)
    g = []
    for c in Vt.split(2_000_000):
        _, gc, _ = field(c, grad=True)
        g.append(gc.float())
    Nf = nnf.normalize(torch.cat(g), dim=1)
    # the shading normals: the paper display's (render_scene.base_discs: the exterior's discs on a lattice of 0.92
    # pitches, the field's normals averaged twice over the reach), carried to each vertex from its 4 nearest discs
    # (inverse-distance weights); the field's own gradient where no disc is within a pitch. Geometry unchanged.
    p, n = base_discs(x, a, "zbs", 0.92 * a)
    dd, nn = gpu.KNN(p).query(Vt, 4)
    w = (1.0 / dd.float().clamp_min(1e-3 * a))[..., None]
    Nd = nnf.normalize((n[nn] * w).sum(1), dim=1)
    far = dd[:, 0].float() > a
    Nd[far] = Nf[far]
    N = Nd.cpu().numpy().astype(np.float32)
    # marching cubes' winding faces the larger values (outside, the field is positive there): keep it outward for the
    # renderer by flipping when the mean of N . face normal is negative
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if float(np.einsum("ij,ij->", fn, Nf.cpu().numpy()[F[:, 0]])) < 0:
        F = F[:, ::-1]
    return V, np.ascontiguousarray(F.astype(np.int32)), N


def write_ply(path, V, N, F):
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n"
            f"property float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face {len(F)}\n"
            "property list uchar int vertex_indices\nend_header\n").encode()
    vb = np.concatenate([V, N], 1).astype("<f4").tobytes()
    fb = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    fb["n"], fb["i"] = 3, F
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        fh.write(head)
        fh.write(vb)
        fh.write(fb.tobytes())
    tmp.rename(path)


if wi == 0:
    (out / "meta.json").write_text(json.dumps(dict(src=src, raws=raws, keep=keep, rows=rows, pitches=pitches,
                                                   floor_y=floor_y, hm=HM)))
for k in mine:
    t0 = time.time()
    x_all = frame(k)
    for b, (s, e) in enumerate(rows):
        p = out / f"k{k:04d}_b{b}.ply"
        if p.exists():
            continue
        x = torch.as_tensor(x_all[s:e], device=dev)
        V, F, N = surface(x, pitches[b])
        write_ply(p, V, N, F)
        (out / f"k{k:04d}_b{b}.json").write_text(json.dumps(dict(lo=V.min(0).tolist(), hi=V.max(0).tolist(), ylow=float(np.percentile(V[:, 1], 0.05)),
                                                                 verts=len(V), faces=len(F))))
    print(f"frame {k} raw {raws[k]} {time.time() - t0:.1f}s faces {len(F)}", flush=True)
