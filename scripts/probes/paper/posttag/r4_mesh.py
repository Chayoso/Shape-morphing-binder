"""r4_mesh.py OUT_DIR OUT_JSONL MESH:LABEL:RAW=ARCHIVE [...] -- D145 reading 4 (2, 4): our display surface as a triangle
mesh, and the harness's apart particles classified on it (CPU; torch, scipy, skimage).

The surface is the yardstick's exterior: the zero set of Zhu and Bridson's field of the particles (radius 3, offset 0.8
pitches; pitch = 0.708 x the independent sample's median 8th-neighbour distance), the field the render loss and
visible_apart.py read, meshed by D154's zb_mesh (tmp/d154/d154_mesh_read.py, copied below unchanged): marching cubes
at half a pitch, the field exact near its zero set. RAW = a raw frame index of the archive's kept frames, `end`, `tgt`
(the archive's own target sample) or `ind` (D90's independent sample): the last two give the floor, a perfect sample.

Per state, one JSON line and OUT_DIR/MESH_LABEL_RAW_mesh.npz (V float32, F int32) for the mesh readers (r4_topo.py):
  verts, faces, the mesh's vertex-connected components and the particles under each (each particle to its nearest
  vertex); the apart particles (outside the largest set connected at 1.5 pitches, the harness's count) split into
  those whose nearest vertex lies on the main component (under the body's own display surface: a bump or hidden) and
  those on another component, with each component's size (particles), gap (its nearest vertex to the main
  component's, pitches) and kind by the sign of its enclosed volume: a solid piece the display draws apart from the
  body (> 0) or a cavity, a closed void inside the body whose surface faces in (< 0)."""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as nnf
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

B = "/data/relcfd/chayo/physmorph_v2"
sys.path.insert(0, f"{B}/repo_r77")                            # the yardstick's code, as read300.sh / D154
from physmorph.render.exterior import ZhuBridson               # noqa: E402

torch.set_grad_enabled(False)
NT = 6
torch.set_num_threads(NT)


def zb_mesh(field, x, a):
    """D154's zb_mesh: the field's zero set as a triangle mesh: marching cubes at half a pitch; the field read exactly
    at the coarse nodes (a pitch) within its radius of a particle and at the fine nodes of every coarse cell the zero
    set crosses and the ring around those; elsewhere the coarse values, trilinear (only their sign reaches the zero set)."""
    hm, hc = 0.5 * a, a
    R = field.radius
    lo = x.min(0) - (R + 2 * hc)
    nc = np.ceil((x.max(0) + (R + 2 * hc) - lo) / hc).astype(int) + 1
    gi = [np.arange(k) for k in nc]
    nodes = np.stack(np.meshgrid(*gi, indexing="ij"), -1).reshape(-1, 3)
    q = lo + hc * nodes
    near = cKDTree(x).query(q, 1, distance_upper_bound=R, workers=NT)[0] < R
    fc = np.full(len(q), R, np.float64)
    f, _, _ = field(torch.as_tensor(q[near], dtype=torch.float32))
    fc[near] = np.nan_to_num(f.numpy().astype(np.float64), posinf=R)
    fc = fc.reshape(*nc)
    s = fc > 0
    crossed = np.zeros(nc - 1, bool)
    for o in np.ndindex(2, 2, 2):
        sl = tuple(slice(oi, oi + k - 1) for oi, k in zip(o, nc))
        crossed |= s[sl] != s[tuple(slice(0, k - 1) for k in nc)]
    band = binary_dilation(crossed, structure=np.ones((3, 3, 3), bool))
    nf = 2 * (nc - 1) + 1
    ff = nnf.interpolate(torch.as_tensor(fc, dtype=torch.float32)[None, None], size=tuple(int(v) for v in nf),
                         mode="trilinear", align_corners=True)[0, 0].numpy().astype(np.float64)
    cells = np.argwhere(band)
    fine = np.unique((2 * cells[:, None, :] + np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1)
                      .reshape(1, -1, 3)).reshape(-1, 3), axis=0)
    f, _, _ = field(torch.as_tensor(lo + hm * fine, dtype=torch.float32))
    ff[fine[:, 0], fine[:, 1], fine[:, 2]] = np.nan_to_num(f.numpy().astype(np.float64), posinf=R)
    verts, faces, _, _ = marching_cubes(ff, level=0.0, spacing=(hm, hm, hm))
    return verts + lo, faces


def frame(path, raw):
    """The archive's frame at RAW, read from its place in the (uncompressed) file as D154's frame_of: not loaded."""
    import zipfile
    from numpy.lib import format as npf
    z = np.load(path, allow_pickle=True)
    if raw in ("tgt", "ind"):                  # the floor: the archive's own target sample, or D90's independent one
        if raw == "tgt":
            return np.asarray(z["tgt"], np.float64), -1
        return np.asarray(np.load(f"{B}/output/gpu/d90/{MESH_OF[path]}_ind300k.npz")["tgt"], np.float64), -2
    raws = [int(r) for r in z["raws"]]
    k = len(raws) - 1 if raw == "end" else raws.index(int(raw))
    with zipfile.ZipFile(path) as zf, zf.open("frames.npy") as f:
        version = npf.read_magic(f)
        shape, fortran, dtype = npf._read_array_header(f, version)
        row = int(np.prod(shape[1:])) * dtype.itemsize
        f.seek(f.tell() + k * row)
        x = np.frombuffer(f.read(row), dtype).reshape(shape[1:]).astype(np.float64)
    return x, raws[k]


out_dir, out_jsonl = sys.argv[1], sys.argv[2]
MESH_OF = {s.split("=", 1)[1]: s.split(":")[0] for s in sys.argv[3:]}
pitch = {}
for spec in sys.argv[3:]:
    key, path = spec.split("=", 1)
    mesh, label, raw = key.split(":")
    t0 = time.time()
    if mesh not in pitch:
        I = np.asarray(np.load(f"{B}/output/gpu/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
        pitch[mesh] = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=NT)[0][:, 8]))
    a = pitch[mesh]
    x, rawk = frame(path, raw)
    V, F = zb_mesh(ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a), x, a)
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nv, vlab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)
    vsize = np.bincount(vlab, minlength=nv)
    main = int(np.argmax(vsize))
    _, nearest = cKDTree(V).query(x, 1, workers=NT)
    plab = vlab[nearest]
    pcount = np.bincount(plab, minlength=nv)
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
                                             shape=(len(x), len(x))), directed=False)
    apart = lab != int(np.argmax(np.bincount(lab)))
    sep = apart & (plab != main)
    others = [c for c in range(nv) if c != main]
    gaps = []
    if others:
        tm = cKDTree(V[vlab == main])
        for c in others:
            gaps.append(float(tm.query(V[vlab == c], 1, workers=NT)[0].min() / a))
    # each component's signed volume (outward normals: the field is negative inside): > 0 a separate solid piece the
    # display draws, < 0 a cavity (a closed void inside the body, its surface facing in)
    flab = vlab[F[:, 0]]
    tri = V[F]
    sv = np.bincount(flab, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0, minlength=nv)
    sign_main = 1.0 if sv[main] >= 0 else -1.0                   # the marching-cubes winding, read off the main body
    solid = {c: sign_main * sv[c] > 0 for c in others}
    pieces = [c for c in others if solid[c]]
    cav = [c for c in others if not solid[c]]
    sep_piece = apart & np.isin(plab, pieces)
    row = dict(mesh=mesh, label=label, raw=rawk, archive=path, pitch=a, verts=int(n), faces=int(len(F)),
               components=int(nv), main_verts=int(vsize[main]), main_volume=float(abs(sv[main])),
               solid_pieces=len(pieces), cavities=len(cav),
               other_components=[dict(particles=int(pcount[c]), verts=int(vsize[c]), gap_a=round(g, 2),
                                      kind="piece" if solid[c] else "cavity", volume=float(sign_main * sv[c]))
                                 for c, g in sorted(zip(others, gaps), key=lambda t: -pcount[t[0]])][:50],
               particles_off_main=int((plab != main).sum()), apart=int(apart.sum()),
               apart_under_main=int((apart & (plab == main)).sum()), apart_separate=int(sep.sum()),
               apart_in_pieces=int(sep_piece.sum()),
               particles_in_pieces=int(sum(pcount[c] for c in pieces)),
               particles_at_cavities=int(sum(pcount[c] for c in cav)),
               pieces_with_particles=int(sum(1 for c in pieces if pcount[c] > 0)),
               pieces_ge10=int(sum(1 for c in pieces if pcount[c] >= 10)),
               pieces_ge10_gap5=int(sum(1 for c, g in zip(others, gaps) if solid[c] and pcount[c] >= 10 and g >= 5)),
               seconds=round(time.time() - t0, 1))
    np.savez(f"{out_dir}/{mesh}_{label}_{raw}_mesh.npz", V=V.astype(np.float32), F=F.astype(np.int32),
             tgt=np.asarray(np.load(path, allow_pickle=True, mmap_mode="r")["tgt"], np.float32))
    with open(out_jsonl, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("other_components", "archive")}), flush=True)
