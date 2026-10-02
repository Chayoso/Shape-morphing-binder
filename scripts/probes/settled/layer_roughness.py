"""layer_roughness.py ARCHIVE_NPZ RUN_JSON TARGET_OBJ — D16: how rough the outermost particle layer is, and at what scale.

For the source sample (the first frame), the run's target sample and the morph's last delivered frame, on the
outermost layer only (the pipeline's layer rule):
  plane residual: what the layer relaxation removes: d - dbar, with d the particle's normal offset from the weighted
    centroid of its 24 same-side layer neighbours (weight width 2 layer spacings) and dbar the neighbours' mean of d;
  offset from the target mesh (target sample and last frame only): the signed distance s of every layer particle to
    the target mesh fitted to the target sample, split by scale with Gaussian means over the layer: at most about 4
    spacings (s - G1 s), about 4 to 11 spacings (G1 s - G2.5 s), and larger (G2.5 s minus its mean). The target's own
    relief is in the mesh, so it is not counted; a bounding-box misfit of the mesh is smooth and falls in the last band.
All in particle spacings a = (V / N)^(1/3). One line of JSON at the end for the gallery table."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import trimesh                                                 # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
prov = run["provenance"]
a = float(prov["mpm"]["dx"]) / float(prov["ppc"]) ** (1 / 3)
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
clouds = {"source": torch.as_tensor(np.asarray(frames[0], np.float32), device=dev), "target": tgt,
          "end": torch.as_tensor(np.asarray(frames[n_del - 1], np.float32), device=dev)}

mesh = load_mesh(sys.argv[3])
mesh.merge_vertices()
o = orient_name(sys.argv[3])
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
t_np = tgt.cpu().numpy().astype(np.float64)
vb = np.asarray(mesh.bounds, np.float64)
step = float(knn_self_torch(tgt, 7)[0][:, 6].median())
mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * float(np.mean((t_np.max(0) - t_np.min(0) + step) / (vb[1] - vb[0]))) \
    + 0.5 * (t_np.max(0) + t_np.min(0))


def rms(v):
    return float(v.double().pow(2).mean().sqrt())


def gmean(P, s, sigma):
    """Gaussian-weighted mean of s over each layer point's 64 nearest layer points (itself included)."""
    d, i = knn_self_torch(P, min(65, len(P)))
    w = torch.exp(-0.5 * (d / sigma) ** 2)
    return (w * s[i]).sum(1) / w.sum(1)


out = {"a": a}
for name, x in clouds.items():
    sp0 = layer_spacing(x)
    mask, nrm, nbr, w = layer_relax_data(x, sp0, k=24, h_sp=2.0)
    L = mask > 0.5
    c = (w[..., None] * x[nbr]).sum(1)
    d = ((x - c) * nrm).sum(1)
    dbar = (w * d[nbr]).sum(1)
    rec = {"layer": int(L.sum()), "layer_share": float(L.float().mean()), "plane_resid": rms(d[L]) / a, "relaxed_part": rms((d - dbar)[L]) / a}
    if name != "source":
        P = x[L]
        closest, dist, tri = trimesh.proximity.closest_point(mesh, P.cpu().numpy().astype(np.float64))
        sign = np.sign(np.einsum("ij,ij->i", P.cpu().numpy() - closest, mesh.face_normals[tri]))
        s = torch.as_tensor(dist * np.where(sign == 0, 1.0, sign), dtype=torch.float32, device=dev) / a
        g1, g25 = gmean(P, s, 1.0 * a), gmean(P, s, 2.5 * a)
        rec.update(offset_rms=rms(s - s.mean()), high=rms(s - g1), mid=rms(g1 - g25), low=rms(g25 - g25.mean()), offset_mean=float(s.mean()))
    out[name] = rec
    print(f"{name:6s}: layer {rec['layer']} ({100 * rec['layer_share']:.1f} %) | plane residual {rec['plane_resid']:.3f} sp, the part the relaxation removes {rec['relaxed_part']:.3f} sp"
          + (f" | offset from the mesh: <= 4 sp {rec['high']:.3f}, 4-11 sp {rec['mid']:.3f}, larger {rec['low']:.3f} (mean {rec['offset_mean']:+.2f}) sp" if name != "source" else ""))
print("JSON " + json.dumps(out))
