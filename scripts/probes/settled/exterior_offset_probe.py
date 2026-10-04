"""exterior_offset_probe.py FRAMES_NPZ TARGET_OBJ [STATE ...] — D66: where the drawn surface of a state departs from
the target mesh, and at what scale.

For the exterior's discs of a state (physmorph/render/exterior.py; one disc per crossed cell of a lattice of 0.92
pitches, the pitch a the volume sample's): the signed distance s of each disc to the target mesh (the mesh fitted to
the target sample by its bounding box and sampled at three million surface points with their face normals; discs
farther than two pitches from every sample are left out and counted: the mesh's open base, closed sets inside; the
distance is taken to the plane of the nearest sample), in pitches, split by Gaussian means over the discs into the
part below about 4 pitches (s - G1 s), between about 4 and 11 pitches (G1 s - G2.5 s) and larger (G2.5 s minus its
mean); and the angle between the mesh's normal there and the disc's normal: the field's gradient, and the displayed
normal (the base display's treatment: the mean over the discs within the reach of its 32 nearest particles, twice).
Also the mesh's own relief in the two finer bands at each disc, and the least-squares slope and the correlation of the
disc's offset on it: a surface that lost the mesh's relief of a band has slope -1 there, one that only carries lumps 0.
STATE: `target` (the file's target sample) or a raw frame index kept in FRAMES_NPZ (default: target and every kept
frame); `save=DIR` among them keeps each state's discs with their offsets there. One JSON line per state."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
import trimesh                                                 # noqa: E402
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
raws, frames = [int(v) for v in z["raws"]], z["frames"]
keep = next((s[5:] for s in sys.argv[3:] if s.startswith("save=")), None)      # save=DIR: the discs of each state, for a closer look
states = [s for s in sys.argv[3:] if not s.startswith("save=")] or ["target"] + [str(r) for r in raws]
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
cov_r = float(td[:, 8].median())
a = .708 * cov_r                                               # the volume sample's pitch
center = tgt.mean(0)
lattice = Lattice(center, 2.8 * float((tgt - center).norm(dim=1).max()))
h = .92 * a

# the target mesh, fitted to the target sample as layer_roughness.py fits it, as surface samples with normals
mesh = load_mesh(sys.argv[2])
mesh.merge_vertices()
o = orient_name(sys.argv[2])
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
t_np = tgt.cpu().numpy().astype(np.float64)
vb = np.asarray(mesh.bounds, np.float64)
step = float(knn_self_torch(tgt, 7)[0][:, 6].median())
mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * float(np.mean((t_np.max(0) - t_np.min(0) + step) / (vb[1] - vb[0]))) \
    + 0.5 * (t_np.max(0) + t_np.min(0))
points, face = trimesh.sample.sample_surface(mesh, 3000000, seed=0)
surface = torch.as_tensor(np.asarray(points, np.float32), device=dev)
surface_normal = torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev)
surface_tree = gpu.KNN(surface)
# the mesh's own relief in the two finer bands, on every 30th surface sample: the sample's height above the Gaussian means
# of the samples around it, along its normal (what a surface that lost that band would be short of)
coarse, coarse_normal = surface[::30].contiguous(), surface_normal[::30].contiguous()
coarse_tree = gpu.KNN(coarse)


def relief(sigma, k):
    d, i = coarse_tree.query(coarse, k)
    w = torch.exp(-.5 * (d.float() / sigma) ** 2)
    return (((w[..., None] * coarse[i]).sum(1) / w.sum(1, keepdim=True) - coarse) * coarse_normal).sum(1) / a


relief1, relief25 = -relief(1. * a, 64), -relief(2.5 * a, 256)


def rms(v):
    return float(v.double().pow(2).mean().sqrt())


def gmean(P, s, sigma, k):
    """Gaussian-weighted mean of s over each disc's k nearest discs (itself included)."""
    d, i = gpu.KNN(P).query(P, k)
    w = torch.exp(-.5 * (d.float() / sigma) ** 2)
    return (w * s[i]).sum(1) / w.sum(1)


def angle(n, m):
    return torch.rad2deg(torch.acos((n * m).sum(1).clamp(-1., 1.)))


print(f"N {len(tgt)}; pitch a {a:.4f} wu; lattice {h / a:.2f} a; mesh {len(mesh.faces)} faces, {len(surface)} surface samples")
with torch.no_grad():
    for name in states:
        x = tgt if name == "target" else torch.as_tensor(np.asarray(frames[raws.index(int(name))], np.float32), device=dev)
        P, g, _, _ = lattice.discs(ZhuBridson(x, a), h, refine=False)
        n = nnf.normalize(g, dim=1)
        at = surface_tree.query(P, 1)[1].reshape(-1)
        near = (P - surface[at]).norm(dim=1) < 2. * a          # the discs the mesh has a surface for (not its open base,
        apart = 1. - float(near.float().mean())                 #   nor closed sets inside the body)
        outside = float((((P - surface[at]) * surface_normal[at]).sum(1)[~near] > 0.).float().mean()) if apart > 0. else 0.
        P, n, at = P[near], n[near], at[near]
        m = surface_normal[at]
        s = ((P - surface[at]) * m).sum(1) / a
        g1, g25 = gmean(P, s, 1. * a, 64), gmean(P, s, 2.5 * a, 256)
        near_coarse = coarse_tree.query(P, 1)[1].reshape(-1)
        r_high, r_mid = relief1[near_coarse], (relief25 - relief1)[near_coarse]       # the mesh's relief below 4 pitches, 4 to 11
        fit = lambda s_, r_: (float((s_ * r_).mean() / r_.square().mean().clamp_min(1e-12)),                       # noqa: E731
                              float(torch.corrcoef(torch.stack((s_, r_)))[0, 1]))
        d, nb = knn_self_torch(x, 33)
        reach = float(d[(x - x[nb[:, 1:]].mean(1)).norm(dim=1) >= .5 * cov_r, 32].median())
        dn, nbn = gpu.KNN(P).query(P, 96)
        w = (dn.float() <= reach)[..., None]
        shown = n
        for _ in range(2):
            shown = nnf.normalize((shown[nbn] * w).sum(1), dim=1, eps=1e-9)
        row = dict(state=name, discs=len(P), apart=apart, apart_outside=outside,     # the share of those on the mesh's outer side
                   offset=dict(mean=float(s.mean()), rms=rms(s - s.mean()), high=rms(s - g1), mid=rms(g1 - g25), low=rms(g25 - g25.mean())),
                   mesh_relief=dict(high=rms(r_high), mid=rms(r_mid), high_fit=fit(s - g1, r_high), mid_fit=fit(g1 - g25, r_mid)),
                   normal_error=dict(field=rms(angle(n, m)), shown=rms(angle(shown, m)),
                                     field_median=float(angle(n, m).median()), shown_median=float(angle(shown, m).median())))
        if keep:
            np.savez(Path(keep) / f"{Path(sys.argv[1]).stem}_{name}.npz", points=P.cpu().numpy(), normals=n.cpu().numpy(), mesh_normals=m.cpu().numpy(),
                     s=s.cpu().numpy(), g1=g1.cpu().numpy(), g25=g25.cpu().numpy(), a=a)
        print(json.dumps(row), flush=True)
