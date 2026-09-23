"""Interior cavities per archived frame, with the photoreal renderer's density and level.

A marching-cubes component whose signed volume has the sign opposite to the body's is a closed
surface around a void INSIDE the material (its normals face the void) — not a piece of the body,
invisible from outside. The photoreal sidecar counts them itself from commit c916743+; this sweep
writes the same per-frame count for videos rendered before that, so the report's "drawn pieces"
never counts a hollow ear as a floating piece (bunny at 150k: 37 frames).

    cavity_sweep.py <out_dir> <run> <stride> <out_txt> [grid] [blur]
per line: archived_frame n_cavities n_outer_components ; summary line with '#'.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as Fn
from skimage import measure

OUT, RUN, stride, out_txt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
G = int(sys.argv[5]) if len(sys.argv) > 5 else 160
blur = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(f"{OUT}/{RUN}_render_full_dt_iso_nn.npz")
frames = z["frames"]; tgt = z["tgt"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
allp = np.concatenate([frames[:dn:max(1, dn // 40)].reshape(-1, 3), tgt], 0)
lo, hi = allp.min(0), allp.max(0)
ctr = torch.as_tensor((lo + hi) / 2, device=dev, dtype=torch.float32)
half = float((hi - lo).max()) * 0.55
vox = 2 * half / G
x0 = torch.as_tensor(frames[0], device=dev)
N = len(x0); n_sub = min(N, 20000)
sub = x0[torch.randperm(N, device=dev)[:n_sub]]
d8 = torch.cdist(sub, sub).topk(9, largest=False).values[:, -1]
spacing = float(d8.median()) * (n_sub / N) ** (1.0 / 3.0)
sig_vox = max(0.6, blur * spacing / vox)


def density(x):
    p = (x - (ctr - half)) / vox - 0.5
    i0 = torch.floor(p).long(); f = p - i0.float()
    rho = torch.zeros(G * G * G, device=dev)
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                w = ((f[:, 0] if dx_ else 1 - f[:, 0]) * (f[:, 1] if dy_ else 1 - f[:, 1]) * (f[:, 2] if dz_ else 1 - f[:, 2]))
                i = i0 + torch.tensor([dx_, dy_, dz_], device=dev)
                ok = ((i >= 0) & (i < G)).all(1)
                idx = (i[ok, 2] * G + i[ok, 1]) * G + i[ok, 0]
                rho.index_put_((idx,), w[ok], accumulate=True)
    rho = rho.view(1, 1, G, G, G)
    r = int(3 * sig_vox)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sig_vox * sig_vox)); k = k / k.sum()
    rho = Fn.conv3d(rho, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
    rho = Fn.conv3d(rho, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
    rho = Fn.conv3d(rho, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
    return rho[0, 0]


rho0 = density(x0); occ = rho0[rho0 > 0]; bulk = float(occ.median())
iso = min(0.5, 2.0 * spacing ** 2 / (np.pi * (sig_vox * vox) ** 2)) * bulk
import open3d as o3d
rows = []
for i in range(0, dn, stride):
    rho = density(torch.as_tensor(frames[i], device=dev)).cpu().numpy()
    if float(rho.max()) <= iso:
        rows.append((i, 0, 0)); continue
    v, f, _, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v[:, ::-1].astype(np.float64)), o3d.utility.Vector3iVector(f[:, ::-1].astype(np.int32)))
    comp = np.asarray(m.cluster_connected_triangles()[0]); n = int(comp.max()) + 1
    vv = v[:, ::-1].astype(np.float64); ff = f[:, ::-1]
    tet = np.einsum("ij,ij->i", vv[ff[:, 0]], np.cross(vv[ff[:, 1]], vv[ff[:, 2]])) / 6.0
    svol = np.bincount(comp, weights=tet, minlength=n)
    bs = np.sign(svol[int(np.argmax(np.abs(svol)))])
    n_cav = int(((np.sign(svol) == -bs) & (svol != 0)).sum())
    rows.append((i, n_cav, n - n_cav))
with open(out_txt, "w") as fh:
    fh.write("archived_frame n_cavities n_outer_components\n")
    for r in rows:
        fh.write(f"{r[0]} {r[1]} {r[2]}\n")
    fh.write(f"# frames {len(rows)}  cavities in {sum(1 for r in rows if r[1] > 0)} frames (max {max(r[1] for r in rows)})  iso {iso / bulk:.3f} x bulk\n")
print(f"{RUN}: frames {len(rows)}  cavities in {sum(1 for r in rows if r[1] > 0)} frames (max {max(r[1] for r in rows)})")
