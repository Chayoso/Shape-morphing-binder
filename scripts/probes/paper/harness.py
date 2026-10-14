"""harness.py OUT_JSONL MESH IND_NPZ LABEL=FRAMES_NPZ [...] -- D145's method-agnostic reading of morph trajectories.

Every method's kept frames (an npz with frames K x N x 3 and raws, the pipeline's archive format) are read on ONE fixed
grid per mesh, against the independent sample I (IND_NPZ, key tgt):
  progress  p_k = 1 - S(x_k, I) / S(x_0, I), S the debiased Sinkhorn divergence of the pipeline's own grid transport
            (physmorph.losses.grid_ot.GridSinkhornLoss, blur one cell) on a grid around I and the first method's
            first frame, the joint box padded by a quarter of its diagonal, cell = that box's diagonal / 96; points
            outside the grid are clamped onto its border and counted (escaped share);
  at the first frames reaching p = 0.25, 0.5, 0.75 and at the last frame:
            apart  = particles outside the largest connected set at 1.5 pitches (pitch = 0.708 x the median
                     8th-neighbour distance of I, as d120_apart.py),
            occupied volume = cells of 1.5 pitches holding at least one particle, against I's own count.
One JSON line per method with the p curve (raw, p), the selected frames and their measures; the selected raws are
also printed for the display's reading (surface_layer_probe) by harness.sh."""
import json
import sys

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from physmorph.losses.grid_ot import GridSinkhornLoss
from physmorph.losses.volumetric import rasterize_mass

out_path, mesh, ind_path = sys.argv[1], sys.argv[2], sys.argv[3]
specs = [s.split("=", 1) for s in sys.argv[4:]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
I = np.asarray(np.load(ind_path)["tgt"], np.float32)
pitch = 0.708 * float(np.median(cKDTree(I).query(I, 9)[0][:, 8]))

first = np.asarray(np.load(specs[0][1], allow_pickle=True, mmap_mode="r")["frames"][0], np.float32)
lo, hi = np.minimum(I.min(0), first.min(0)), np.maximum(I.max(0), first.max(0))
diag = float(np.linalg.norm(hi - lo))
lo, hi = lo - 0.25 * diag, hi + 0.25 * diag
dx = float(np.linalg.norm(hi - lo)) / 96.0
dims = tuple(int(np.ceil((h - l) / dx)) + 1 for l, h in zip(lo, hi))
grid_min = torch.tensor(lo, dtype=torch.float32, device=dev)
box_hi = lo + dx * (np.array(dims) - 1) - 1e-4 * dx


def grid_of(x):
    xc = np.clip(x, lo, box_hi)
    escaped = float(np.mean(np.any(xc != x, axis=1)))
    xt = torch.tensor(xc, dtype=torch.float32, device=dev)
    m = torch.full((len(xt),), 1.0 / len(xt), dtype=torch.float32, device=dev)
    return rasterize_mass(xt, m, grid_min, dx, dims), escaped


target, _ = grid_of(I)
ot = GridSinkhornLoss(target, grid_min, dx, dims, eps=dx * dx, iters=4000, tol=1e-3)


def apart(x):
    pairs = cKDTree(x).query_pairs(1.5 * pitch, output_type="ndarray")
    n = len(x)
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return int(n - np.bincount(lab).max())


def occupied(x):
    return int(len(np.unique(np.floor((x - lo) / (1.5 * pitch)).astype(np.int64), axis=0)))


occ_I = occupied(I)
with open(out_path, "a") as out:
    for label, path in specs:
        z = np.load(path, allow_pickle=True, mmap_mode="r")
        frames, raws = z["frames"], np.asarray(z["raws"]).astype(int)
        S, esc = [], []
        for k in range(len(frames)):
            gk, e = grid_of(np.asarray(frames[k], np.float32))
            with torch.no_grad():
                S.append(float(ot(gk)))
            esc.append(e)
        S = np.array(S)
        p = 1.0 - S / S[0]
        picks = {}
        for q in (0.25, 0.5, 0.75):
            hit = np.nonzero(p >= q)[0]
            picks[f"p{q:g}"] = int(hit[0]) if len(hit) else None
        picks["end"] = len(frames) - 1
        read = {}
        for key, k in picks.items():
            if k is None:
                read[key] = None
                continue
            x = np.asarray(frames[k], np.float32)
            na = apart(x)
            read[key] = dict(frame=k, raw=int(raws[k]), tau=float(raws[k] / max(raws[-1], 1)), p=float(p[k]),
                             apart=na, apart_share=na / len(x), occupied_ratio=occupied(x) / occ_I,
                             escaped=esc[k])
        row = dict(label=label, mesh=mesh, frames=path, n=int(frames.shape[1]), pitch=pitch, grid=list(dims), dx=dx,
                   S0=float(S[0]), S_end=float(S[-1]), p_end=float(p[-1]), escaped_max=float(max(esc)),
                   curve=[[int(r), float(v)] for r, v in zip(raws, p)], read=read)
        out.write(json.dumps(row) + "\n")
        out.flush()
        sel = " ".join(str(read[k]["raw"]) for k in ("p0.25", "p0.5", "p0.75", "end") if read[k] is not None)
        print(f"{label}: p_end {p[-1]:.4f}, S0 {S[0]:.3e}; " + ", ".join(
            f"{k} raw {v['raw']} apart {v['apart']} occ {v['occupied_ratio']:.3f}" for k, v in read.items() if v)
            + f"; escaped max {max(esc):.2e}", flush=True)
        print(f"RAWS {label} {sel}", flush=True)
