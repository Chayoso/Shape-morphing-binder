"""d161_features.py OUT_JSON MESH [MESH ...] -- D161: the real fine features of the target meshes (the dragon's scales, the
bunny's wrinkles) as amplitude and wavelength in pitches at 40k and 300k, so that they can be placed on the response
curve. CPU only (numpy, scipy, trimesh).

Each mesh is put in its 40k run frame (d161_lib.target_frame, seed 97); the pitch a(N) = 0.708 x the median 8th-neighbour
distance of the run's target sample at N (40k and 300k, the cached samples). The mesh's surface is sampled
area-uniformly (2M points) with face normals, smoothed over 16 neighbours (d161_lib.Surface).
Relief bands as D107 / D158 read them: h_s(y) = n(y) . (y - ybar_s(y)), ybar_s the Gaussian mean (sigma s) of the
surface's points around y (truncated at 2 s, K nearest of a population thinned so that the K-th lies near 2 s); the band
between s and 2s is h_2s - h_s minus its own Gaussian mean at 4s (the mean-curvature bias (4 s^2 - s^2) H / 2, smooth
at the scale of the curvature, leaves; relief is locally zero-mean). sigma = 0.25, 0.5, 1, 2, 4, 8 pitches of 300k
(nominal wavelengths 5.4 sigma: 1.35 ... 43 pitches of 300k). Per band, over 60k query points: the rms and the 90th
and 99th percentiles of |band| in pitches of 40k and 300k, and the same over the query points in the top decile of the
band's local energy (where the features are: the scales, the wrinkles). The mesh's own resolution: its median edge
length in pitches (relief below about twice it is not in the mesh)."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d161_lib import Surface, mesh_in_frame, target_frame  # noqa: E402

NT = 8
SIG = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
K = 64


def pitch_of(x):
    sub = x[np.random.default_rng(0).choice(len(x), min(20000, len(x)), replace=False)]
    d = cKDTree(sub).query(sub, 9, workers=NT)[0][:, 8]
    return 0.708 * float(np.median(d)) * (len(sub) / len(x)) ** (1.0 / 3.0)


def gauss_mean(Q, P, area, s, rng):
    """The Gaussian mean (sigma s, truncated at 2 s) of the points P around each query, P thinned to put K
    neighbours within 2 s."""
    want = int(K * area / (np.pi * (2 * s) ** 2))
    Ps = P if want >= len(P) else P[rng.choice(len(P), want, replace=False)]
    d, i = cKDTree(Ps).query(Q, K, workers=NT)
    w = np.exp(-0.5 * (d / s) ** 2) * (d < 2 * s)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (w[..., None] * Ps[i]).sum(1)


def smooth_scalar(Q, v, s):
    d, i = cKDTree(Q).query(Q, K, workers=NT)
    w = np.exp(-0.5 * (d / s) ** 2) * (d < 2 * s)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (w * v[i]).sum(1)


rows = []
for name in sys.argv[2:]:
    path = f"assets/{name}.obj"
    frame, x40 = target_frame("assets/isosphere.obj", path, 40000, 97)
    a40 = pitch_of(x40)
    frame3, x300 = target_frame("assets/isosphere.obj", path, 300000, 97)
    a300 = pitch_of(x300) * frame["scale"] / frame3["scale"]       # in the 40k frame's lengths
    S = Surface(path, frame, 2_000_000, x40.astype(np.float64))
    V, F = mesh_in_frame(path, frame)
    tm = trimesh.Trimesh(V, F, process=False)
    edge = float(np.median(tm.edges_unique_length))
    area = float(tm.area)
    rng = np.random.default_rng(1)
    qi = rng.choice(S.n, 60000, replace=False)
    Q, Nq = S.P[qi], S.N[qi]
    h = {}
    for s in SIG:
        sw = s * a300
        h[s] = ((Q - gauss_mean(Q, S.P, area, sw, rng)) * Nq).sum(1)
    bands = []
    for s0, s1 in zip(SIG[:-1], SIG[1:]):
        b = h[s1] - h[s0]
        b = b - smooth_scalar(Q, b, 4 * s1 * a300)
        e_loc = smooth_scalar(Q, b * b, 2 * s1 * a300)
        top = e_loc >= np.quantile(e_loc, 0.9)
        ab = np.abs(b)
        bands.append(dict(sigma_300k=[s0, s1], wavelength_300k=[5.4 * s0, 5.4 * s1],
                          wavelength_40k=[5.4 * s0 * a300 / a40, 5.4 * s1 * a300 / a40],
                          rms_300k=float(np.sqrt((b * b).mean()) / a300), rms_40k=float(np.sqrt((b * b).mean()) / a40),
                          p90_300k=float(np.quantile(ab, 0.9) / a300), p99_300k=float(np.quantile(ab, 0.99) / a300),
                          top_rms_300k=float(np.sqrt((b[top] ** 2).mean()) / a300),
                          top_rms_40k=float(np.sqrt((b[top] ** 2).mean()) / a40),
                          top_p90_300k=float(np.quantile(ab[top], 0.9) / a300)))
    row = dict(mesh=name, pitch40k=a40, pitch300k=a300, edge_wu=edge, edge_pitch300k=edge / a300,
               edge_pitch40k=edge / a40, area_wu2=area, bands=bands)
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
