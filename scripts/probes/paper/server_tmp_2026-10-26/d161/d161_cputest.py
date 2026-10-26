"""D161 CPU smoke test of d161_lib (scratch): frame, surface, patterns, the field's divergence, the relief reading."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
import numpy as np
from scipy.spatial import cKDTree
from d161_lib import Stripes, Surface, Zonal, displacement, read_relief, target_frame
from physmorph.sampling.mesh import draws_in_frame

for mesh, path in (("sphere", "assets/isosphere.obj"), ("bunny", "assets/bunny.obj")):
    t0 = time.time()
    frame, xt = target_frame("assets/isosphere.obj", path, 40000, 97)
    X0 = draws_in_frame(path, frame, 40000, [99])[0].astype(np.float64)
    d8 = cKDTree(xt).query(xt[:5000], 9)[0][:, 8]
    a = 0.708 * float(np.median(d8))
    S = Surface(path, frame, 400000, xt.astype(np.float64))
    c = S.P.mean(0)
    R = float(np.linalg.norm(S.P - c, axis=1).mean())
    print(mesh, "frame", {k: (float(v) if np.ndim(v) == 0 else np.round(v, 3).tolist()) for k, v in frame.items()},
          "pitch", round(a, 4), "R", round(R, 3), "ext", np.round(xt.max(0) - xt.min(0), 2).tolist(), round(time.time() - t0, 1), "s")
    if mesh == "sphere":
        pat = Zonal(c, np.array([0, 0, 1.0]), R, int(round(2 * np.pi * R / (11 * a))))
    else:
        xc = xt.mean(0); ext = xt.max(0) - xt.min(0)
        lat = np.eye(3)[0] if ext[0] < ext[2] else np.eye(3)[2]
        fwd = np.eye(3)[2] if ext[0] < ext[2] else np.eye(3)[0]
        band = np.abs(S.P[:, 1] - xc[1]) < 0.5
        cf = S.P[band][np.argmax((S.P[band] - xc) @ lat)]
        print("flank", np.round(cf, 3).tolist(), "lat", lat.tolist())
        pat = Stripes(fwd, 2 * np.pi / (11 * a), cf, 1.8)
    P, Gs, W = pat.at(S)
    f0 = S.foot(X0)
    Phi = displacement(X0, S, P, Gs, pat.k, f0)
    # divergence of the field by the change in local number density: count within 1.5 a before / after
    for amp in (0.5, 1.0, 2.0):
        X1 = X0 + amp * a * Phi
        sel = (np.abs(f0[1]) < 6 * a) & (W[f0[0]] > 0.5) & (f0[1] > 1.0 * a)
        c0 = cKDTree(X0).query_ball_point(X0[sel][:3000], 1.5 * a, return_length=True)
        c1 = cKDTree(X1).query_ball_point(X1[sel][:3000], 1.5 * a, return_length=True)
        rel = c1 / np.maximum(c0, 1) - 1
        # a crude outer layer: within 0.7 a of the surface
        lay0 = f0[1] < 0.7 * a
        r = read_relief(X1[lay0], S, P, W, reach=3 * a)
        print(f"  amp {amp}: density change rms {np.sqrt((rel**2).mean()):.4f} corr(P) {np.corrcoef(rel, P[f0[0]][sel][:3000])[0,1]:+.3f}; "
              f"layer relief A/a {r['A']/a:.3f} corr {r['corr']:.3f} n {r['n']}")
    print("  wavelength", round(pat.wavelength / a, 2), "pitches; |Phi| max", round(float(np.linalg.norm(Phi, axis=1).max()), 3))
