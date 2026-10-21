"""B9b forensic (CPU): is the standing target statically stable on the floor, and where is the body's centre of mass
against its support during the run? Support = the particles within one pitch of the lowest centre (their convex hull
in the floor plane); the margin = the distance from the centroid's floor projection to the hull's boundary (positive
inside), in pitches."""
import json, sys
import numpy as np
from scipy.spatial import ConvexHull

def margin(x, pitch):
    low = x[x[:, 1] <= x[:, 1].min() + pitch][:, [0, 2]]
    c = x.mean(0)[[0, 2]]
    if len(low) < 3:
        return -np.inf, len(low)
    h = ConvexHull(low)
    # hull equations: n.x + d <= 0 inside; the distance inside = -max(n.c + d)
    d = -(h.equations[:, :2] @ c + h.equations[:, 2]).max()
    return d / pitch, len(low)

R = sys.argv[1]
for tag in sys.argv[2:]:
    z = np.load(f"{R}/{tag}.npz")
    j = json.load(open(f"{R}/{tag}.json"))
    fr, raws = z["frames"], z["raws"]
    floor_y = float(z["floor_y"])
    pitch = 2.0 * (float(fr[0][:, 1].min()) - floor_y)
    tg = np.asarray(z["tgt0"], np.float64)
    m, n = margin(tg, pitch)
    print(f"{tag}: pitch {pitch:.4f}; target standing: centroid {m:+.2f} pitches inside its support ({n} support particles; "
          f"its centroid {tg[:,1].mean() - tg[:,1].min():.3f} wu above its lowest point)")
    hist = {h["animation"]: h for h in j["history"] if "bodies" in h}
    for k in range(0, len(fr), max(1, len(fr) // 25)):
        x = np.asarray(fr[k], np.float64)
        m, n = margin(x, pitch)
        t = raws[k] / 240.0
        w = hist.get(int(t * 6) - 1)
        tilt = np.degrees(np.arccos(np.clip(w["bodies"][0]["R"][4], -1, 1))) if w else 0.0
        print(f"  t {t:5.2f}: centroid {m:+6.2f} pitches inside the support ({n:5d} particles), height {x.mean(0)[1] - floor_y:.3f} wu, tilt {tilt:5.1f}")
