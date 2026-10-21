"""fx_tilt.py RUNS_DIR TAG [TAG ...] -- B9b option 1 forensic (CPU): why a standing body tilts. Per kept frame: the
shape's tilt (tobody.shape_rotation, warm-started from the previous frame's) and its axis, the material (Buckens) tilt,
the centroid's height against the standing target (pitches), the floor contact (centres within one pitch of the floor's
centre plane: count, their centroid) and the lever (the body centroid's horizontal offset from the contact centroid,
pitches, along the tilt's lean direction), the rigid angular speed between kept frames; per window from the record: the
angular velocity, the angular momentum's change by update (grid / u / spacing / relaxation; horizontal), the optimiser's
accepted iterations and line-search trials, the transport and silhouette terms. Prints every 0.25 s and a summary."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tobody import polar, shape_rotation, rigid  # noqa: E402

R_ = Path(sys.argv[1])
for tag in sys.argv[2:]:
    j = json.loads((R_ / f"{tag}.json").read_text())
    z = np.load(R_ / f"{tag}.npz")
    fr, raws = z["frames"], z["raws"]
    dt = j["mpm"]["dt"]
    floor_y = float(z["floor_y"])
    X = np.asarray(z["src0"], np.float64); X -= X.mean(0)
    tgt = np.asarray(z["tgt0"], np.float64); tc = tgt - tgt.mean(0); tree = cKDTree(tc)
    floor_c = float(fr[0][:, 1].min()); pitch = 2 * (floor_c - floor_y)
    h_t = float(tgt[:, 1].mean() - tgt[:, 1].min())
    hist = {h["animation"]: h for h in j["history"] if "bodies" in h}
    Rs = None
    rows = []
    for k in range(len(fr)):
        x = np.asarray(fr[k], np.float64); c = x.mean(0); xc = x - c
        Rb = polar(xc.T @ X)
        Rs = shape_rotation(xc, tc, tree, Rb if Rs is None or k < 8 else Rs)
        up = Rs[:, 1]
        lean = np.array([up[0], up[2]]); ln = np.linalg.norm(lean)
        con = x[x[:, 1] <= floor_c + pitch]
        lever = (c[[0, 2]] - con[:, [0, 2]].mean(0)) / pitch if len(con) else np.full(2, np.nan)
        w = np.nan
        if k + 1 < len(fr):
            v = (np.asarray(fr[k + 1], np.float64) - x) / ((raws[k + 1] - raws[k]) * dt)
            w = float(np.linalg.norm(rigid(x, v)[1]))
        rows.append(dict(t=raws[k] * dt, shape=float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))),
                         mat=float(np.degrees(np.arccos(np.clip(Rb[1, 1], -1, 1)))), dh=(c[1] - floor_c - h_t) / pitch,
                         ncon=len(con), lever_along=float(lever @ (lean / ln)) if ln > 1e-9 else 0.0,
                         lever=float(np.linalg.norm(lever)), lean_dir=float(np.degrees(np.arctan2(lean[1], lean[0]))), w=w))
    print(f"== {tag}: pitch {pitch:.4f}")
    nxt = 0.0
    for r in rows:
        if r["t"] + 1e-9 >= nxt:
            a = int(round(r["t"] / (2 * j["config"]["T"] * dt))) - 1
            h = hist.get(max(a, 0), {})
            b = (h.get("bodies") or [{}])[0]
            s = (h.get("L_split") or [None])[0]
            ls = "" if not s else " L(grid,u,space,relax) horiz " + ",".join(
                f"{np.hypot(s['own'][q][0], s['own'][q][2]):.3g}" for q in ("L_grid", "L_u", "L_space", "L_relax"))
            print(f"  t {r['t']:5.2f} shape tilt {r['shape']:5.2f} (lean dir {r['lean_dir']:+6.1f}) material {r['mat']:5.2f}  "
                  f"height {r['dh']:+6.2f} sp  contact {r['ncon']:5d}  lever {r['lever']:5.2f} sp (along the lean "
                  f"{r['lever_along']:+5.2f})  |w| {r['w']:.3f}  | win {a + 1}: acc {h.get('accepted')} ls {h.get('ls_trials')}"
                  f" tr {b.get('transport', float('nan')):.2e} sil {b.get('d_sil', float('nan')):.2e}{ls}")
            nxt += 0.25
    sh = np.array([r["shape"] for r in rows]); ts = np.array([r["t"] for r in rows])
    late = ts >= 2.0
    print(f"  summary: shape tilt at 1 / 2 / 4 / 6 / end s: " + " / ".join(f"{np.interp(q, ts, sh):.2f}" for q in (1, 2, 4, 6, ts[-1]))
          + f"; slope after 2 s {np.polyfit(ts[late], sh[late], 1)[0]:+.3f} deg/s; largest drop "
          f"{min(r['dh'] for r in rows):+.2f} sp at {rows[int(np.argmin([r['dh'] for r in rows]))]['t']:.2f} s")
