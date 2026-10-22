"""tobody.py RUNS_DIR TAG [T_SECONDS ...] -- B9b: a one-body scene archive (scripts/scene_run.py's TAG.npz, every kept
frame in the world) for the single-body readers, and its world track. CPU only.

Writes, next to the run:
  TAG_body12.npz    the kept frames in the body's Buckens frame y = (x - c) R (R the polar factor of sum (x - c)(X - Xbar)^T,
                    X the source sample at launch, as scene/frame.body_frame), with the body's target (about its centroid),
                    raws and deliver_n: the frames12 format the single-body readers take (gallery_ind.py, endgen145t.py,
                    render_terms_probe.py, surface_layer_probe.py); the body-frame end is what the scene optimised.
  TAG_frames12.npz  the same kept frames in the world (momentum_probe.py's input) with the same target (its pitch).
  TAG_at<T>_body.npz  for each T_SECONDS: the body-frame kept frame nearest T (a reading at a single-body run's end time).
  TAG_track.json    per kept frame: t, the centroid (wu), its height above the floor's centre plane (where the lowest
                    centre stood at launch) against the target standing there, the lowest centre above that plane
                    (pitches), the frame's turn, tilt (the body's up axis from the vertical) and turn about the vertical
                    (deg), and over the pair to the next kept frame the centroid's speed, the angular speed, the RMS
                    deformation speed (v less the least-squares rigid velocity) and the mean particle move (pitches).
The pitch: the sample's (vp^(1/3)); the at-rest layout put the floor's plane half of it below the lowest centre.
Option 1 (frame "standing", 2026-10-20): the archives are in the objective's frame, fixed in the world (the run's
fixed_frames: the single-body run's world, its target lifted onto the floor), to be read against the independent sample
lifted the same way (tmp/d145/ind_shift.py), as B9's; and every 4th kept frame (--shape_every K) and the last carry the
shape's own tilt (shape_rotation), the upright item's measure, beside the Buckens (material) frame's."""
import json
import sys
from pathlib import Path

import numpy as np


def polar(A):
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def shape_rotation(xc, tc, tree_t, R, iters=40, sub=20000):
    """The rotation (and shift) that best aligns a body's shape to its target's, y = xc R + t ~ the target: symmetric
    nearest-neighbour pairs (body to target and target to body, up to `sub` points each), Kabsch on them, from R (the
    Buckens rotation), until R moves by less than 1e-5. A shape measure, unlike the Buckens frame (the material's: a
    morph that turns material inside an upright shape turns it, B9's 5-8 deg). xc, tc: about their centroids."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    xs = xc[rng.choice(len(xc), min(sub, len(xc)), replace=False)]
    ts = tc[rng.choice(len(tc), min(sub, len(tc)), replace=False)]
    t = np.zeros(3)
    for _ in range(iters):
        y = xs @ R + t
        i = tree_t.query(y)[1]
        j = cKDTree(y).query(ts)[1]
        P = np.concatenate([xs, xs[j]])
        Q = np.concatenate([tc[i], ts])
        pm, qm = P.mean(0), Q.mean(0)
        Rn = polar((P - pm).T @ (Q - qm))
        t = qm - pm @ Rn
        done = np.abs(Rn - R).max() < 1e-5
        R = Rn
        if done:
            break
    return R


def shape_fit(xc, tc, tree_t, starts, iters=400, sub=20000):
    """The shape's best alignment (corrected 2026-10-21 after option 1's 40k gate): shape_rotation from every start
    (the identity = the target's own orientation, the Buckens rotation, the previous frame's fit), each run to its end
    (iters 400; 40 from the Buckens start stopped partway when the material had turned: bunny L99 3.50 deg at a mean
    nearest distance 0.6717 pitches against 0.37-0.41 deg at 0.6596 once converged, B9's bunny 3.10 against 0.22), and
    the one with the lowest symmetric residual (the mean nearest distance both ways). Returns (R, residual)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    xs = xc[rng.choice(len(xc), min(sub, len(xc)), replace=False)]
    tq = tc[::max(1, len(tc) // sub)]
    best = None
    for R0 in starts:
        R = shape_rotation(xc, tc, tree_t, R0, iters=iters, sub=sub)
        y = xs @ R
        y = y + (tc[tree_t.query(y)[1]].mean(0) - y.mean(0))
        r = 0.5 * (tree_t.query(y)[0].mean() + cKDTree(y).query(tq)[0].mean())
        if best is None or r < best[1]:
            best = (R, float(r))
    return best


def rigid(x, v):
    c = x.mean(0)
    r = x - c
    vc = v.mean(0)
    L = np.cross(r, v - vc).sum(0)
    I = (r * r).sum() * np.eye(3) - r.T @ r
    w = np.linalg.solve(I, L)
    return vc, w, v - vc - np.cross(w, r)


def main():
    runs, tag = Path(sys.argv[1]), sys.argv[2]
    track_only = "--track_only" in sys.argv      # the track alone (its archives already written and read)
    args = [s for i, s in enumerate(sys.argv) if s not in ("--shape_every", "--track_only") and sys.argv[i - 1] != "--shape_every"]
    times = [float(s) for s in args[3:]]
    j = json.loads((runs / f"{tag}.json").read_text())
    z = np.load(runs / f"{tag}.npz")
    fr, raws, rows = z["frames"], z["raws"].astype(int), z["rows"]
    assert len(rows) == 1, "one body"
    dt = float(j["mpm"]["dt"])
    floor_y = float(z["floor_y"])
    X = np.asarray(z["src0"], np.float64)
    X = X - X.mean(0)
    tgt = np.asarray(z["tgt0"], np.float32)
    floor_c = float(fr[0][:, 1].min())                     # the lowest centre at launch: the floor's centre plane
    vps = (j.get("scene") or {}).get("vp")                  # the sample's pitch, vp^(1/3), where recorded; else from the
    pitch = float(vps[0]) ** (1.0 / 3.0) if vps else 2.0 * (floor_c - floor_y)   # at-rest floor (half a pitch under)
    h_tgt = float(tgt[:, 1].mean() - tgt[:, 1].min())       # the target's centroid above its lowest point
    # frame "standing" (option 1): the archive in the objective's frame, fixed in the world (the single-body run's world,
    # the target lifted onto the floor: read against the independent sample lifted the same way, as B9's); otherwise the
    # body's Buckens frame
    ff = (j.get("scene") or {}).get("fixed_frames")
    obj_frame = None if not ff else (np.asarray(ff[0]["c"], np.float64), np.asarray(ff[0]["R"], np.float64))
    from scipy.spatial import cKDTree
    tc = np.asarray(tgt, np.float64) - np.asarray(tgt, np.float64).mean(0)
    tree_t = cKDTree(tc)
    shape_every = int(sys.argv[sys.argv.index("--shape_every") + 1]) if "--shape_every" in sys.argv else 4
    ys, track, Rs_prev = [], [], None
    for k in range(len(fr)):
        x = np.asarray(fr[k], np.float64)
        c = x.mean(0)
        R = polar((x - c).T @ X)
        ys.append((((x - obj_frame[0]) @ obj_frame[1]) if obj_frame is not None else ((x - c) @ R)).astype(np.float32))
        up = R[:, 1]                                        # the body's vertical axis in the world
        yaw = float(np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])))
        rec = {"k": k, "raw": int(raws[k]), "t": float(raws[k] * dt), "c": c.tolist(),
               "h_wu": float(c[1] - floor_c), "dh_wu": float(c[1] - floor_c - h_tgt),
               "dh_sp": float((c[1] - floor_c - h_tgt) / pitch),
               "lowest_sp": float((x[:, 1].min() - floor_c) / pitch), "turn_deg": angle(R),
               "tilt_deg": float(np.degrees(np.arccos(np.clip(up[1], -1.0, 1.0)))), "yaw_deg": yaw,
               "drift_xz_sp": float(np.hypot(c[0] - fr[0][:, 0].mean(), c[2] - fr[0][:, 2].mean()) / pitch)}
        if k % shape_every == 0 or k == len(fr) - 1:      # the shape's own tilt (shape_fit: converged, the best start)
            Rs, res = shape_fit(x - c, tc, tree_t, [np.eye(3), R] + ([Rs_prev] if Rs_prev is not None else []))
            Rs_prev = Rs
            rec.update(shape_tilt_deg=float(np.degrees(np.arccos(np.clip(Rs[1, 1], -1.0, 1.0)))),
                       shape_turn_deg=angle(Rs), shape_resid_wu=res)
        if k + 1 < len(fr):
            x1 = np.asarray(fr[k + 1], np.float64)
            tau = float(raws[k + 1] - raws[k]) * dt
            v = (x1 - x) / tau
            vc, w, vd = rigid(x, v)
            rec.update(v_com=float(np.linalg.norm(vc)), omega=float(np.linalg.norm(w)),
                       v_def_rms=float(np.sqrt((vd * vd).sum(1).mean())),
                       move_sp=float(np.linalg.norm(x1 - x, axis=1).mean() / pitch), tau=tau)
        track.append(rec)
    ys = np.stack(ys)
    dn = np.int64(raws[-1] + 1)
    if not track_only:
        np.savez(runs / f"{tag}_body12.npz", frames=ys, raws=raws, tgt=tgt, deliver_n=dn, src=X.astype(np.float32))
        np.savez(runs / f"{tag}_frames12.npz", frames=fr, raws=raws, tgt=tgt, deliver_n=dn)
    at = {}
    for T in times:
        k = int(np.argmin(np.abs(raws * dt - T)))
        name = f"{tag}_at{T:g}_body.npz"
        if not track_only:
            np.savez(runs / name, frames=ys[k:k + 1], raws=raws[k:k + 1], tgt=tgt, deliver_n=np.int64(raws[k] + 1))
        at[f"{T:g}"] = {"file": name, "k": k, "t": float(raws[k] * dt)}
    out = {"tag": tag, "pitch": pitch, "floor_y": floor_y, "floor_c": floor_c, "h_target_wu": h_tgt,
           "R_body": float(np.linalg.norm(X, axis=1).max()), "at": at, "track": track}
    (runs / f"{tag}_track.json").write_text(json.dumps(out))
    e = track[-1]
    print(f"{tag}: {len(fr)} kept frames to {e['t']:.2f} s; pitch {pitch:.4f} wu; end: height vs the standing target "
          f"{e['dh_wu']:+.4f} wu ({e['dh_sp']:+.2f} sp), lowest {e['lowest_sp']:+.2f} sp, turn {e['turn_deg']:.2f} "
          f"(tilt {e['tilt_deg']:.2f}, about the vertical {e['yaw_deg']:+.2f}) deg, drift {e['drift_xz_sp']:.2f} sp; "
          f"the shape's tilt {e.get('shape_tilt_deg', float('nan')):.2f} deg; archive in the "
          f"{'objective (standing) frame' if obj_frame is not None else 'Buckens frame'}")


if __name__ == "__main__":
    main()
