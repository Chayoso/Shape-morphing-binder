"""D147 readings of the 40k runs (output/gpu/d147/runs): V0 (fixed vs Buckens frame), V1 (one body thrown onto the floor)
and the scene (E1-E4 from the records and the archive). Read-only; run on hyde06 with the repository on the path.

  $PY scripts/probes/d147/read40k.py RUNS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(p):
    return json.loads(Path(p).read_text())


def v0(runs):
    print("== V0: one body at rest, the frame fixed at launch against the Buckens frame")
    keys = ("sil_iou", "chamfer", "thin_uncovered")
    for mesh in ("bunny", "dragon"):
        tab = {}
        for arm in ("fixed", "body"):
            for s in (97, 99):
                f = runs / f"v0_{mesh}_{arm}_s{s}.json"
                if not f.exists():
                    continue
                j = load(f)
                m, h = j["metrics"][0], j["history"]
                tab[arm, s] = m
                print(f"  {mesh} {arm:5s} s{s}: " + "  ".join(f"{k}={m.get(k, float('nan')):.4f}" for k in keys)
                      + f"  R_end={h[-1]['bodies'][0]['R_angle']:.2f}deg  windows={len(h)}  {j['seconds'] / 60:.1f} min")
        for k in keys:
            try:
                d = [abs(tab["fixed", s][k] - tab["body", s][k]) for s in (97, 99)]
                spread = max(abs(tab[a, 97][k] - tab[a, 99][k]) for a in ("fixed", "body"))
                print(f"  {mesh} {k}: |fixed - body| = {d[0]:.4f} (s97), {d[1]:.4f} (s99); seed spread {spread:.4f}"
                      f" -> {'within' if max(d) <= spread else 'OUTSIDE'}")
            except KeyError:
                pass


def ballistic(fr, rows, dt, g, upto):
    """Largest distance of each body's centre of mass from its parabola over frames [0, upto)."""
    out = []
    for s, e in rows:
        c = fr[:upto, s:e].astype(np.float64).mean(1)
        t = np.arange(upto) * dt
        v0 = (c[1] - c[0]) / dt + np.array([0.0, 0.5 * g * dt, 0.0])        # the first step's mean velocity
        pred = c[0] + np.outer(t, v0) + np.outer(0.5 * t * t, [0.0, -g, 0.0])
        out.append(float(np.linalg.norm(c - pred, axis=1).max()))
    return out


def v1(runs):
    print("== V1: one bunny thrown onto the floor")
    for gtag in ("005", "010"):
        f = runs / f"v1_bunny_g{gtag}.json"
        if not f.exists():
            continue
        j = load(f)
        z = np.load(runs / f"v1_bunny_g{gtag}.npz", mmap_mode="r")
        fr, floor_y = z["frames"], float(z["floor_y"])
        dt, g = j["mpm"]["dt"], -j["mpm"]["f_ext"][1]
        sp = (j["scene"]["R"] ** 3 * 4.18879 / fr.shape[1]) ** (1 / 3)
        low = np.array([float(fr[k, :, 1].min()) for k in range(len(fr))]) - floor_y
        touch = int(np.argmax(low < sp)) if (low < sp).any() else len(fr)
        last = [r for r in j["history"] if "bodies" in r][-1]["bodies"][0]
        print(f"  G {int(gtag) / 100:.2f}: {j['metrics'][0]}  first floor contact frame {touch} "
              f"({touch * dt:.2f} s)  end: lowest {low[-1] / sp:.2f} spacings above the floor, |v_com| "
              f"{np.linalg.norm(last['v_com']):.4f}, |w| {np.linalg.norm(last['omega']):.4f}; CoM off its parabola "
              f"before the floor {ballistic(fr, [(0, fr.shape[1])], dt, g, max(2, touch - 1))[0]:.2e} wu")


def scene(runs, tag):
    f = runs / f"{tag}.json"
    if not f.exists():
        return
    j = load(f)
    h = [r for r in j["history"] if "bodies" in r]          # the last record of a run that left the domain has none
    z = np.load(runs / f"{tag}.npz", mmap_mode="r")
    fr, rows = z["frames"], z["rows"]
    dt, g = j["mpm"]["dt"], -j["mpm"]["f_ext"][1]
    M = sum(40000.0 for _ in rows)
    print(f"== scene {tag}: {len(h)} windows, {j['seconds'] / 60:.1f} min, metrics {j['metrics']}")
    first = next((r for r in h if np.linalg.norm(r["contact_impulse"]) > 0), None)
    if first is not None:
        print(f"  first contact in window {first['animation'] + 1} (t {first['t']:.2f} s): gap at its end "
              f"{first['gap_sp']:.2f} spacings")
    P0 = np.array([sum(b["P"][k] for b in h[0]["bodies"]) for k in range(3)])
    for r in h:
        P = np.array([sum(b["P"][k] for b in r["bodies"]) for k in range(3)])
        L = np.array([sum(b["L"][k] for b in r["bodies"]) for k in range(3)])
        print(f"  w{r['animation'] + 1:3d} t={r['t']:5.2f} gap={r['gap_sp']:6.2f}sp near={r['near']} "
              f"J=({r['contact_impulse'][0]:+.3g},{r['contact_impulse'][1]:+.3g},{r['contact_impulse'][2]:+.3g}) "
              f"dP=({P[0] - P0[0]:+.3g},{P[1] - P0[1] + M * g * (r['t'] - h[0]['t']):+.3g},{P[2] - P0[2]:+.3g}) "
              + " ".join(f"b{b}: c=({x['com'][0]:+.2f},{x['com'][1]:+.2f}) v=({x['v_com'][0]:+.2f},{x['v_com'][1]:+.2f})"
                         f" w=({x['omega'][0]:+.2f},{x['omega'][1]:+.2f},{x['omega'][2]:+.2f}) R={x['R_angle']:.0f}"
                         f" sil={x['d_sil']:.4f}" for b, x in enumerate(r["bodies"])) + f" Lz={L[2]:+.4g}")
    touch = len(fr)
    if first is not None:
        touch = int(round(first["t"] / dt)) - 2 * int(j["config"]["T"])
    print(f"  centres off their parabolas before the first contact window: {ballistic(fr, rows, dt, g, max(2, touch))}")


def timeline(runs, tag):
    """The selection rule's readings for one scene variant (pre-registered in docs/experiments.md, D147): the contact's
    duration, the restitution, whether the pair parts before either body touches the floor, the landing, the tumble
    (each body's frame turning by more than 30 degrees after its landing), the end shape error (1 - silhouette IoU,
    mean over the bodies, each in its frame), the isolated particles (no particle of its own body within two sample
    spacings) per window, and the pair's momentum budget before the floor."""
    f = runs / f"{tag}.json"
    if not f.exists():
        return None
    j = load(f)
    h = [r for r in j["history"] if "bodies" in r]          # the last record of a run that left the domain has none
    z = np.load(runs / f"{tag}.npz", mmap_mode="r")
    fr, rows, raws = z["frames"], z["rows"], (z["raws"] if "raws" in z.files else np.arange(len(z["frames"])))
    dt, g, floor_y = j["mpm"]["dt"], -j["mpm"]["f_ext"][1], float(z["floor_y"])
    R = j["scene"]["R"]
    sp = (4.18879 * R ** 3 / (rows[0][1] - rows[0][0])) ** (1 / 3)
    contact = [r for r in h if np.linalg.norm(r["contact_impulse"]) > 0]
    t_on = contact[0]["t"] - 2 * j["config"]["T"] * dt if contact else None
    t_off = contact[-1]["t"] if contact else None
    land = []
    for s, e in rows:
        low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - floor_y
        hit = np.nonzero(low < sp)[0]
        land.append(float(raws[hit[0]] * dt) if len(hit) else None)
    first_land = min([t for t in land if t is not None], default=None)
    parted_in_air = bool(contact) and first_land is not None and t_off <= first_land
    pre = next((r for r in h if not np.linalg.norm(r["contact_impulse"]) > 0), h[0])
    post = next((r for r in h if t_off is not None and r["t"] > t_off + 1e-9), h[-1])
    c0, c1 = (np.array(b["com"]) for b in post["bodies"])
    n = (c1 - c0) / max(np.linalg.norm(c1 - c0), 1e-12)
    vrel_pre = np.dot(np.array(pre["bodies"][1]["v_com"]) - np.array(pre["bodies"][0]["v_com"]), n)
    vrel_post = np.dot(np.array(post["bodies"][1]["v_com"]) - np.array(post["bodies"][0]["v_com"]), n)
    tumble = []
    for b, tl in enumerate(land):
        if tl is None:
            tumble.append(None)
            continue
        Rl = np.array(next(r for r in h if r["t"] >= tl)["bodies"][b]["R"]).reshape(3, 3)
        Re = np.array(h[-1]["bodies"][b]["R"]).reshape(3, 3)
        tumble.append(float(np.degrees(np.arccos(np.clip((np.trace(Rl.T @ Re) - 1) / 2, -1, 1)))))
    err = float(np.mean([1.0 - m["sil_iou"] for m in j["metrics"]]))
    rest = [np.linalg.norm(b["v_com"]) for b in h[-1]["bodies"]]
    iso = []
    for k in range(0, len(fr), max(1, len(fr) // 12)):
        cnt = []
        for s, e in rows:
            x = np.asarray(fr[k, s:e], np.float64)
            from scipy.spatial import cKDTree
            d = cKDTree(x).query(x, 2)[0][:, 1]
            cnt.append(int((d > 2 * sp).sum()))
        iso.append((float(raws[k] * dt), cnt))
    win_s = 2 * j["config"]["T"] * dt
    last = [r for r in h if r["t"] > h[-1]["t"] - 1.0 - 1e-9]     # the last second: does each body hold its shape?
    hold = [[round(r["bodies"][b]["d_sil"], 5) for r in last] for b in range(len(rows))]
    out_hold = {"last_second_dsil": hold, "window_s": win_s}
    out = {"tag": tag, **out_hold, "contact_s": (t_off - t_on) if contact else 0.0, "t_contact": t_on, "t_part": t_off,
           "restitution": float(-vrel_post / vrel_pre) if vrel_pre else None, "landing_s": land,
           "parted_in_air": parted_in_air, "tumble_deg": tumble,
           "tumbles": all(t is not None and t > 30.0 for t in tumble), "end_rigid_speed": rest,
           "end_shape_error": err, "isolated": iso}
    print(json.dumps(out))
    return out


def select(runs):
    """The 300k scene's variant by the rule written before reading (D147): among those that part in the air, land and
    tumble within the scene's length, the lower end shape error; within 0.002 (the 40k run-to-run silhouette spread),
    the shorter contact."""
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["scene40k_L", "scene40k_LB", "scene40k_LS"]
    cand = [o for o in (timeline(runs, t) for t in tags) if o]
    ok = [o for o in cand if o["parted_in_air"] and all(t is not None for t in o["landing_s"]) and o["tumbles"]]
    if not ok:
        print("== select: no variant parts in the air, lands and tumbles: report, no choice")
        return None
    best = min(o["end_shape_error"] for o in ok)
    tie = [o for o in ok if o["end_shape_error"] <= best + 0.002]
    pick = min(tie, key=lambda o: o["contact_s"])
    print(f"== select: {pick['tag']} (of {[o['tag'] for o in ok]} that part in the air, land and tumble; "
          f"end shape error {pick['end_shape_error']:.4f}, contact {pick['contact_s']:.2f} s)")
    return pick["tag"]


if __name__ == "__main__":
    runs = Path(sys.argv[1])
    v0(runs)
    v1(runs)
    for tag in ("scene40k_L", "scene40k_P", "scene40k_LB", "scene40k_LS"):
        scene(runs, tag)
    select(runs)
