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
            for s in (97, 99, 101):
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
                seeds = [s for s in (97, 99, 101) if ("fixed", s) in tab and ("body", s) in tab]
                d = [abs(tab["fixed", s][k] - tab["body", s][k]) for s in seeds]
                spread = max(max(tab[a, s][k] for s in seeds) - min(tab[a, s][k] for s in seeds) for a in ("fixed", "body"))
                print(f"  {mesh} {k}: |fixed - body| = " + ", ".join(f"{v:.4f} (s{s})" for v, s in zip(d, seeds)) + f"; seed range {spread:.4f}"
                      f" -> {'within' if max(d) <= spread else 'OUTSIDE'}")
            except (KeyError, ValueError):
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
    for gtag in ("v1_bunny_g005", "v1_bunny_g010", "v1e_g005", "v1e_g010", "v1_af"):
        f = runs / f"{gtag}.json"
        if not f.exists():
            continue
        j = load(f)
        z = np.load(runs / f"{gtag}.npz", mmap_mode="r")
        fr, floor_y = z["frames"], float(z["floor_y"])
        dt, g = j["mpm"]["dt"], -j["mpm"]["f_ext"][1]
        sp = (j["scene"]["R"] ** 3 * 4.18879 / fr.shape[1]) ** (1 / 3)
        low = np.array([float(fr[k, :, 1].min()) for k in range(len(fr))]) - floor_y
        touch = int(np.argmax(low < sp)) if (low < sp).any() else len(fr)
        last = [r for r in j["history"] if "bodies" in r][-1]["bodies"][0]
        print(f"  {gtag} (G {j['scene']['spec']['G']}): {j['metrics'][0]}  first floor contact frame {touch} "
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
    win_s = 2 * j["config"]["T"] * dt
    hit_w = [bool(np.linalg.norm(r["contact_impulse"]) > 0) for r in h]
    contact = [r for r, c in zip(h, hit_w) if c]
    t_on = contact[0]["t"] - win_s if contact else None
    # the contact's episodes (runs of windows with a contact impulse); the pair parts at the end of the first window
    # after the first episode's start with no impulse and the bodies more than a spacing apart (the floor's own test);
    # a later touch (the bodies rolling into each other on the floor) is a re-contact, not the first contact's end
    episodes, k = [], 0
    while k < len(h):
        if hit_w[k]:
            e = k
            while e + 1 < len(h) and hit_w[e + 1]:
                e += 1
            episodes.append((round(h[k]["t"] - win_s, 3), round(h[e]["t"], 3)))
            k = e + 1
        else:
            k += 1
    k0 = h.index(contact[0]) if contact else len(h)
    part = next((r for r, c in zip(h[k0:], hit_w[k0:]) if not c and r["gap_sp"] > 1.0), None)
    t_part = part["t"] if part is not None else None
    t_off = max((r["t"] for r in contact if t_part is None or r["t"] < t_part), default=None)
    land = []
    for s, e in rows:
        low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - floor_y
        hit = np.nonzero(low < sp)[0]
        land.append(float(raws[hit[0]] * dt) if len(hit) else None)
    first_land = min([t for t in land if t is not None], default=None)
    parted_in_air = t_part is not None and first_land is not None and t_part <= first_land
    pre = next((r for r, c in zip(h, hit_w) if not c), h[0])
    post = part if part is not None else h[-1]
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
    last = [r for r in h if r["t"] > h[-1]["t"] - 1.0 - 1e-9]     # the last second: does each body hold its shape?
    hold = [[round(r["bodies"][b]["d_sil"], 5) for r in last] for b in range(len(rows))]
    out_hold = {"last_second_dsil": hold, "window_s": win_s}
    out = {"tag": tag, **out_hold, "contact_s": (t_off - t_on) if contact else 0.0, "t_contact": t_on, "t_part": t_part,
           "contact_span_s": (contact[-1]["t"] - t_on) if contact else 0.0, "episodes": episodes,
           "restitution": float(-vrel_post / vrel_pre) if vrel_pre else None, "landing_s": land,
           "parted_in_air": parted_in_air, "tumble_deg": tumble,
           "tumbles": all(t is not None and t > 30.0 for t in tumble), "end_rigid_speed": rest,
           "end_shape_error": err, "isolated": iso}
    print(json.dumps(out))
    return out


def l_split(runs, tag):
    """The angular momentum's change per term (window records' L_split, per unit mass times the particle's mass): over the
    contact's windows and over the whole run, each body about its own start centre of mass, and the pair's grid share
    about the pair's start centroid (the contact and the stress must keep it: gravity's torque taken out)."""
    f = runs / f"{tag}.json"
    if not f.exists():
        return
    j = load(f)
    h = [r for r in j["history"] if "bodies" in r and r.get("L_split")]
    if not h:
        print(f"== L split {tag}: no records")
        return
    m = 40000.0 / (int(j["provenance"]["scene_args"]["n"]))
    terms = ("L_grid", "L_gravity", "L_u", "L_space", "L_relax")
    for name, sel in (("contact windows", [r for r in h if np.linalg.norm(r["contact_impulse"]) > 0]), ("all", h)):
        if not sel:
            continue
        print(f"== L split {tag}, {name} ({len(sel)} windows), momentum units")
        for b in range(len(sel[0]["L_split"])):
            for ref in ("own", "pair"):
                tot = {k: m * np.sum([r["L_split"][b][ref][k] for r in sel], axis=0) for k in terms}
                print(f"  b{b} about {ref:4s}: " + "  ".join(f"{k[2:]}=({v[0]:+.0f},{v[1]:+.0f},{v[2]:+.0f})"
                                                           for k, v in tot.items()))
        pair = m * np.sum([[r["L_split"][b]["pair"]["L_grid"] for b in range(len(r["L_split"]))] for r in sel], axis=(0, 1))
        grav = m * np.sum([[r["L_split"][b]["pair"]["L_gravity"] for b in range(len(r["L_split"]))] for r in sel], axis=(0, 1))
        print(f"  the pair's grid share minus gravity's torque (contact + stress, should be ~0): {np.round(pair - grav, 1)}")


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
    pick = min(tie, key=lambda o: o["contact_span_s"])      # the registered duration: first to last impulse
    print(f"== select: {pick['tag']} (of {[o['tag'] for o in ok]} that part in the air, land and tumble; "
          f"end shape error {pick['end_shape_error']:.4f}, contact {pick['contact_s']:.2f} s, span {pick['contact_span_s']:.2f} s)")
    return pick["tag"]


if __name__ == "__main__":
    runs = Path(sys.argv[1])
    v0(runs)
    v1(runs)
    for tag in ("scene40k_L", "scene40k_P", "scene40k_LB", "scene40k_LS"):
        scene(runs, tag)
    select(runs)
    for tag in (sys.argv[2].split(",") if len(sys.argv) > 2 else []):
        l_split(runs, tag)
