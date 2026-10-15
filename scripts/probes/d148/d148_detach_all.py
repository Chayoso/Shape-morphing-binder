"""d148_detach_all.py LABEL FRAMES12 RUN_JSON DUMP_DIR [--w0 W] [--w1 W] [--jsonl OUT] -- every detachment in a dump, by update (CPU)

From D146's per-particle dump (see d148_detach.py), per window: the particles apart from the body at the window's end
(outside the largest set connected at 1.5 pitches) that were in it at the window's start. For each, the opening of its
distance to its 8 nearest body particles (body: in the largest set at the end, outside the particle's own set; the
nearest at the window's start), split exactly by update: grid, u on the particle itself, u on its body neighbours, the
spacing push, the relaxation (d148_detach.py's split); the update that opened it most; whether it was in the layer and in
u's gate; the spray cleanup's isolation gate at the window's start (objective.isolation_gate: the 8th-neighbour distance
over its median, ramp 1.2-1.8: > 0 = the cleanup counts it); and the term whose descent along its normal is largest.

And what a change of reach would touch, per window: the layer particles in u's gate the isolation gate counts, their share
of u's displacement and of each term's descent along the normal on the gated layer; and the particles off the layer the
isolation gate counts; at three readings of the gate (> 0, >= 0.5, = 1: the ramp's top, the cleanup's true singletons), with the
share of the layer's u above one pitch. Each detachment is also read against the display (visible_apart.classify at
each window's end): whether its particle is a blob, its own piece of the display's surface (and one the side view shows),
within the next second (six windows) of its detaching window; the summary is given for all, for those and for the side
ones."""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

args = sys.argv[1:]
opt = {}
for k in ("--w0", "--w1", "--jsonl"):
    if k in args:
        i = args.index(k)
        opt[k] = args[i + 1]
        del args[i:i + 2]
label, fr_path, jpath, ddir = args[:4]
run = json.load(open(jpath))
cfg = next(iter(run["arms"].values()))["config"]
T = int(cfg["T"])
STEPS = 2 * T
min_sp = float(cfg.get("min_spacing", 0.0))
jout = open(opt["--jsonl"], "w") if "--jsonl" in opt else None
w0, w1 = int(opt.get("--w0", 1)), int(opt.get("--w1", 10 ** 6))
TERMS = ("ot", "surf", "near", "spray", "rend")
z = np.load(fr_path)
tgt = np.asarray(z["tgt"], np.float64)
A = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))          # the target sample's pitch
fr0 = np.asarray(z["frames"][0], np.float64)
R_SP = min_sp * (ConvexHull(fr0).volume / len(fr0)) ** (1.0 / 3.0)          # d148_detach.py's spacing radius
del z, fr0


def labels(x):
    pairs = cKDTree(x).query_pairs(1.5 * A, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    return cl, int(np.bincount(cl).argmax())


def push_vec(snaps, P, nb16):
    tot = np.zeros((len(P), 3))
    for xs in snaps:
        d = xs[P][:, None, :] - xs[nb16]
        L = np.linalg.norm(d, axis=2)
        ov = (L < R_SP) & (L > 1e-9)
        tot += 10 * (1.0 / T) * 0.5 * np.where(ov[..., None], ((R_SP - L) / np.where(L > 0, L, 1))[..., None] * d,
                                               0.0).sum(1)
    return tot


events, blob_sets, side_sets = [], {}, {}
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_d148x/scripts/probes/paper")
import visible_apart as VA  # noqa: E402   (the display's classification: blob = its own piece of the display surface)
CAM = VA.side_camera(tgt)
risk = []
import glob, os  # noqa: E401,E402
files = sorted(glob.glob(f"{ddir}/w_*.npz"))
for f in files:
    w = int(os.path.basename(f)[2:6])
    if not w0 <= w <= w1:
        continue
    d = np.load(f)
    if "x0" not in d.files:
        continue
    sp = float(d["sp0"])
    x0 = d["x0"].astype(np.float64)
    D = {k: d[k].astype(np.float64) for k in ("d10", "d20", "d30", "d40")}
    xe = x0 + D["d40"]
    n = d["lnrm"].astype(np.float64)
    lm = d["lmask"].astype(np.float64) > 0.5
    ug = d["ug"].astype(np.float64)
    uu_all = d["u"].astype(np.float64) * ug
    tree0 = cKDTree(x0)
    d9 = tree0.query(x0, 9)[0][:, 8]
    gate = np.clip((d9 / np.median(d9) - 1.2) / 0.6, 0.0, 1.0)
    cl0, b0 = labels(x0)
    cle, be = labels(xe)
    new = np.nonzero((cle != be) & (cl0 == b0))[0]
    ap_e, kind_e, _, occ_e = VA.classify(xe, A, CAM)
    blob_end = set(ap_e[kind_e == 2].tolist())
    side_end = set(ap_e[(kind_e == 2) & ~occ_e].tolist())
    blob_sets[w], side_sets[w] = blob_end, side_end
    # the reach: layer particles in u's gate, the gated ones among them
    on = lm & (ug > 0.5)
    rk = dict(window=w, layer=int(lm.sum()), u_reach=int(on.sum()), new=int(len(new)))
    for lvl, sel in (("gt0", gate > 0), ("half", gate >= 0.5), ("full", gate >= 1.0)):
        g_on = on & sel
        rk["u_reach_" + lvl] = int(g_on.sum())
        rk["u_share_" + lvl] = float(np.abs(uu_all[g_on]).sum() / max(np.abs(uu_all[on]).sum(), 1e-30))
        rk["u_big_" + lvl] = float(((np.abs(uu_all) > A) & g_on).sum() / max(((np.abs(uu_all) > A) & on).sum(), 1))
        rk["off_layer_" + lvl] = int((~lm & sel).sum())
        for k in TERMS:
            if "g_" + k in d.files:
                gn = -float(d["s_" + k]) * (d["g_" + k].astype(np.float64) * n).sum(1)
                rk[f"share_{k}_{lvl}"] = float(np.sqrt((gn[g_on] ** 2).sum() / max((gn[on] ** 2).sum(), 1e-60)))
    risk.append(rk)
    if len(new):
        _, cand = tree0.query(x0[new], 40)
        nb8 = []
        for i, p in enumerate(new):
            ok = [q for q in cand[i][1:] if cle[q] == be][:8]
            nb8.append(ok + [ok[-1]] * (8 - len(ok)) if ok else [cand[i][1]] * 8)
        nb8 = np.asarray(nb8)
        grp = np.unique(np.concatenate([new, nb8.reshape(-1)]))
        _, nb16 = tree0.query(x0[grp], 17)
        nb16 = nb16[:, 1:]
        snaps = [x0, x0 + D["d10"], x0 + D["d20"], x0 + D["d30"]]
        pd, pr = push_vec(snaps[:2], grp, nb16), push_vec(snaps[2:], grp, nb16)
        ng = n[grp]
        uu = uu_all[grp]
        rl_d = d["n_jmp_drv"].astype(np.float64)[grp] * sp - (pd * ng).sum(1)
        rl_r = d["n_jmp_rel"].astype(np.float64)[grp] * sp - (pr * ng).sum(1)
        parts = {"grid": D["d40"][grp] - uu[:, None] * ng - pd - pr - (rl_d + rl_r)[:, None] * ng,
                 "u": uu[:, None] * ng, "push": pd + pr, "relax": (rl_d + rl_r)[:, None] * ng}
        idx = {q: i for i, q in enumerate(grp)}
        tn_all = {k: -float(d["s_" + k]) * (d["g_" + k][new].astype(np.float64) * n[new]).sum(1)
                  for k in TERMS if "g_" + k in d.files}
        for i, p in enumerate(new):
            ip, iq = idx[p], np.asarray([idx[q] for q in nb8[i]])
            v0, v1 = x0[p] - x0[nb8[i]], xe[p] - xe[nb8[i]]
            r0, r1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
            wq = (v0 + v1) / (r0 + r1)[:, None]
            op = {}
            for k, v in parts.items():
                if k == "u":
                    op["u_own"] = float(np.mean((v[ip] * wq).sum(1))) / A
                    op["u_nbr"] = float(np.mean((-v[iq] * wq).sum(1))) / A
                else:
                    op[k] = float(np.mean(((v[ip] - v[iq]) * wq).sum(1))) / A
            top = max(op, key=op.get)
            tn = {k: float(v[i]) for k, v in tn_all.items()}
            ttop = max(tn, key=lambda k: abs(tn[k])) if tn else None
            ev = dict(window=w, p=int(p), open=op, top=top, exact=float((r1.mean() - r0.mean()) / A),
                      layer=int(lm[p]), ug=float(ug[p]), u=float(uu_all[p]) / A, gate=float(gate[p]),
                      own_set=int(np.sum(cle == cle[p])), term_top=ttop,
                      term_top_out=(None if ttop is None else int(tn[ttop] > 0)))
            ev["blob"] = int(p in blob_end)
            ev["side_blob"] = int(p in side_end)
            events.append(ev)
    print(f"[{label}] w{w} ({(int(d['frame_end']) - STEPS) / 240:.2f} s): newly apart {len(new)} (ending as a blob "
          f"{sum(1 for p in new if p in blob_end)}, side {sum(1 for p in new if p in side_end)}); u reach {rk['u_reach']}; "
          + "; ".join(f"gate {lvl}: {rk['u_reach_' + lvl]} of it ({rk['u_share_' + lvl]:.3f} of |u|, {rk['u_big_' + lvl]:.2f} "
                      f"of |u| > 1 a; render {rk.get('share_rend_' + lvl, float('nan')):.2f}, spray "
                      f"{rk.get('share_spray_' + lvl, float('nan')):.2f} of the normal descent), off the layer "
                      f"{rk['off_layer_' + lvl]}" for lvl in ("gt0", "half", "full")), flush=True)


def summarize(name, evs):
    E = len(evs)
    if not E:
        print(f"[{label}] {name}: none")
        return
    by = {}
    for e in evs:
        by[e["top"]] = by.get(e["top"], 0) + 1
    f = lambda c: sum(1 for e in evs if c(e)) / E  # noqa: E731
    print(f"[{label}] {name}: {E}; the update that opened the link most: "
          + ", ".join(f"{k} {v / E:.2f}" for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
          + f"; in the layer {f(lambda e: e['layer']):.2f}, in u's gate {f(lambda e: e['layer'] and e['ug'] > 0.5):.2f}; "
          f"isolation gate at the window's start > 0 {f(lambda e: e['gate'] > 0):.2f}, >= 0.5 "
          f"{f(lambda e: e['gate'] >= 0.5):.2f}, = 1 {f(lambda e: e['gate'] >= 1):.2f}; u on itself the largest and the gate "
          f"= 1 {f(lambda e: e['top'] == 'u_own' and e['gate'] >= 1):.2f}, >= 0.5 "
          f"{f(lambda e: e['top'] == 'u_own' and e['gate'] >= 0.5):.2f}; off the layer with the gate >= 0.5 "
          f"{f(lambda e: not e['layer'] and e['gate'] >= 0.5):.2f}; the largest term along its normal: spray "
          f"{f(lambda e: e['term_top'] == 'spray'):.2f} (outward {f(lambda e: e['term_top'] == 'spray' and e['term_top_out']):.2f}),"
          f" render {f(lambda e: e['term_top'] == 'rend'):.2f} (outward "
          f"{f(lambda e: e['term_top'] == 'rend' and e['term_top_out']):.2f}), transport {f(lambda e: e['term_top'] == 'ot'):.2f},"
          f" near {f(lambda e: e['term_top'] == 'near'):.2f}; u on it median {np.median([e['u'] for e in evs]):+.2f} a",
          flush=True)


for e in events:                       # a blob within the next second (six windows) of its detaching window
    e["blob_1s"] = int(any(e["p"] in blob_sets.get(v, ()) for v in range(e["window"], e["window"] + 7)))
    e["side_1s"] = int(any(e["p"] in side_sets.get(v, ()) for v in range(e["window"], e["window"] + 7)))
print()
summarize("all detachments", events)
summarize("those a blob of the display (their own piece of its surface) within the next second", [e for e in events if e["blob_1s"]])
summarize("those a side-view blob within the next second", [e for e in events if e["side_1s"]])
if risk:
    m = {k: float(np.median([r[k] for r in risk])) for k in risk[0] if k != "window"}
    print(f"[{label}] reach, median over {len(risk)} windows: " + ", ".join(f"{k} {v:.3g}" for k, v in m.items()))
if jout:
    for e in events:
        jout.write(json.dumps(e) + "\n")
    jout.write(json.dumps(dict(risk=risk)) + "\n")
