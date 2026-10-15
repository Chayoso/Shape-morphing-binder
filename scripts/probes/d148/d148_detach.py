"""d148_detach.py FRAMES12 RUN_JSON DUMP_DIR P1,P2,...@T_VISIBLE[@W1,W2] [...] [--jsonl OUT] -- who detaches a visible
floating piece? (CPU)

Per piece (its particles and the time it is first seen, from d148_why_floating.py), from the run's kept frames (every 12
steps): the frames where each particle is apart from the body (not connected at 1.5 pitches to a set of more than 3000
particles, i.e. outside the body's largest set), and the window where the apart stretch that reaches the visible frame
began (the detaching window; windows from the dump's frame_end, 2T steps each).

From D146's per-particle dump (w_AAAA.npz, windows D146_A0..D148_A1: x0 the window's start, dTT the displacement at step
TT, u and its gate ug, the layer mask and its frozen normal n, n_adv_* the grid's advection and n_jmp_* the position
updates below the grid less u along n, in spacings, in the driven (drv) and released (rel) halves): the particle's
displacement over the window, split by update as vectors (body neighbours: the nearest at the window's start outside
the particle's own set at 1.5 pitches):
  u        = u ug n                                  (k_layer_project; the driven half)
  push     = the minimum spacing's push, recomputed (k_update: the 16 frozen nearest at the window's start, r = min_spacing
             x the rest volume's pitch, 0.5 x bond_frac = 1/T a step), integrated from the snapshots at steps 0, 10, 20, 30
  relax    = (n_jmp - push . n) n                    (k_layer_relax / k_layer_project move along n only; the bonds act on
             the fragment mask only, empty in these runs: `fragments 0` every window)
  grid     = the rest (dt v of the G2P velocity, APIC included); its n part is checked against the dump's n_adv.
Per part: the opening of the particle's distance to its 8 nearest body particles (the mean over q of (dx_p - dx_q) .
(r0 + r1) / (|r0| + |r1|) with r = x_p - x_q at the start and end, exact; in pitches: what crosses the 1.5-pitch link), and its displacement against the
mean of its 64 nearest body particles along that displacement (the parts sum to its length, in spacings). Also the
layer test's asymmetry |x - centroid of its 32 nearest| (layer.layer_by_asymmetry: in the layer from 0.5 spacings).
Windows: the detaching window and the one before, up to the visible one (at most 8), plus W1,W2 if given."""
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

args = [s for s in sys.argv[1:]]
jout = None
if "--jsonl" in args:
    i = args.index("--jsonl")
    jout = open(args[i + 1], "w")
    del args[i:i + 2]
fr_path, jpath, ddir = args[:3]
z = np.load(fr_path, mmap_mode="r")
FR = z["frames"]
raws = np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
A = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
run = json.load(open(jpath))
cfg = next(iter(run["arms"].values()))["config"]
T = int(cfg["T"])
STEPS = 2 * T
min_sp = float(cfg.get("min_spacing", 0.0))
P_VOL = (ConvexHull(np.asarray(FR[0], np.float64)).volume / FR.shape[1]) ** (1.0 / 3.0)   # D149's P_vol
R_SP = min_sp * P_VOL
CAP = 3000
label = fr_path.split("/")[-1].replace("_frames12.npz", "").replace("dragon300k_", "").replace("bunny300k_", "")

# windows: dump index w (1-based) -> steps [frame_end - 2T, frame_end)
wend = {}
import glob, os  # noqa: E401,E402
for f in sorted(glob.glob(f"{ddir}/w_*.npz")):
    w = int(os.path.basename(f)[2:6])
    with np.load(f) as d:
        wend[w] = int(d["frame_end"])
W_ALL = sorted(wend)


def window_of(step):
    for w in W_ALL:
        if wend[w] - STEPS <= step < wend[w]:
            return w
    return None


def component(tree, X, p, r=1.5 * A, cap=CAP):
    """p's set connected at 1.5 pitches, or None when it reaches more than cap particles (the body)"""
    seen = {int(p)}
    front = [int(p)]
    while front:
        new = set()
        for lst in tree.query_ball_point(X[front], r):
            new.update(lst)
        new -= seen
        if len(seen) + len(new) > cap:
            return None
        seen |= new
        front = list(new)
    return seen


def attached(tree, X, p):
    return component(tree, X, p) is None


def push_vec(snaps, P, nb16):
    """the spacing push's displacement of particles P from the given snapshots (each standing for 10 steps)"""
    tot = np.zeros((len(P), 3))
    for xs in snaps:
        d = xs[P][:, None, :] - xs[nb16]
        L = np.linalg.norm(d, axis=2)
        ov = (L < R_SP) & (L > 1e-9)
        tot += 10 * (1.0 / T) * 0.5 * np.where(ov[..., None], ((R_SP - L) / np.where(L > 0, L, 1))[..., None] * d,
                                               0.0).sum(1)
    return tot


specs = []
for s in args[3:]:
    parts = s.split("@")
    P = [int(v) for v in parts[0].split(",")]
    tv = float(parts[1])
    extra = [int(v) for v in parts[2].split(",")] if len(parts) > 2 else []
    specs.append((P, tv, extra))

# apart frames per particle, up to each piece's visible frame
fvis = [int(np.argmin(np.abs(raws / 240.0 - tv))) for _, tv, _ in specs]
fmax = max(fvis)
allP = sorted({p for P, _, _ in specs for p in P})
ap = {p: np.zeros(fmax + 1, bool) for p in allP}
for f in range(fmax + 1):
    X = np.asarray(FR[f], np.float64)
    tree = cKDTree(X)
    for p in allP:
        ap[p][f] = not attached(tree, X, p)
print(f"[{label}] pitch {A:.5f} wu, spacing radius {R_SP:.5f} wu ({min_sp} x P_vol {P_VOL:.5f}), frames 0-{fmax}, "
      f"dump windows {W_ALL[0]}-{W_ALL[-1]}", flush=True)

for (P, tv, extra), fv in zip(specs, fvis):
    print(f"\n== {label} piece {P} visible from {tv:.2f} s (frame {fv})", flush=True)
    wdet = {}
    for p in P:
        a = ap[p][:fv + 1]
        if not a[fv]:
            print(f"  p{p}: not apart at the visible frame (connected at 1.5 pitches)")
            f1 = fv
        else:
            f1 = fv
            while f1 > 0 and a[f1 - 1]:
                f1 -= 1
        first = int(np.argmax(a)) if a.any() else -1
        nfl = int(np.sum(np.diff(a.astype(int)) == 1))
        wdet[p] = window_of(int(raws[f1]))
        print(f"  p{p}: apart from {raws[f1] / 240:.2f} s (frame {f1}, window {wdet[p]}); first ever {raws[first] / 240:.2f} s; "
              f"{nfl} separate apart stretches before it is seen", flush=True)
    wd = min(w for w in wdet.values() if w is not None) if any(w is not None for w in wdet.values()) else None
    wv = window_of(int(raws[fv]))
    if wd is None:
        continue
    ws = list(range(wd - 1, min(wv if wv else wd, wd + 6) + 1)) + extra
    ws = sorted(set(w for w in ws if w in wend))
    for w in ws:
        d = np.load(f"{ddir}/w_{w:04d}.npz")
        if "x0" not in d.files:
            print(f"  w{w}: no per-particle record (before D146_A0)")
            continue
        sp = float(d["sp0"])
        x0 = d["x0"].astype(np.float64)
        D = {k: d[k].astype(np.float64) for k in ("d10", "d20", "d30", "d40")}
        n = d["lnrm"].astype(np.float64)
        tree0 = cKDTree(x0)
        xe = x0 + D["d40"]
        treee = cKDTree(xe)
        # each particle's own set (at the window's end if apart there, else at its start), and its body neighbours: the
        # 64 nearest at the window's start outside that set
        own, nb = [], []
        for p in P:
            S = component(treee, xe, p) or component(tree0, x0, p) or {int(p)}
            _, cand = tree0.query(x0[p], 64 + len(S) + 1)
            nb.append(np.asarray([q for q in cand if q not in S][:64]))
            own.append(S)
        nb = np.asarray(nb)
        grp = np.unique(np.concatenate([np.asarray(P), nb.reshape(-1)]))
        _, nb16 = tree0.query(x0[grp], 17)
        nb16 = nb16[:, 1:]
        snaps = [x0, x0 + D["d10"], x0 + D["d20"], x0 + D["d30"]]
        pd = push_vec(snaps[:2], grp, nb16)
        pr = push_vec(snaps[2:], grp, nb16)
        ng = n[grp]
        uu = (d["u"].astype(np.float64) * d["ug"].astype(np.float64))[grp]
        rl_d = d["n_jmp_drv"].astype(np.float64)[grp] * sp - (pd * ng).sum(1)
        rl_r = d["n_jmp_rel"].astype(np.float64)[grp] * sp - (pr * ng).sum(1)
        tot_d, tot_r = D["d20"][grp], (D["d40"] - D["d20"])[grp]
        parts = {
            "grid_drv": tot_d - uu[:, None] * ng - pd - rl_d[:, None] * ng,
            "grid_rel": tot_r - pr - rl_r[:, None] * ng,
            "u": uu[:, None] * ng,
            "push": pd + pr,
            "relax": (rl_d + rl_r)[:, None] * ng,
        }
        chk = np.abs(((parts["grid_drv"] * ng).sum(1) / sp - d["n_adv_drv"].astype(np.float64)[grp])).max()
        chk2 = np.abs(((parts["grid_rel"] * ng).sum(1) / sp - d["n_adv_rel"].astype(np.float64)[grp])).max()
        idx = {q: i for i, q in enumerate(grp)}
        rows = []
        for i, p in enumerate(P):
            ip = idx[p]
            iq = np.asarray([idx[q] for q in nb[i]])
            rel = {k: v[ip] - v[iq].mean(0) for k, v in parts.items()}
            tot = (xe[p] - x0[p]) - (xe[nb[i]] - x0[nb[i]]).mean(0)
            L = np.linalg.norm(tot)
            e = tot / max(L, 1e-12)
            on = {k: float(v @ e) / sp for k, v in rel.items()}
            # the opening to its 8 nearest body particles: d|x_p - x_q| ~ (dx_p - dx_q) . (r0 + r1) / (|r0| + |r1|), exact, mean over q
            q8 = nb[i][:8]
            iq8 = np.asarray([idx[q] for q in q8])
            # exact: |r1| - |r0| = (r1 - r0) . (r1 + r0) / (|r1| + |r0|), r = x_p - x_q, linear in the displacements
            v0, v1 = x0[p] - x0[q8], xe[p] - xe[q8]
            r0, r1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
            uq = (v0 + v1) / (r0 + r1)[:, None]
            op = {k: float(np.mean(((v[ip] - v[iq8]) * uq).sum(1))) / A for k, v in parts.items()}
            a0, a1 = not attached(tree0, x0, p), not attached(treee, xe, p)
            asym = np.linalg.norm(x0[p] - x0[tree0.query(x0[p], 33)[1][1:]].mean(0)) / sp
            g = on["grid_drv"] + on["grid_rel"]
            go = op["grid_drv"] + op["grid_rel"]
            # each term's descent along the particle's normal at the window's end (-s_k g_k . n: + = outward), the
            # direction u's own gradient takes (u moves along n with an identity adjoint)
            tn = {k: -float(d["s_" + k]) * float(d["g_" + k][p].astype(np.float64) @ n[p])
                  for k in ("ot", "surf", "near", "spray", "rend") if "g_" + k in d.files}
            tsum = sum(abs(v) for v in tn.values()) or 1.0
            terms = ", ".join(f"{k} {v / tsum:+.2f}" for k, v in tn.items())
            rows.append(f"p{p}: apart {int(a0)}->{int(a1)} (own set {len(own[i])}); layer {int(float(d['lmask'][p]) > .5)} "
                        f"(asymmetry {asym:.2f} sp) ug {float(d['ug'][p]):.2f}; nearest body {r0.min() / A:.2f} -> "
                        f"{r1.min() / A:.2f} a, its 8 nearest body {r0.mean() / A:.2f} -> {r1.mean() / A:.2f} a\n"
                        f"      opening to its 8 nearest body (a): grid {go:+.2f} (driven {op['grid_drv']:+.2f}, released "
                        f"{op['grid_rel']:+.2f}), u {op['u']:+.2f}, push {op['push']:+.2f}, relaxation {op['relax']:+.2f}"
                        f" (sum {sum(op.values()):+.2f}, exact {(r1.mean() - r0.mean()) / A:+.2f})\n"
                        f"      moved {L / sp:.2f} sp against its 64 nearest body, along that: grid {g:+.2f}, u "
                        f"{on['u']:+.2f}, push {on['push']:+.2f}, relaxation {on['relax']:+.2f}\n"
                        f"      terms' descent along its normal (shares of their sum of sizes, + outward): {terms}; "
                        f"u {float(d['u'][p]) / A:+.2f} a")
            if jout:
                jout.write(json.dumps(dict(run=label, piece=P, p=int(p), window=int(w), t0=(wend[w] - STEPS) / 240.0,
                                           apart0=int(a0), apart1=int(a1), own=len(own[i]), moved=L / sp, along=on,
                                           open=op, near0=r0.min() / A, near1=r1.min() / A, near8_0=r0.mean() / A,
                                           near8_1=r1.mean() / A, layer=float(d["lmask"][p]), asym=asym,
                                           ug=float(d["ug"][p]), u=float(d["u"][p]) / A, terms_n=tn,
                                           detach_window=wdet.get(p))) + "\n")
        print(f"  w{w} ({(wend[w] - STEPS) / 240:.2f}-{wend[w] / 240:.2f} s; grid's n part vs the dump's n_adv: max "
              f"|diff| {max(chk, chk2):.3f} sp):\n    " + "\n    ".join(rows), flush=True)
