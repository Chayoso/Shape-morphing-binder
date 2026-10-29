"""d148_link_cf.py LABEL FRAMES12 RUN_JSON DUMP_DIR EVENTS.jsonl -- would 'the layer's step keeps a particle within the
berth of its neighbours' plane' have held the detachments that became visible? (CPU)

The candidate definition: k_layer_project moves a layer particle by u and the relaxation along its frozen normal n; its
offset from its same-side neighbours' plane is k_layer_resid's d = n . (x - c), c the weighted centroid of its layer
neighbours (layer_relax_data: the 24 nearest layer particles, a Gaussian of 2 spacings times the normals' agreement).
The candidate clamps the step so |d| does not grow beyond max(berth, |d| before the step): the berth is the near band's
(objective.berth = nn_berth_k x the target's nearest-neighbour spacing = the target sample's median 8th-neighbour
distance, 'inside it is legitimate rim'); a step back toward the plane is always taken.

From d148_detach_all.py's events (EVENTS.jsonl, with blob_1s / side_1s) and D146's dump: per event, the layer graph of its
window rebuilt from the dump (x0, lmask, lnrm, sp0), d at the window's start and end, the layer step's own normal
displacement (u ug + the relaxation, as d148_detach.py splits them); the clamp bites when |d| at the end exceeds the
berth and the layer step moved it outward; then the counterfactual offset max(berth, |d0|) and how much of the opening
the clamp removes (the excess along n, in pitches) against the event's opening to its 8 nearest body particles. And, per
window read, the share of u's reach whose |d| at the end exceeds the berth (where the clamp would act at all), with their
share of |u|. P3: the same with the link read as the nearest of the particle's 16 frozen neighbours (the minimum
spacing's list, k_update's snbr) kept within max(berth, its distance at the window's start): the window-level clamp of
the layer step's part along that pair, and whether the pair then stays within the 1.5-pitch link."""
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

label, fr_path, jpath, ddir, epath = sys.argv[1:6]
z = np.load(fr_path)
tgt = np.asarray(z["tgt"], np.float64)
d8 = cKDTree(tgt).query(tgt, 9)[0][:, 8]
BERTH = float(np.median(d8))                    # = nn_berth_k x nn_spacing (prepare.sampling_berth x the median NN)
A = 0.708 * BERTH
fr0 = np.asarray(z["frames"][0], np.float64)
run = json.load(open(jpath))
cfg = next(iter(run["arms"].values()))["config"]
T = int(cfg["T"])
LK, LH = int(cfg.get("layer_k", 24)), float(cfg.get("layer_h_sp", 2.0))
R_SP = float(cfg.get("min_spacing", 0.0)) * (ConvexHull(fr0).volume / len(fr0)) ** (1.0 / 3.0)
del z, fr0
ev = [json.loads(s) for s in open(epath)]
ev = [e for e in ev if "p" in e and e.get("blob_1s")]
byw = {}
for e in ev:
    byw.setdefault(e["window"], []).append(e)
print(f"[{label}] berth {BERTH:.5f} wu = {BERTH / A:.3f} pitches; {len(ev)} detachments that became blobs within a "
      f"second ({sum(e['side_1s'] for e in ev)} side), in {len(byw)} windows", flush=True)


def push_vec(snaps, P, nb16):
    tot = np.zeros((len(P), 3))
    for xs in snaps:
        d = xs[P][:, None, :] - xs[nb16]
        L = np.linalg.norm(d, axis=2)
        ov = (L < R_SP) & (L > 1e-9)
        tot += 10 * (1.0 / T) * 0.5 * np.where(ov[..., None], ((R_SP - L) / np.where(L > 0, L, 1))[..., None] * d,
                                               0.0).sum(1)
    return tot


rows, reach = [], []
for w in sorted(byw):
    d = np.load(f"{ddir}/w_{w:04d}.npz")
    sp = float(d["sp0"])
    x0 = d["x0"].astype(np.float64)
    D = {k: d[k].astype(np.float64) for k in ("d10", "d20", "d30", "d40")}
    xe = x0 + D["d40"]
    n = d["lnrm"].astype(np.float64)
    lm = d["lmask"].astype(np.float64) > 0.5
    ug = d["ug"].astype(np.float64)
    uu = d["u"].astype(np.float64) * ug
    idx = np.nonzero(lm)[0]
    P, R = x0[idx], n[idx]
    dd, nb = cKDTree(P).query(P, LK + 1)
    dd, nb = dd[:, 1:], nb[:, 1:]
    ww = np.exp(-(dd / (LH * sp)) ** 2) * np.clip((R[nb] * R[:, None, :]).sum(-1), 0.0, None)
    tw = ww.sum(1)
    ok = tw > 1e-12
    W = np.where(ok[:, None], ww / np.where(ok, tw, 1.0)[:, None], 0.0)
    nbg = idx[nb]

    def offset(x):
        c = np.where(ok[:, None], (W[..., None] * x[nbg]).sum(1), x[idx])
        return (R * (x[idx] - c)).sum(1)

    dl0, dl1 = offset(x0), offset(xe)
    pos = {p: i for i, p in enumerate(idx)}
    on = (ug[idx] > 0.5)
    big = on & (np.abs(dl1) > BERTH)
    reach.append(dict(window=w, reach=int(on.sum()), beyond=int(big.sum()),
                      u_share=float(np.abs(uu[idx][big]).sum() / max(np.abs(uu[idx][on]).sum(), 1e-30))))
    # P3: the nearest of the 16 frozen neighbours (the minimum spacing's list) within the berth
    ir = idx[on]
    _, n16r = cKDTree(x0).query(x0[ir], 17)
    r1r = np.linalg.norm(xe[ir][:, None, :] - xe[n16r[:, 1:]], axis=2).min(1)
    r0r = np.linalg.norm(x0[ir][:, None, :] - x0[n16r[:, 1:]], axis=2).min(1)
    bn = r1r > np.maximum(BERTH, r0r)
    reach[-1].update(beyond_nn=int(bn.sum()), u_share_nn=float(np.abs(uu[ir][bn]).sum() / max(np.abs(uu[ir]).sum(), 1e-30)))
    P_ev = np.asarray([e["p"] for e in byw[w]], np.int64)
    _, nb16 = cKDTree(x0).query(x0[P_ev], 17)
    nb16 = nb16[:, 1:]
    snaps = [x0, x0 + D["d10"], x0 + D["d20"], x0 + D["d30"]]
    pv = push_vec(snaps, P_ev, nb16)
    jmp = (d["n_jmp_drv"].astype(np.float64) + d["n_jmp_rel"].astype(np.float64))[P_ev] * sp
    for i, e in enumerate(byw[w]):
        p = e["p"]
        if p not in pos:                              # off the layer: the layer's step does not move it
            rows.append(dict(e, d0=None, d1=None, layer_step=0.0, bites=0, removed=0.0, bites_nn=0, removed_nn=0.0,
                             linked_cf=0, r0n=None, r1n=None))
            continue
        k = pos[p]
        relax = jmp[i] - float(pv[i] @ n[p])
        step = float(uu[p]) + relax                   # the layer step's normal displacement over the window
        s = np.sign(dl1[k]) if dl1[k] != 0 else 1.0
        bound = max(BERTH, abs(dl0[k]))
        excess = max(0.0, abs(dl1[k]) - bound)
        bites = int(excess > 0 and step * s > 0)
        removed = min(excess, max(0.0, step * s)) / A if bites else 0.0
        q = nb16[i]
        r0n = float(np.linalg.norm(x0[p] - x0[q], axis=1).min())
        r1 = np.linalg.norm(xe[p] - xe[q], axis=1)
        qs, r1n = q[int(np.argmin(r1))], float(r1.min())
        cnn = float(step * (n[p] @ (xe[p] - xe[qs])) / max(r1n, 1e-12))
        exn = max(0.0, r1n - max(BERTH, r0n))
        bnn = int(exn > 0 and cnn > 0)
        rem = min(exn, cnn) if bnn else 0.0
        rows.append(dict(e, d0=dl0[k] / A, d1=dl1[k] / A, layer_step=step / A, bites=bites, removed=removed,
                         bites_nn=bnn, removed_nn=rem / A, r0n=r0n / A, r1n=r1n / A,
                         linked_cf=int(r1n - rem <= 1.5 * A)))


def report(name, rs):
    if not rs:
        print(f"[{label}] {name}: none")
        return
    E = len(rs)
    lay = [r for r in rs if r["d0"] is not None]
    b = [r for r in rs if r["bites"]]
    opened = [max(r["exact"], 1e-9) for r in b]
    frac = [min(1.0, r["removed"] / o) for r, o in zip(b, opened)]
    print(f"[{label}] {name}: {E}; in the layer {len(lay) / E:.2f}; |d| at the end beyond the berth "
          f"{sum(1 for r in lay if abs(r['d1']) * A > BERTH) / E:.2f} (median |d| {np.median([abs(r['d1']) for r in lay]):.2f} a, "
          f"start {np.median([abs(r['d0']) for r in lay]):.2f} a); the clamp bites {len(b) / E:.2f}; where it bites it removes "
          f"a median {np.median([r['removed'] for r in b]) if b else 0:.2f} a of the excess, {np.median(frac) if frac else 0:.2f} of "
          f"the window's opening (removes all of it in {sum(1 for f in frac if f >= 1.0) / E:.2f} of the events); by the "
          f"update that opened the link most, bites: " + ", ".join(
              f"{t} {sum(1 for r in b if r['top'] == t)}/{sum(1 for r in rs if r['top'] == t)}"
              for t in sorted({r['top'] for r in rs})), flush=True)


report("detachments that became blobs within a second", rows)
report("of them side-view blobs", [r for r in rows if r["side_1s"]])
m = {k: float(np.median([r[k] for r in reach])) for k in ("reach", "beyond", "u_share")}
print(f"[{label}] where the clamp would act at all (u's reach with |d| beyond the berth at the window's end), median over "
      f"{len(reach)} windows: {m['beyond']:.0f} of {m['reach']:.0f} particles, {m['u_share']:.3f} of |u|", flush=True)


def report_nn(name, rs):
    if not rs:
        return
    E = len(rs)
    lay = [r for r in rs if r["r0n"] is not None]
    b = [r for r in rs if r["bites_nn"]]
    print(f"[{label}] {name}, P3 (the layer step keeps the nearest of the 16 frozen neighbours within the berth): nearest "
          f"at the start a median {np.median([r['r0n'] for r in lay]):.2f} a, at the end {np.median([r['r1n'] for r in lay]):.2f} a; "
          f"beyond the berth at the end {sum(1 for r in lay if r['r1n'] * A > BERTH) / E:.2f}; the clamp bites {len(b) / E:.2f}, "
          f"removing a median {np.median([r['removed_nn'] for r in b]) if b else 0:.2f} a; the nearest then within the 1.5-pitch "
          f"link {sum(1 for r in rs if r['linked_cf']) / E:.2f} (already within it without the clamp "
          f"{sum(1 for r in lay if r['r1n'] <= 1.5) / E:.2f}: the link broken elsewhere, a set moving together); by the update "
          f"that opened the link most, bites: " + ", ".join(
              f"{t} {sum(1 for r in b if r['top'] == t)}/{sum(1 for r in rs if r['top'] == t)}"
              for t in sorted({r['top'] for r in rs})), flush=True)


report_nn("detachments that became blobs within a second", rows)
report_nn("of them side-view blobs", [r for r in rows if r["side_1s"]])
m = {k: float(np.median([r[k] for r in reach])) for k in ("beyond_nn", "u_share_nn")}
print(f"[{label}] where P3 would act at all (u's reach whose nearest frozen neighbour ends beyond max(berth, start)), median "
      f"over {len(reach)} windows: {m['beyond_nn']:.0f} particles, {m['u_share_nn']:.3f} of |u|", flush=True)
