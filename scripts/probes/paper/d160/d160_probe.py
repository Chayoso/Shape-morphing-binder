"""d160_probe.py -- D160, before the pre-registration (measurement only, CPU, no run): the colour layout the tag's own
40k runs give candidate colour scenes. Every physics term is blind to which particle is where, so the tag's runs (the
twin W, lambda = 0, and the render arm V without colour) already show where the transport puts each piece of source
material: colour the source sample (frame 0) by a scene's source field, carry the colours by index to the delivered
frame, read them on the delivered state's exterior discs (Zhu-Bridson, radius 3, offset 0.8 pitch, the disc's own
kernel weights) and compare with the scene's target field at the disc. Also: the bunny's ear base and head direction.
One JSON line per (mesh, run, scene) on stdout."""
import json
import sys

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/repo_t1")
from physmorph.render.exterior import Lattice, make_field  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
torch.set_num_threads(6)


def pitch_of(x):
    d, _ = cKDTree(x).query(x, 9)
    return .708 * float(np.median(d[:, 8]))


def discs(x, a):
    xt = torch.as_tensor(np.asarray(x, np.float32))
    c = xt.mean(0)
    R = float((xt - c).norm(dim=1).max())
    f = make_field("zb", xt, a)
    p, g, _, _ = Lattice(c, 2.8 * R).discs(f, .5 * a, refine=False)
    return xt, p, f.neighbours(p, a)


def share(xt, p, idx, s, a):
    R = 3. * a
    valid = idx >= 0
    q = xt[idx.clamp_min(0)]
    w = ((1. - ((p[:, None] - q) ** 2).sum(-1) / (R * R)).clamp_min(0.) ** 3) * valid
    st = torch.as_tensor(np.asarray(s, np.float32))
    return ((w * st[idx.clamp_min(0)]).sum(1) / w.sum(1).clamp_min(1e-12)).numpy()


def components(pts, link):
    if len(pts) < 2:
        return len(pts), np.zeros(len(pts), int)
    t = cKDTree(pts)
    pairs = t.query_pairs(link, output_type="ndarray")
    n, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2),
                                  directed=False)
    return n, lab


def ear_base(tgt, a):
    """The highest height at which a one-pitch slab of the target sample is one piece (of pieces >= 20 points),
    scanning down from the top: above it the slab has two big pieces (the ears)."""
    y = tgt[:, 1]
    top = y.max()
    hs = np.arange(top - 2 * a, y.min(), -.5 * a)
    two_seen = False
    for h in hs:
        sl = tgt[(y >= h) & (y < h + a)]
        n, lab = components(sl, 1.5 * a)
        big = int((np.bincount(lab) >= 20).sum()) if len(sl) else 0
        if big >= 2:
            two_seen = True
        if two_seen and big == 1:
            return float(h + a), float((y >= h + a).mean())
    return None, None


def head_dir(tgt, yb, a):
    """The horizontal direction from the body's centroid to the centroid of the slab just below the ear base (the
    head), unit (x, z)."""
    c = tgt.mean(0)
    sl = tgt[(tgt[:, 1] < yb) & (tgt[:, 1] >= yb - 6 * a)]
    d = sl.mean(0) - c
    d[1] = 0.
    return d / np.linalg.norm(d), sl.mean(0).tolist(), c.tolist()


def scenes(src, tgt, yb, hd):
    """Candidate scenes: (source red mask on the source sample, target field: position -> red bool)."""
    ys = src[:, 1]
    ph_s = src @ hd
    ph_t_med = float(np.median(tgt @ hd))
    out = {}
    for f in (.2, .3, .5):
        q = np.quantile(ys, 1 - f)
        out[f"ears_top{int(f * 100)}"] = (ys >= q, lambda p, yb=yb: p[:, 1] >= yb)
    out["turn_top_to_front"] = (ys >= np.median(ys), lambda p: p @ hd >= ph_t_med)
    out["turn_top_to_back"] = (ys >= np.median(ys), lambda p: p @ hd < ph_t_med)
    out["front20_to_ears"] = (ph_s >= np.quantile(ph_s, .8), lambda p, yb=yb: p[:, 1] >= yb)
    out["top_half_to_top_half"] = (ys >= np.median(ys), lambda p: p[:, 1] >= float(np.median(tgt[:, 1])))
    return out


def read(label, frames, src, tgt, yb, hd, a):
    xt, p, idx = discs(frames, a)
    pn = p.numpy()
    rows = []
    for name, (sred, tfield) in scenes(src, tgt, yb, hd).items():
        s = share(xt, p, idx, sred.astype(np.float32), a)
        want = tfield(pn)
        mis = float(((s >= .5) != want).mean())
        rows.append(dict(run=label, scene=name, discs=len(pn), src_red=float(sred.mean()), want_red=float(want.mean()),
                         shown_red=float((s >= .5).mean()), mis=mis, mixed=float(((s > .1) & (s < .9)).mean())))
    return rows


def floor_rows(tgt, yb, hd, a):
    """The target sample itself, coloured by each scene's target field: the floor of the mis-coloured share."""
    xt, p, idx = discs(tgt, a)
    pn = p.numpy()
    rows = []
    for name, (_, tfield) in scenes(tgt, tgt, yb, hd).items():
        s = share(xt, p, idx, tfield(tgt).astype(np.float32), a)
        rows.append(dict(run="target_floor", scene=name, mis=float(((s >= .5) != tfield(pn)).mean())))
    return rows


def main():
    mesh = sys.argv[1]
    runs = sys.argv[2:]
    first = np.load(runs[0].split("=")[1])
    tgt = np.asarray(first["tgt"], np.float32)
    a = pitch_of(tgt)
    if mesh == "bunny":
        yb, above = ear_base(tgt, a)
    else:
        yb, above = float(np.quantile(tgt[:, 1], .9)), .1
    hd, head_c, body_c = head_dir(tgt, yb, a)
    print(json.dumps(dict(mesh=mesh, pitch=a, ear_base=yb, vol_above=above, head_dir=hd.tolist(), head_c=head_c,
                          body_c=body_c, bbox=[tgt.min(0).tolist(), tgt.max(0).tolist()])), flush=True)
    for r in floor_rows(tgt, yb, hd, a):
        print(json.dumps(dict(mesh=mesh, **r)), flush=True)
    for spec in runs:
        label, path = spec.split("=")
        z = np.load(path)
        X = z["frames"]
        if int(z["raws"][0]) != 0 or len(X) < 2:
            print(json.dumps(dict(mesh=mesh, run=label, skipped="no source frame")), flush=True)
            continue
        src = np.asarray(X[0], np.float32)                    # frame 0 of a frames12 archive: the run's own source
        for r in read(label, np.asarray(X[-1], np.float32), src, tgt, yb, hd, a):
            print(json.dumps(dict(mesh=mesh, **r)), flush=True)


if __name__ == "__main__":
    main()
