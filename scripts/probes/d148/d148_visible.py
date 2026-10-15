"""d148_visible.py [dump=FB_DUMP_DIR] [sheet=OUT.jpg:FRAMES_DIR] LABEL=FRAMES12 [...] -- which of the early particles apart
from the body (windows 3-10 by d120_apart's count: outside the largest set connected at 1.5 pitches) does the zbs display
show? (CPU, the display's own field: render/exterior.py ZhuBridsonSolenthaler, radius 3, offset 0.8 pitches, the pitch
0.708 x the target sample's median 8th-neighbour distance, as render_deliverable.py.) Per apart particle at every kept
frame of windows 3-10: hidden where the field of the body alone (the particles not apart) is negative at it (inside the
body's display surface); else a blob where the field of all the particles turns positive somewhere on the segment from it
to the nearest body particle (its own piece of surface, a floating Gaussian), else a bump (outside, joined to the body's
surface). Prints per window the apart particles (the union over its kept frames) and how many are hidden / bump / blob.
dump=: D144's per-particle dump of the same run (FB): for the visible ones, at the window they first stand apart, u times
its transport gate, the outer layer, the gate, their normal displacement less their neighbours' (spacings, normals by the
asymmetry rule as d146_fbdump.py) and dF_c's rms, against the hidden apart particles and the body. sheet=: a contact sheet
of the visible ones (blobs first) cropped from the run's side render (azimuth 90)."""
import math
import sys

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from physmorph.render.exterior import make_field

torch.set_num_threads(16)
opts = dict(a.split("=", 1) for a in sys.argv[1:] if a.split("=", 1)[0] in ("dump", "sheet"))
runs = [a.split("=", 1) for a in sys.argv[1:] if a.split("=", 1)[0] not in ("dump", "sheet")]


def camera(tgt):
    center = tgt.mean(0)
    radius = float(np.linalg.norm(tgt - center, axis=1).max())
    W, H, AZ, EL = 3840, 2160, math.radians(90.0), math.radians(18.0)
    cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
    fwd = (center - cam) / np.linalg.norm(center - cam)
    right = np.cross([0.0, 1.0, 0.0], fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    tan_y = math.tan(math.radians(15.0))
    tan_x = tan_y * W / H

    def pixels(p):
        d = p - cam
        xc, yc, zc = d @ right, d @ up, d @ fwd
        return np.stack((((xc / zc) / tan_x + 1) / 2 * W - 0.5, (1 - (yc / zc) / tan_y) / 2 * H - 0.5), 1)
    pixels.depth = lambda p: (p - cam) @ fwd                                   # noqa: E731
    pixels.px_per_wu = lambda zc: W / (2.0 * zc * tan_x)                        # noqa: E731
    return pixels


def phi_without(x, q, own, cl, a, tree):
    """the zbs field (Zhu and Bridson with Solenthaler's factor, radius 3, offset 0.8 pitches; render/exterior.py's
    _parts, ev_max and sol_factor) at the points q of all the particles except those of each point's own connected set
    (own[i] the set's label, cl every particle's)."""
    from physmorph.render.exterior import ev_max, sol_factor
    R, off = 3.0 * a, 0.8 * a
    out = np.full(len(q), np.inf)
    lists = tree.query_ball_point(q, R)
    S_l, xb_l, J_l, keep = [], [], [], []
    for i, nb in enumerate(lists):
        nb = np.asarray(nb, np.int64)
        nb = nb[cl[nb] != own[i]]
        if not len(nb):
            continue
        u = q[i] - x[nb]
        t = np.clip(1.0 - (u * u).sum(1) / (R * R), 0.0, None)
        w = t ** 3
        S = w.sum()
        if S <= 1e-6:
            continue
        xb = (w[:, None] * x[nb]).sum(0) / S
        dw = (-6.0 / (R * R)) * (t * t)[:, None] * u
        J = (x[nb].T @ dw - np.outer(xb, dw.sum(0))) / S
        S_l.append(S); xb_l.append(xb); J_l.append(J); keep.append(i)
    if keep:
        ev = ev_max(torch.as_tensor(np.stack(J_l)))
        f, _ = sol_factor(ev)
        xb = np.stack(xb_l)
        out[keep] = np.linalg.norm(q[keep] - xb, axis=1) - off * f.numpy()
    return out


def classify(x, a):
    """per particle apart from the largest set: hidden (inside the display surface of all the other particles: the
    field without its own connected set negative at it), blob (outside it, and the full field turns positive between it
    and the nearest particle outside its set: a separate piece), bump (outside it, joined); and its set's size."""
    tree = cKDTree(x)
    pairs = tree.query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    sizes = np.bincount(cl)
    ap = np.nonzero(cl != sizes.argmax())[0]
    if not len(ap):
        return ap, np.zeros(0, np.int8), np.zeros(0), np.zeros(0, np.int64)
    phi_r = phi_without(x, x[ap], cl[ap], cl, a, tree)
    xt = torch.as_tensor(x, dtype=torch.float32)
    fa = make_field("zbs", xt, a)
    nb = np.zeros((len(ap), 3))
    for i, p in enumerate(ap):                      # the nearest particle outside its own set
        k = 8
        while True:
            _, ii = tree.query(x[p], min(k, len(x)))
            ii = np.atleast_1d(ii)
            o = ii[cl[ii] != cl[p]]
            if len(o) or k >= len(x):
                nb[i] = x[o[0]] if len(o) else x[p]
                break
            k *= 4
    ts = np.arange(1, 8) / 8.0
    seg = x[ap][:, None, :] + ts[None, :, None] * (nb - x[ap])[:, None, :]
    phi_s = fa(torch.as_tensor(seg.reshape(-1, 3), dtype=torch.float32), grad=False)[0].numpy().reshape(len(ap), -1)
    kind = np.where(phi_r < 0, 0, np.where((phi_s > 0).any(1), 2, 1)).astype(np.int8)   # 0 hidden, 1 bump, 2 blob
    return ap, kind, phi_r / a, sizes[cl[ap]]


for label, path in runs:
    z = np.load(path)
    fr, raws = z["frames"], np.asarray(z["raws"])
    tgt = np.asarray(z["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
    pixels = camera(tgt)
    per_w, per_h, per_s, per_z, first, vis_log = {}, {}, {}, {}, {}, []
    for k in range(len(raws)):
        w = raws[k] / 40.0
        if not 3.0 <= w <= 10.0:
            continue
        x = np.asarray(fr[k], np.float64)
        ap, kind, out, csz = classify(x, a)
        occl = np.zeros(len(ap), bool)                 # the side view: a nearer particle over its blob's footprint
        vis_i = np.nonzero(kind > 0)[0]
        if len(vis_i):
            pall, dall = pixels(x), pixels.depth(x)
            t2 = cKDTree(pall)
            for i in vis_i:
                p = ap[i]
                r = 0.8 * a * pixels.px_per_wu(dall[p])
                nb = np.asarray(t2.query_ball_point(pall[p], r), np.int64)
                occl[i] = bool((dall[nb] < dall[p] - 2.0 * a).any())
        W = int(w)
        d = per_w.setdefault(W, {})
        hgt = per_h.setdefault(W, {})
        sv = per_s.setdefault(W, {})
        zz = per_z.setdefault(W, {})
        for p, c, o, oc, sz in zip(ap.tolist(), kind.tolist(), out.tolist(), occl.tolist(), csz.tolist()):
            if c > 0:
                zz[p] = (max(zz.get(p, (0, 0))[0], c), sz)
            if c > 0 and not oc:
                sv[p] = max(sv.get(p, 0), c)
            d[p] = max(d.get(p, 0), c)
            if c == 1:
                hgt[p] = max(hgt.get(p, -9.0), o)
            if c > 0:
                vis_log.append((c, o, k, p, x[p], oc))
                first.setdefault(p, int(raws[k] // 40) + 1)
        print(f"  {label} frame {k} (raw {raws[k]}, window {w:.1f}): apart {len(ap)}, hidden {int((kind == 0).sum())}, "
              f"bump {int((kind == 1).sum())}, blob {int((kind == 2).sum())}", flush=True)
    rows = []
    for W in sorted(per_w):
        v = np.array(list(per_w[W].values()), int)
        h = np.array(list(per_h.get(W, {}).values()), float)
        hq = (f" (outside the body surface by median {np.median(h):.2f}, p90 {np.quantile(h, .9):.2f} a; > 0.5 a "
              f"{int((h > .5).sum())}, > 1 a {int((h > 1.).sum())})") if len(h) else ""
        sv = np.array(list(per_s.get(W, {}).values()), int)
        zv = np.array(list(per_z.get(W, {}).values()), int).reshape(-1, 2)
        zq = (f" {{visible by set size: 1: {int((zv[:, 1] == 1).sum())}, 2-8: {int(((zv[:, 1] > 1) & (zv[:, 1] <= 8)).sum())}, "
              f"9-64: {int(((zv[:, 1] > 8) & (zv[:, 1] <= 64)).sum())}, >64: {int((zv[:, 1] > 64).sum())}}}")
        rows.append(f"w{W}: apart {len(v)} hidden {int((v == 0).sum())} bump {int((v == 1).sum())}{hq} blob "
                    f"{int((v == 2).sum())} [side view, not occluded: bump {int((sv == 1).sum())} blob {int((sv == 2).sum())}]{zq}")
    allv = {}
    for d in per_w.values():
        for p, c in d.items():
            allv[p] = max(allv.get(p, 0), c)
    v = np.array(list(allv.values()), int)
    print(f"== {label}: windows 3-10, per window (union of its kept frames): " + "; ".join(rows))
    print(f"   over windows 3-10: apart {len(v)} unique, hidden {int((v == 0).sum())}, bump {int((v == 1).sum())}, "
          f"blob {int((v == 2).sum())}", flush=True)
    if "dump" in opts:
        from scipy.spatial import cKDTree as KD
        vis = np.asarray(sorted(p for p, c in allv.items() if c > 0), np.int64)
        hid = np.asarray(sorted(p for p, c in allv.items() if c == 0), np.int64)
        rec = {"visible": [], "hidden": [], "body": []}
        for W in sorted(set(first.values())):
            try:
                dz = np.load(f"{opts['dump']}/d137_{W:04d}.npz")
            except OSError:
                continue
            x0, x1 = dz["x_start"].astype(np.float64), dz["x"].astype(np.float64)
            tree = KD(x0)
            _, nb = tree.query(x0, 33, workers=16)
            off = x0 - x0[nb[:, 1:]].mean(1)
            nrm = off / (np.linalg.norm(off, axis=1)[:, None] + 1e-12)
            dn = ((x1 - x0) * nrm).sum(1)
            rel = dn - np.median(dn[nb[:, 1:17]], axis=1)
            u = dz["u"].astype(np.float64) * dz["u_gate"].astype(np.float64)
            for key, sel in (("visible", vis[[first[p] == W for p in vis]] if len(vis) else vis),
                             ("hidden", hid), ("body", np.random.default_rng(W).choice(len(x0), 2000, replace=False))):
                for p in sel:
                    rec[key].append((u[p] / a, float(dz["lmask"][p]) > 0.5, float(dz["u_gate"][p]) > 0.5, rel[p] / a,
                                     float(dz["dfc_rms"][p])))
        for key, r in rec.items():
            if not r:
                continue
            r = np.array(r, float)
            print(f"   {key} ({len(r)} particle-windows): u x gate median {np.median(r[:, 0]):+.3f} (|.| p90 "
                  f"{np.quantile(np.abs(r[:, 0]), .9):.3f}) pitches; outer layer {r[:, 1].mean():.2f}, u gate on "
                  f"{r[:, 2].mean():.2f}; normal displacement less the neighbours' median {np.median(r[:, 3]):+.3f} "
                  f"(p90 {np.quantile(r[:, 3], .9):+.3f}) pitches; dF_c rms median {np.median(r[:, 4]):.2e}", flush=True)
    if "sheet" in opts:
        from PIL import Image, ImageDraw
        out, fdir = opts["sheet"].split(":", 1)
        vis_log.sort(key=lambda t: (-t[0], -t[1]))
        seen, tiles = set(), []
        for c, o, k, p, xp, oc in vis_log:
            if p in seen or len(tiles) >= 32 or oc:
                continue
            seen.add(p)
            try:
                im = Image.open(f"{fdir}/{k:04d}.jpg").convert("RGB")
            except OSError:
                continue
            px, py = pixels(xp[None])[0]
            crop = im.crop((int(px) - 80, int(py) - 80, int(px) + 80, int(py) + 80)).resize((240, 240))
            dr = ImageDraw.Draw(crop)
            dr.ellipse((112, 112, 128, 128), outline=(220, 30, 30))
            dr.text((4, 4), f"{'blob' if c == 2 else 'bump'} p{p} {raws[k] / 240:.2f}s +{o:.1f}a", fill=(0, 0, 0))
            tiles.append(crop)
        if tiles:
            cols = 8
            sheet = Image.new("RGB", (240 * cols, 240 * ((len(tiles) + cols - 1) // cols)), (255, 255, 255))
            for i, t in enumerate(tiles):
                sheet.paste(t, ((i % cols) * 240, (i // cols) * 240))
            sheet.save(out, quality=90)
            print(f"   sheet {out}: {len(tiles)} tiles", flush=True)
