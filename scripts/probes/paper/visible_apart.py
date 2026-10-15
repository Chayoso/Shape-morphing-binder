"""visible_apart.py -- which particles apart from the body would a viewer see? (the paper's harness reader; CPU only)

usage:
  python scripts/probes/paper/visible_apart.py RUN_frames12.npz [MORE.npz ...]
         [--w0 3] [--w1 10] [--every 1] [--jsonl OUT.jsonl] [--label NAME ...]
         [--sheet OUT.jpg --frames-dir SIDE_RENDER_FRAMES] [--threads 16]

Input: a run's kept frames (frames12: frames (K, N, 3), raws (K,) the raw step of each, tgt (N, 3) the target sample;
any particle method's run in this format works: ours, Xu et al.'s through --baseline xu, an OT baseline). Read: every
kept frame whose window (raw / 40) lies in [w0, w1] (every --every-th of them).

Per frame:
  apart   the particles outside the largest set connected at 1.5 pitches (pitch a = 0.708 x the target sample's median
          8th-neighbour distance; the count d120_apart.py and the evidence harness report)
  hidden  apart, but inside the display surface of all the other particles: the display's own field
          (render/exterior.py, Zhu and Bridson with Solenthaler et al.'s factor, radius 3, offset 0.8 pitches) of every
          particle except those of the particle's own connected set is negative at it
  bump    apart, outside that surface, joined to it: the full field stays negative on the segment from the particle to
          the nearest particle outside its set; its height = the field's value of the others at it (pitches)
  blob    apart, outside, and the full field turns positive on that segment: its own separate piece of surface (what a
          viewer reads as a floating piece)
and, for the side camera of render_deliverable.py (azimuth 90, elevation 18, distance 3.6 radii), whether a bump or blob
is occluded (a particle nearer by two pitches over its blob's footprint).

Output: per run and window (the union of its kept frames, a particle counted once at its most visible) the counts
apart / hidden / bump / blob, bumps above 0.5 and 1 pitch, the bumps' median and p90 height, the side view's unoccluded
bumps and blobs; then the same over the whole range (unique particles). --jsonl writes one JSON row per run and window
and one for the range. --sheet crops the unoccluded blobs (then the highest bumps) from a run's side render frames
(FRAMES_DIR/NNNN.jpg, NNNN the kept frame's index), 32 tiles; with several runs only the first is drawn.

D148's forensic (docs/experiments.md, 2026-10-15): about 90 % of the early apart particles of our runs are bumps of
about half a pitch (u's outer-layer relief), 1-2 % are blobs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from physmorph.render.exterior import ev_max, make_field, sol_factor  # noqa: E402

RADIUS, OFFSET, LINK = 3.0, 0.8, 1.5          # the display's field (pitches) and the apart count's link (pitches)


def side_camera(tgt):
    """render_deliverable.py's camera at azimuth 90 (pixels, depth, pixels per world unit at a depth)."""
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
    return pixels, (lambda p: (p - cam) @ fwd), (lambda zc: W / (2.0 * zc * tan_x))


def field_without_own_set(x, q, own, cl, a, tree):
    """The display field at points q of every particle except those of each point's own connected set."""
    R, off = RADIUS * a, OFFSET * a
    out = np.full(len(q), np.inf)
    keep, xb_l, J_l = [], [], []
    for i, nb in enumerate(tree.query_ball_point(q, R)):
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
        keep.append(i)
        xb_l.append(xb)
        J_l.append(J)
    if keep:
        f, _ = sol_factor(ev_max(torch.as_tensor(np.stack(J_l))))
        out[keep] = np.linalg.norm(q[keep] - np.stack(xb_l), axis=1) - off * f.numpy()
    return out


def classify(x, a, camera=None):
    """(apart indices, kind 0 hidden / 1 bump / 2 blob, height in pitches, occluded in the side view)."""
    tree = cKDTree(x)
    pairs = tree.query_pairs(LINK * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    ap = np.nonzero(cl != np.bincount(cl).argmax())[0]
    if not len(ap):
        return ap, np.zeros(0, np.int8), np.zeros(0), np.zeros(0, bool)
    height = field_without_own_set(x, x[ap], cl[ap], cl, a, tree) / a
    near = np.zeros((len(ap), 3))
    for i, p in enumerate(ap):                       # the nearest particle outside its own set
        k = 8
        while True:
            _, ii = tree.query(x[p], min(k, len(x)))
            ii = np.atleast_1d(ii)
            o = ii[cl[ii] != cl[p]]
            if len(o) or k >= len(x):
                near[i] = x[o[0]] if len(o) else x[p]
                break
            k *= 4
    full = make_field("zbs", torch.as_tensor(x, dtype=torch.float32), a, radius=RADIUS, offset=OFFSET)
    ts = np.arange(1, 8) / 8.0
    seg = x[ap][:, None, :] + ts[None, :, None] * (near - x[ap])[:, None, :]
    phi = full(torch.as_tensor(seg.reshape(-1, 3), dtype=torch.float32), grad=False)[0].numpy().reshape(len(ap), -1)
    kind = np.where(height < 0, 0, np.where((phi > 0).any(1), 2, 1)).astype(np.int8)
    occl = np.zeros(len(ap), bool)
    if camera is not None and (kind > 0).any():
        pixels, depth, ppw = camera
        pall, dall = pixels(x), depth(x)
        t2 = cKDTree(pall)
        for i in np.nonzero(kind > 0)[0]:
            p = ap[i]
            nb = np.asarray(t2.query_ball_point(pall[p], OFFSET * a * ppw(dall[p])), np.int64)
            occl[i] = bool((dall[nb] < dall[p] - 2.0 * a).any())
    return ap, kind, height, occl


def summary(rec):
    """counts of {particle: (kind, height, occluded)}"""
    if not rec:
        return dict(apart=0, hidden=0, bump=0, blob=0, bump_gt_half=0, bump_gt_one=0, bump_h_median=None,
                    bump_h_p90=None, side_bump=0, side_blob=0)
    k = np.array([v[0] for v in rec.values()])
    h = np.array([v[1] for v in rec.values()])
    o = np.array([v[2] for v in rec.values()])
    hb = h[k == 1]
    return dict(apart=int(len(k)), hidden=int((k == 0).sum()), bump=int((k == 1).sum()), blob=int((k == 2).sum()),
                bump_gt_half=int((hb > 0.5).sum()), bump_gt_one=int((hb > 1.0).sum()),
                bump_h_median=float(np.median(hb)) if len(hb) else None,
                bump_h_p90=float(np.quantile(hb, 0.9)) if len(hb) else None,
                side_bump=int(((k == 1) & ~o).sum()), side_blob=int(((k == 2) & ~o).sum()))


def merge(rec, ap, kind, height, occl):
    for p, c, h, oc in zip(ap.tolist(), kind.tolist(), height.tolist(), occl.tolist()):
        old = rec.get(p)
        if old is None or c > old[0] or (c == old[0] and h > old[1]):
            rec[p] = (c, h, oc if old is None else (oc and old[2]))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("runs", nargs="+", help="frames12 npz files (frames, raws, tgt)")
    ap_.add_argument("--label", nargs="*", default=None, help="a name per run (default: the file's stem)")
    ap_.add_argument("--w0", type=float, default=3.0, help="first window (raw / 40), default 3")
    ap_.add_argument("--w1", type=float, default=10.0, help="last window, default 10")
    ap_.add_argument("--every", type=int, default=1, help="read every n-th kept frame in the range")
    ap_.add_argument("--jsonl", default="", help="write one JSON row per run and window, and per run over the range")
    ap_.add_argument("--sheet", default="", help="a contact sheet (jpg) of the first run's visible ones")
    ap_.add_argument("--frames-dir", default="", help="the first run's side render frames (NNNN.jpg)")
    ap_.add_argument("--threads", type=int, default=16)
    args = ap_.parse_args()
    torch.set_num_threads(args.threads)
    labels = args.label or [os.path.basename(r).replace("_frames12.npz", "") for r in args.runs]
    out = open(args.jsonl, "w") if args.jsonl else None
    for ri, (label, path) in enumerate(zip(labels, args.runs)):
        z = np.load(path)
        fr, raws = z["frames"], np.asarray(z["raws"])
        tgt = np.asarray(z["tgt"], np.float64)
        a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
        camera = side_camera(tgt)
        per_w, overall, tiles_src = {}, {}, []
        ks = [k for k in range(len(raws)) if args.w0 <= raws[k] / 40.0 <= args.w1][::max(1, args.every)]
        for k in ks:
            x = np.asarray(fr[k], np.float64)
            ap, kind, height, occl = classify(x, a, camera)
            W = int(raws[k] // 40)
            merge(per_w.setdefault(W, {}), ap, kind, height, occl)
            merge(overall, ap, kind, height, occl)
            if args.sheet and ri == 0:
                for p, c, h, oc in zip(ap.tolist(), kind.tolist(), height.tolist(), occl.tolist()):
                    if c > 0 and not oc:
                        tiles_src.append((c, h, k, p, x[p]))
        for W in sorted(per_w):
            s = summary(per_w[W])
            print(f"{label} w{W}: " + ", ".join(f"{key} {val:.2f}" if isinstance(val, float) else f"{key} {val}"
                                                 for key, val in s.items()), flush=True)
            if out:
                out.write(json.dumps(dict(run=label, path=path, window=W, pitch=a, **s)) + "\n")
        s = summary(overall)
        print(f"{label} windows {args.w0:g}-{args.w1:g}, unique: " + ", ".join(
            f"{key} {val:.2f}" if isinstance(val, float) else f"{key} {val}" for key, val in s.items()), flush=True)
        if out:
            out.write(json.dumps(dict(run=label, path=path, window="range", w0=args.w0, w1=args.w1, pitch=a, **s)) + "\n")
        if args.sheet and ri == 0 and tiles_src and args.frames_dir:
            from PIL import Image, ImageDraw
            tiles_src.sort(key=lambda t: (-t[0], -t[1]))
            seen, tiles = set(), []
            for c, h, k, p, xp in tiles_src:
                if p in seen or len(tiles) >= 32:
                    continue
                seen.add(p)
                try:
                    im = Image.open(os.path.join(args.frames_dir, f"{k:04d}.jpg")).convert("RGB")
                except OSError:
                    continue
                px, py = camera[0](xp[None])[0]
                crop = im.crop((int(px) - 80, int(py) - 80, int(px) + 80, int(py) + 80)).resize((240, 240))
                dr = ImageDraw.Draw(crop)
                dr.ellipse((112, 112, 128, 128), outline=(220, 30, 30))
                dr.text((4, 4), f"{'blob' if c == 2 else 'bump'} p{p} {raws[k] / 240:.2f}s +{h:.1f}a", fill=(0, 0, 0))
                tiles.append(crop)
            if tiles:
                sheet = Image.new("RGB", (240 * 8, 240 * ((len(tiles) + 7) // 8)), (255, 255, 255))
                for i, t in enumerate(tiles):
                    sheet.paste(t, ((i % 8) * 240, (i // 8) * 240))
                sheet.save(args.sheet, quality=90)
                print(f"{label}: sheet {args.sheet} ({len(tiles)} tiles)", flush=True)
    if out:
        out.close()


if __name__ == "__main__":
    main()
