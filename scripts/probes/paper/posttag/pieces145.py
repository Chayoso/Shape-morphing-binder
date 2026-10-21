"""pieces145.py OUT_JSONL NAME=NPZ [...] [--every K] [--threads T] -- D145 post-tag: the pieces off the body (CPU only).

For each archive (a run's frames12, or its end1 end frame) and each read frame (every K-th kept frame and always the
last): the connected sets at 1.5 pitches (pitch a = 0.708 x the target sample's median 8th-neighbour distance, the link
of d120_apart.py and visible_apart.py); every set but the largest is a piece, with its size (particles) and its gap to
the largest set (the nearest particle pair, in pitches). One JSON row per run:
  end   pieces, particles in them, the largest piece, the farthest gap, pieces of >= 10 particles, pieces >= 5 pitches off;
        each piece of >= 3 particles or >= 3 pitches: size, gap and (frames12 only) the time since when every particle of
        it has been off the body in every later read frame;
  run   the largest piece and the farthest gap over the read frames, with their times (raw / 240 s); the far pieces
        (gap >= 5 pitches): read frames holding one, the first and last such time, the largest one, the most particles
        in them in one frame, frames with a far piece of >= 10 particles, frames with any piece of >= 10 particles,
        and the per-frame timeline [t, far pieces, their particles, the largest, the farthest gap, pieces >= 10,
        far pieces >= 10].
10 particles and 5 pitches are the coordinator's marks for a serious floating piece (the 2026-10-16 brief), not a gate.
A frame whose median 8th-neighbour distance is below half a pitch (a baseline's body collapsed to a small ball) is not
read (its neighbour pairs would number ~1e9) and is counted in collapsed_frames / collapsed_s."""
import argparse
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def pieces(x, a, threads):
    """Labels, the largest set's label, and per piece (label, size, gap in pitches)."""
    tree = cKDTree(x)
    pairs = tree.query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    big = int(np.argmax(sizes))
    off = np.flatnonzero(lab != big)
    out = []
    if len(off):
        d = cKDTree(x[lab == big]).query(x[off], workers=threads)[0] / a
        cnt = np.bincount(lab[off], minlength=len(sizes))      # per set, in one pass (a loop over the sets is
        gmin = np.full(len(sizes), np.inf)                     # quadratic when a method leaves 1e5 particles apart)
        np.minimum.at(gmin, lab[off], d)
        for c in np.flatnonzero(cnt):
            out.append((int(c), int(cnt[c]), float(gmin[c])))
    return lab, big, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    for spec in args.runs:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        fr, raws, tgt = z["frames"], np.asarray(z["raws"]), np.asarray(z["tgt"], np.float64)
        a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9, workers=args.threads)[0][:, 8]))
        K = len(raws)
        ks = sorted(set(range(0, K, max(1, args.every))) | {K - 1})
        off_masks, run_big, run_far, timeline, collapsed = {}, (0, None), (0.0, None), [], []
        for k in ks:
            xk = np.asarray(fr[k], np.float64)
            sub = xk[:: max(1, len(xk) // 4000)]
            d8 = float(np.median(cKDTree(xk).query(sub, 9, workers=args.threads)[0][:, 8]))
            if d8 < 0.5 * a and k != K - 1:       # a collapsed body (a baseline's level set shrunk to a ball): 300k
                collapsed.append(round(float(raws[k]) / 240.0, 3))   # points at a fifth of the pitch make ~1e9 pairs
                off_masks[k] = np.zeros(len(xk), bool)                # at 1.5 pitches; skipped and counted, not read
                continue
            lab, big, pc = pieces(xk, a, args.threads)
            off_masks[k] = lab != big
            t = float(raws[k]) / 240.0
            far = [p for p in pc if p[2] >= 5.0]
            timeline.append([round(t, 3), len(far), sum(p[1] for p in far), max((p[1] for p in far), default=0),
                             round(max((p[2] for p in far), default=0.0), 2), sum(1 for p in pc if p[1] >= 10),
                             sum(1 for p in far if p[1] >= 10)])
            if pc:
                s = max(p[1] for p in pc)
                g = max(p[2] for p in pc)
                if s > run_big[0]:
                    run_big = (s, t)
                if g > run_far[0]:
                    run_far = (g, t)
            if k == K - 1:
                end_lab, end_big, end_pc = lab, big, pc
        listed = []
        for c, s, g in sorted(end_pc, key=lambda p: (-p[1], -p[2])):
            if s < 3 and g < 3:
                continue
            idx = np.flatnonzero(end_lab == c)
            since = None
            if len(ks) > 1:
                since = float(raws[K - 1]) / 240.0
                for k in reversed(ks):
                    if off_masks[k][idx].all():
                        since = float(raws[k]) / 240.0
                    else:
                        break
            listed.append(dict(size=s, gap_a=round(g, 2), off_since_s=None if since is None else round(since, 3)))
        row = dict(run=name, path=path, pitch=a, frames_read=len(ks), collapsed_frames=len(collapsed), collapsed_s=collapsed,
                   end_s=float(raws[K - 1]) / 240.0,
                   end_pieces=len(end_pc), end_particles=int(sum(p[1] for p in end_pc)),
                   end_largest=max((p[1] for p in end_pc), default=0),
                   end_farthest_a=round(max((p[2] for p in end_pc), default=0.0), 2),
                   end_ge10=sum(1 for p in end_pc if p[1] >= 10), end_ge5a=sum(1 for p in end_pc if p[2] >= 5),
                   end_listed=listed, run_largest=run_big[0], run_largest_s=run_big[1],
                   run_farthest_a=round(run_far[0], 2), run_farthest_s=run_far[1])
        if len(ks) > 1:                             # the far pieces over the run (>= 5 pitches off the body)
            ft = [x for x in timeline if x[1]]
            row.update(far_frames=len(ft), far_first_s=ft[0][0] if ft else None, far_last_s=ft[-1][0] if ft else None,
                       far_largest=max((x[3] for x in ft), default=0), far_particles_max=max((x[2] for x in ft), default=0),
                       far10_frames=sum(1 for x in timeline if x[6]), big10_frames=sum(1 for x in timeline if x[5]),
                       timeline=timeline)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "path"}), flush=True)


if __name__ == "__main__":
    main()
