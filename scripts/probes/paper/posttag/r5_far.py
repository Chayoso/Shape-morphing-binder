"""r5_far.py NAME [...] -- D145 reading 5 (1), observed: the particles >= 5 pitches off the body in a cell-size run
(output/gpu/d145t_cell/NAME_frames12.npz): when, where, and whether they rejoin (CPU; scipy).

Per kept frame (every 12 raws) up to the last frame with such particles + 1 s: the sets linked at 1.5 pitches (pitch from
the independent sample, as pieces145.py); the particles of every set but the largest whose nearest particle of the
largest set is >= 5 pitches away ("far"). For the frame with the most far particles: their sets (sizes), the sets'
centres in the target's box (0..1 per axis) and gap; then those same particles followed (by index) through the later
kept frames: the share still apart (outside the largest set) and still far, the first time all of them are back in the
largest set. One JSON line per run."""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
G = f"{B}/output/gpu"


def sets(x, a):
    pr = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr), np.int8), (pr[:, 0], pr[:, 1])), shape=(len(x), len(x))),
                                  directed=False)
    big = int(np.argmax(np.bincount(lab)))
    return lab, big


for name in sys.argv[1:]:
    mesh = name.split("300k_")[0]
    I = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=6)[0][:, 8]))
    z = np.load(f"{G}/d145t_cell/{name}_frames12.npz", mmap_mode="r")
    raws = [int(r) for r in z["raws"]]
    tgt = np.asarray(z["tgt"], np.float64)
    lo, hi = tgt.min(0), tgt.max(0)
    timeline, best = [], None
    for k, r in enumerate(raws):
        if r > 6 * 240:
            break
        x = np.asarray(z["frames"][k], np.float64)
        lab, big = sets(x, a)
        ap = np.nonzero(lab != big)[0]
        if not len(ap):
            timeline.append((r / 240, 0, 0))
            continue
        d = cKDTree(x[lab == big]).query(x[ap], 1, workers=6)[0] / a
        far = ap[d >= 5]
        timeline.append((r / 240, int(len(far)), int(len(np.unique(lab[far])))))
        if len(far) and (best is None or len(far) > len(best[1])):
            sz = np.bincount(lab)
            groups = []
            for s in np.unique(lab[far]):
                m = far[lab[far] == s]
                groups.append(dict(size=int(sz[s]), far=int(len(m)),
                                   centre=[round(float(v), 2) for v in (x[m].mean(0) - lo) / (hi - lo)],
                                   gap_a=round(float(d[np.isin(ap, m)].min()), 1)))
            best = (k, far, sorted(groups, key=lambda g: -g["far"]))
    out = dict(run=name, pitch=a, timeline=[t for t in timeline if t[1]], frames_with_far=sum(1 for t in timeline if t[1]))
    if best:
        k0, far, groups = best
        out.update(peak_t=raws[k0] / 240, peak_far=int(len(far)), peak_sets=groups[:12])
        follow, back = [], None
        for k in range(k0 + 1, len(raws)):
            x = np.asarray(z["frames"][k], np.float64)
            lab, big = sets(x, a)
            apart = lab[far] != big
            if apart.any():
                d = cKDTree(x[lab == big]).query(x[far[apart]], 1, workers=6)[0] / a
                still_far = int((d >= 5).sum())
            else:
                still_far = 0
            follow.append((raws[k] / 240, round(float(apart.mean()), 3), still_far))
            if not apart.any():
                back = raws[k] / 240
                break
            if raws[k] - raws[k0] > 240 * 3:
                break
        out.update(follow=follow[:: max(1, len(follow) // 30)], all_back_s=back)
        x = np.asarray(z["frames"][len(raws) - 1], np.float64)
        lab, big = sets(x, a)
        out["apart_at_end"] = int((lab[far] != big).sum())
    print(json.dumps(out), flush=True)
