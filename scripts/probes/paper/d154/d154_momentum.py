"""d154_momentum.py OUT_JSONL LABEL=FRAMES12_NPZ [...] -- D154: momentum over a run from its kept frames, the definitions
of tmp/momentum_probe.py (D145's mom.sh) on the CPU (numpy float64; the pitch a = 0.708 x the median 8th-neighbour
distance of every 50th target point among all of them, as there). Per pair of kept frames: the centre of mass's
displacement from the first frame (a), the net over the gross linear move |mean dx| / mean |dx|, the angular one
|mean r x dx| / mean |r| |dx| about the centre of mass, the net rotation summed so far (deg), the mean particle move (a);
then one summary line per run. The frames are read one at a time from their place in the archive."""
import json
import sys
import zipfile

import numpy as np
from numpy.lib import format as npf
from scipy.spatial import cKDTree


def frames(npz):
    with zipfile.ZipFile(npz) as zf, zf.open("frames.npy") as f:
        version = npf.read_magic(f)
        shape, _, dtype = npf._read_array_header(f, version)
        row = int(np.prod(shape[1:])) * dtype.itemsize
        for _ in range(shape[0]):
            yield np.frombuffer(f.read(row), dtype).reshape(shape[1:]).astype(np.float64)


with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        label, path = spec.split("=")
        z = np.load(path, allow_pickle=True)
        raws = [int(v) for v in z["raws"]]
        tgt = np.asarray(z["tgt"], np.float32)
        d8 = np.sort(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8].astype(np.float32))
        a = .708 * float(d8[(len(d8) - 1) // 2])                # torch.median: the lower middle value
        com0 = prev = None
        path_len, far, lin, ang, still, rot = 0., 0., [], [], [], np.zeros(3)
        for i, x in enumerate(frames(path)):
            c = x.mean(0)
            if prev is None:
                com0 = c
            else:
                dx = x - prev
                r = prev - prev.mean(0)
                gross = np.linalg.norm(dx, axis=1).mean()
                path_len += float(np.linalg.norm(c - prev.mean(0)))
                lin.append(float(np.linalg.norm(dx.mean(0)) / max(gross, 1e-30)))
                L = np.cross(r, dx).mean(0)
                ang.append(float(np.linalg.norm(L) / max((np.linalg.norm(r, axis=1) * np.linalg.norm(dx, axis=1)).mean(), 1e-30)))
                rot += L / (r ** 2).sum(1).mean() * 1.5
                still.append(float(gross) / a)
                out.write(json.dumps(dict(run=label, state=raws[i], com=float(np.linalg.norm(c - com0)) / a, lin=lin[-1],
                                          ang=ang[-1], rot=float(np.degrees(np.linalg.norm(rot))), move=still[-1])) + "\n")
            far = max(far, float(np.linalg.norm(c - com0)))
            prev = x
        end = float(np.linalg.norm(prev.mean(0) - com0))
        summ = dict(run=label, summary=1, kept=len(raws), com_far_a=far / a, com_end_a=end / a, com_path_a=path_len / a,
                    lin_max=max(lin), lin_mean=float(np.mean(lin)), lin_last20=float(np.mean(lin[-20:])),
                    ang_max=max(ang), ang_mean=float(np.mean(ang)), ang_last20=float(np.mean(ang[-20:])),
                    rot_deg=float(np.degrees(np.linalg.norm(rot))), move_last20_a=float(np.mean(still[-20:])), a=a)
        out.write(json.dumps(summ) + "\n")
        out.flush()
        print(json.dumps(summ), flush=True)
