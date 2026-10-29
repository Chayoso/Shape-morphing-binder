"""d159_course.py OUT_JSONL MESH TWIN_NPZ LABEL=NPZ[:SEED] ... -- D159 R2 (CPU; measurement only): does the render change
where the material goes, over the run. On each kept frame k (every one to 4 s, then every 4th, and the runs' last ones)
the twin's ("dF_c + layer u", TWIN_NPZ, seed 97) outer-layer particles p (the pipeline's layer) are the common points;
for each other state A (another seed moved into the seed-97 frame by the mesh frames, d154/meshes):
  delta(p) = f_A(p) - f_twin(p), f a state's Zhu and Bridson field (radius 3, offset 0.8 pitches): the local displacement
             of A's surface from the twin's along the normal, no particle correspondence needed;
  its rms (pitches), and split over the twin layer's same-side graph at sigma = 2 pitches: the part below 10.8 pitches
             (the detail band, D158's relief band) and above; their rms and the detail share of the energy; also in
             the region fixed before the runs (labels from the nearest region sample of each particle);
  for a state of the twin's seed also the particles' own distance |x_A - x_twin| (rms over the twin's layer, pitches).
A = the full arm (D159 or the tag), and a twin of another seed: the floor (the twin-vs-twin seed spread)."""
import json
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.pipeline.window.layer import layer_by_asymmetry
from physmorph.render.exterior import ZhuBridson

torch.set_grad_enabled(False)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def spacing8(x):
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    return float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (20000 / len(x)) ** (1 / 3)


def frame_map(mesh, seed):
    m97, ms = np.load(f"{G}/d154/meshes/{mesh}_97.npz"), np.load(f"{G}/d154/meshes/{mesh}_{seed}.npz")
    return lambda x: ((x / float(ms["scale"]) + ms["offset"]) - m97["offset"]) * float(m97["scale"])


def low_pass(p, n, a, sigma=2.0):
    d, nb = cKDTree(p).query(p, 17)
    side = (n[nb] * n[:, None]).sum(-1) > 0.5
    s = float(np.median(d[:, 1]))
    w = np.exp(-0.5 * (d / s) ** 2) * side
    w[:, 0] = 1.0
    w /= w.sum(1, keepdims=True)
    var = float(np.median((w * d ** 2).sum(1))) / 2.0
    steps = max(1, int(round((sigma * a) ** 2 / var)))

    def lp(v):
        for _ in range(steps):
            v = (w * v[nb]).sum(1)
        return v
    return lp


def main():
    out, mesh, twin_path, specs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    R = np.load(f"{G}/d159/step0/regions_{mesh}_b5.4.npz")
    a = float(R["pitch"])
    rtree, rlab = cKDTree(R["P"]), R["region"]
    tw = np.load(twin_path, allow_pickle=True)
    TF = tw["frames"]
    arms = []
    for spec in specs:
        label, rest = spec.split("=", 1)
        seed = 97
        if ":" in rest:
            rest, seed = rest.rsplit(":", 1)
            seed = int(seed)
        z = np.load(rest, allow_pickle=True)
        arms.append((label, seed, z["frames"], frame_map(mesh, seed) if seed != 97 else (lambda x: x)))
    n = min([len(TF)] + [len(f) for _, _, f, _ in arms])
    ks = sorted(set(list(range(0, min(n, 81))) + list(range(84, n, 4)) + [n - 1]))
    pairs = [(k, k) for k in ks] + [("end", "end")]
    for k, _ in pairs:
        xt = np.asarray(TF[-1] if k == "end" else TF[k], np.float64)
        lm, ln = layer_by_asymmetry(torch.as_tensor(xt), spacing8(xt))
        sel = lm.numpy()
        p, nrm = xt[sel], ln[lm].numpy()
        P = torch.as_tensor(p)
        f_t = ZhuBridson(torch.as_tensor(xt), a)(P, grad=False)[0].numpy()
        lp = low_pass(p, nrm, a)
        reg = rlab[rtree.query(p)[1]]
        row = dict(mesh=mesh, kept=k, t_s=(None if k == "end" else k / 20.0), layer=int(sel.sum()), arms={})
        for label, seed, frames, fm in arms:
            xa = fm(np.asarray(frames[-1] if k == "end" else frames[k], np.float64))
            f_a = ZhuBridson(torch.as_tensor(xa), a)(P, grad=False)[0].numpy()
            ok = np.isfinite(f_a) & np.isfinite(f_t)
            dlt = np.where(ok, f_a - f_t, 0.0)
            low = lp(dlt)
            det = dlt - low
            e = lambda v, m=ok: float(np.sqrt((v[m] ** 2).mean()) / a)      # noqa: E731
            r = dict(rms_a=e(dlt), low_rms_a=e(low), detail_rms_a=e(det),
                     detail_share=float((det[ok] ** 2).sum() / max((dlt[ok] ** 2).sum(), 1e-30)),
                     region_detail_rms_a=e(det, ok & reg), region_rms_a=e(dlt, ok & reg), valid=float(ok.mean()))
            if seed == 97 and len(xa) == len(xt):
                r["particle_rms_a"] = float(np.sqrt(((xa[sel] - p) ** 2).sum(1).mean()) / a)
            row["arms"][label] = r
        with open(out, "a") as fo:
            fo.write(json.dumps(row) + "\n")
        print(k, {l: (round(v["rms_a"], 3), round(v["detail_rms_a"], 3), round(v["detail_share"], 2)) for l, v in row["arms"].items()},
              flush=True)


if __name__ == "__main__":
    main()
