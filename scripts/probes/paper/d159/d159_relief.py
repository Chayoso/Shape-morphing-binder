"""d159_relief.py OUT_JSONL MESH LABEL=NPZ[@FRAME][:SEED] ... -- D159 R1 (CPU; measurement only): how much of the mesh's
relief the particles' own surface carries, by band and by region (the regions fixed before the runs,
step0/regions_MESH_b5.4.npz: the mesh's area samples P, normals N, relief bands, labels).

A state is the kept frame FRAME of NPZ (default its last; `tgt` = its target sample, the run's own cap) of a run at SEED
(default 97); a state of another seed is moved into the seed-97 frame by the two seeds' mesh frames (d154/meshes:
world = (oriented mesh - offset) x scale). At every mesh sample P the state's surface height above P along N:
  exterior  t = -f(P) / |grad f(P)|, f Zhu and Bridson's field of the particles at the run's pitch (radius 3, offset
            0.8): the surface the loss and the metrics read;
  layer     t = the Gaussian-weighted (sigma one pitch) mean of (x_j - P) . N over the 8 nearest outer-layer particles
            (the pipeline's layer): the particles themselves, as the display carries them (D158's weighting).
t and the mesh's relief r are split alike on the mesh samples' graph (regions' operator) into the bands below 2.7 /
2.7-5.4 / 5.4-10.8 pitches; per band the carried share k = 1 + cov(t_b, r_b) / var(r_b): 1 when the surface stands at a
constant height above the mesh (it follows the relief), 0 when it is smooth over it (t_b = -r_b); and the rms of t_b
(pitches). In the region, the contrast and on the not-thin surface."""
import json
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.pipeline.window.layer import layer_by_asymmetry
from physmorph.render.exterior import ZhuBridson

torch.set_grad_enabled(False)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
SIG = (0.5, 1.0, 2.0)


def spacing8(x):
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    return float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (20000 / len(x)) ** (1 / 3)


class Graph:
    """The regions' operator (d159_regions.py): 16 nearest, same side, Gaussian of one sample spacing."""

    def __init__(self, P, N, a, k=16):
        d, nb = cKDTree(P).query(P, k + 1)
        side = ((N[nb] * N[:, None]).sum(-1) > 0.5)
        s = float(np.median(d[:, 1]))
        w = np.exp(-0.5 * (d / s) ** 2) * side
        w[:, 0] = 1.0
        self.w, self.nb = w / w.sum(1, keepdims=True), nb
        var = float(np.median((self.w * d ** 2).sum(1))) / 2.0
        self.steps = [max(1, int(round((s_ * a) ** 2 / var))) for s_ in SIG]

    def split(self, v):
        out, g, done = [], v, 0
        for k in self.steps:
            for _ in range(k - done):
                g = (self.w * g[self.nb]).sum(1)
            done = k
            out.append(g)
        return [v - out[0], out[0] - out[1], out[1] - out[2]]


def frame_map(mesh, seed):
    m97, ms = np.load(f"{G}/d154/meshes/{mesh}_97.npz"), np.load(f"{G}/d154/meshes/{mesh}_{seed}.npz")
    return lambda x: ((x / float(ms["scale"]) + ms["offset"]) - m97["offset"]) * float(m97["scale"])


def main():
    out, mesh, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    R = np.load(f"{G}/d159/step0/regions_{mesh}_b5.4.npz")
    a = float(R["pitch"])
    P, N = R["P"].astype(np.float64), R["N"].astype(np.float64)
    rb = [R["bands"][:, k].astype(np.float64) for k in range(3)]
    g = Graph(P, N, a)
    labels = {"region": R["region"], "contrast": R["contrast"], "not_thin": ~R["thin"]}
    for spec in specs:
        label, rest = spec.split("=", 1)
        seed = 97
        if ":" in rest:
            rest, seed = rest.rsplit(":", 1)
            seed = int(seed)
        path, frame = (rest.split("@") + ["-1"])[:2]
        z = np.load(path, allow_pickle=True)
        x = np.asarray(z["tgt"] if frame == "tgt" else z["frames"][int(frame)], np.float64)
        if seed != 97:
            x = frame_map(mesh, seed)(x)
        xt = torch.as_tensor(x)
        f, gr, s = ZhuBridson(xt, a)(torch.as_tensor(P))
        gn = gr.norm(dim=1).clamp_min(1e-6)
        t_ext = (-f / gn).numpy()
        ok_ext = np.isfinite(t_ext) & (np.abs(t_ext) < 3 * a)
        lm, _ = layer_by_asymmetry(xt, spacing8(x))
        L = x[lm.numpy()]
        dd, ii = cKDTree(L).query(P, 8)
        w = np.exp(-(dd / a) ** 2)
        t_lay = (w * ((L[ii] - P[:, None]) * N[:, None]).sum(-1)).sum(1) / np.maximum(w.sum(1), 1e-30)
        ok_lay = dd[:, 0] < 2 * a
        row = dict(mesh=mesh, label=label, npz=path, frame=frame, seed=seed, pitch=a, layer=int(lm.sum()), steps=g.steps,
                   sets={})
        for sname, t, ok in (("exterior", np.where(ok_ext, t_ext, 0.0), ok_ext), ("layer", t_lay, ok_lay)):
            tb = g.split(t)
            res = {"mean_height_a": float(t[ok].mean() / a), "valid": float(ok.mean())}
            for rname, sel0 in labels.items():
                sel = sel0 & ok
                per = []
                for k in range(3):
                    x1, x2 = tb[k][sel], rb[k][sel]
                    vr = float(((x2 - x2.mean()) ** 2).mean())
                    cv = float(((x1 - x1.mean()) * (x2 - x2.mean())).mean())
                    per.append(dict(k=1.0 + cv / max(vr, 1e-30), t_rms_a=float(np.sqrt((x1 ** 2).mean()) / a),
                                    r_rms_a=float(np.sqrt((x2 ** 2).mean()) / a)))
                ta, ra = (tb[0] + tb[1] + tb[2])[sel], (rb[0] + rb[1] + rb[2])[sel]
                vr = float(((ra - ra.mean()) ** 2).mean())
                cv = float(((ta - ta.mean()) * (ra - ra.mean())).mean())
                res[rname] = dict(n=int(sel.sum()), bands=per, k_below_10_8=1.0 + cv / max(vr, 1e-30))
            row["sets"][sname] = res
        with open(out, "a") as fo:
            fo.write(json.dumps(row) + "\n")
        print(label, {k: {r: (round(v[r]["k_below_10_8"], 3), [round(b["k"], 3) for b in v[r]["bands"]]) for r in ("region", "contrast")}
                      for k, v in row["sets"].items()}, flush=True)


if __name__ == "__main__":
    main()
