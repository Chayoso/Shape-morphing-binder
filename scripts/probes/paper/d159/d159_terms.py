"""d159_terms.py OUT_JSONL MESH NAME TERMS_DIR [SEED] -- D159 R5 (CPU; measurement only; amendment 3): which term holds the
detail band, against the render's push. TERMS_DIR holds the run's --term_dump files (terms_NNNN.npz at every committed
window: the state x and each term's per-particle position gradient: g_ot transport, g_surf surface proximity, g_near
near band, g_spray spray cleanup, g_rend the lambda-weighted render). Per window, on the state's outer layer (the
pipeline's layer and normals): each gradient's normal component, split on the layer's same-side graph at sigma = 2
pitches into the detail band (below 10.8 pitches) and the rest; per term the rms of its detail part (whole layer and the
region fixed before the runs), the rms of the rest, and the cosine of its detail part with the render's (< 0: it pushes
the detail band against the render); the physics' sum (transport + proximity + near + spray) likewise. At the last
window also the relaxation (a position edit, not a term), by its own operator at that state: the window's layer graph
(layer_relax_data: 24 same-side layer neighbours, Gaussian of 2 spacings) and the relief reference (window/layer.TargetRelief:
the residual d - dbar read on the layer's feet on 75 000 points of the mesh's surface, as prepare.py draws them):
r = (d - dbar) - ref, which the relaxation removes a twentieth of per step; its detail-band rms (pitches) and the
cosine of its removal (-r) with the render's detail push."""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_relax_data

torch.set_grad_enabled(False)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
TERMS = ("ot", "surf", "near", "spray", "rend")


def spacing8(x):
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    return float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (20000 / len(x)) ** (1 / 3)


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


def rms(v, m=None):
    v = v if m is None else v[m]
    return float(np.sqrt((v ** 2).mean())) if len(v) else None


def cos(u, v, m=None):
    if m is not None:
        u, v = u[m], v[m]
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float((u * v).sum() / (nu * nv)) if nu > 0 and nv > 0 else None


def relief_surface(mesh, seed, n):
    """The relaxation's reference points, as prepare.py draws them (surface_in_frame(path, frame, n // 4), seed 0)."""
    from physmorph.sampling.mesh import load_normalized, surface_in_frame
    cwd = os.getcwd()
    os.chdir(os.environ.get("D159_REPO", "/data/relcfd/chayo/physmorph_v2/repo_d159"))
    try:
        _, v_src = load_normalized("assets/isosphere.obj", n, seed, return_volume=True, sample="stratified", fill={})
        frame = {}
        load_normalized(f"assets/{mesh}.obj", n, seed + 1, match_volume=v_src, sample="stratified", return_volume=True,
                        frame=frame, fill={})
        return surface_in_frame(f"assets/{mesh}.obj", frame, n // 4)
    finally:
        os.chdir(cwd)


def main():
    out, mesh, name, tdir = sys.argv[1:5]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 97
    R = np.load(f"{G}/d159/step0/regions_{mesh}_b5.4.npz")
    a = float(R["pitch"])
    rtree, rlab = cKDTree(R["P"]), R["region"]
    files = sorted(glob.glob(f"{tdir}/terms_*.npz"))
    for i, f in enumerate(files):
        z = np.load(f)
        x = np.asarray(z["x"], np.float64)
        xt = torch.as_tensor(x)
        sp = spacing8(x)
        lm, ln = layer_by_asymmetry(xt, sp)
        m = lm.numpy()
        p, n = x[m], ln[lm].numpy().astype(np.float64)
        lp = low_pass(p, n, a)
        reg = rlab[rtree.query(p)[1]]
        parts = {}
        for t in TERMS:
            g = np.asarray(z[f"g_{t}"], np.float64)[m]
            gn = (g * n).sum(1)
            low = lp(gn)
            parts[t] = (gn - low, low)
        phys_d = sum(parts[t][0] for t in TERMS if t != "rend")
        phys_l = sum(parts[t][1] for t in TERMS if t != "rend")
        parts["physics"] = (phys_d, phys_l)
        rd = parts["rend"][0]
        row = dict(mesh=mesh, name=name, window=int(os.path.basename(f)[6:10]), last=i == len(files) - 1, layer=int(m.sum()),
                   terms={t: dict(detail_rms=rms(d), detail_rms_region=rms(d, reg), rest_rms=rms(l),
                                  cos_with_render=(None if t == "rend" else cos(d, rd)),
                                  cos_with_render_region=(None if t == "rend" else cos(d, rd, reg)))
                          for t, (d, l) in parts.items()})
        if i == len(files) - 1:
            surf_p, surf_n = relief_surface(mesh, seed, len(x))
            surf_p, surf_n = surf_p.astype(np.float64), surf_n.astype(np.float64)
            inside = x[cKDTree(x).query(surf_p, 8)[1]].mean(1)
            if float(np.sign(((surf_p - inside) * surf_n).sum(1)).mean()) < 0:
                surf_n = -surf_n
            mask, nrm, nbr, w = layer_relax_data(xt, sp)
            mask, nrm, nbr, w = mask.numpy() > 0.5, nrm.numpy().astype(np.float64), nbr.numpy(), w.numpy().astype(np.float64)
            on = np.nonzero(mask)[0]
            dd, ii = cKDTree(surf_p).query(x[on], 1)
            q, mm = surf_p[ii], surf_n[ii]
            foot = x.copy()
            foot[on] = x[on] - ((x[on] - q) * mm).sum(1, keepdims=True) * mm
            res = (nrm * (foot - (w[..., None] * foot[nbr]).sum(1))).sum(1)
            near = np.zeros(len(x), bool)
            near[on] = dd < sp
            ref = np.where(near, res - (w * res[nbr]).sum(1), 0.0)
            d = (nrm * (x - (w[..., None] * x[nbr]).sum(1))).sum(1)
            r = (d - (w * d[nbr]).sum(1)) - ref                     # what the relaxation removes, a twentieth a step
            rl = r[on]
            pl, nl = x[on], nrm[on]
            lp2 = low_pass(pl, nl, a)
            rdet = rl - lp2(rl)
            # the render's detail push on the same particles (the relaxation's layer may differ from the layer above)
            g = np.asarray(z["g_rend"], np.float64)[on]
            gn = (g * nl).sum(1)
            gdet = gn - lp2(gn)
            reg2 = rlab[rtree.query(pl)[1]]
            row["relaxation"] = dict(layer=int(len(on)), residual_detail_rms_a=rms(rdet) / a, residual_detail_rms_region_a=rms(rdet, reg2) / a,
                                     residual_rest_rms_a=rms(lp2(rl)) / a,
                                     cos_removal_with_render=cos(-rdet, -gdet), cos_removal_with_render_region=cos(-rdet, -gdet, reg2),
                                     ref_rms_a=rms(ref[on]) / a)
        with open(out, "a") as fo:
            fo.write(json.dumps(row) + "\n")
        if i % 10 == 0 or i == len(files) - 1:
            t = row["terms"]
            print(row["window"], {k: (f"{v['detail_rms']:.2e}", None if v["cos_with_render"] is None else round(v["cos_with_render"], 2))
                                  for k, v in t.items()}, row.get("relaxation", ""), flush=True)


if __name__ == "__main__":
    main()
