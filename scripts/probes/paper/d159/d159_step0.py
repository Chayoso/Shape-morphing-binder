"""d159_step0.py OUT_JSONL MESH [MESH ...] -- D159 step 0 (a), no run (CPU): what the new pictures ask of the tag's end
states. Per mesh (B1, seed 97, 300k): the run's own target exterior discs (the target sample and its seven draws, seeds
seed + 1 + 100 k, Zhu and Bridson's field at the run's pitch, the lattice of target.py: half a 188-px pixel, refine off);
c = their mean signed distance from the mesh (render/mesh_shell.py: the mesh in the run's frame, closed and turned
outward); the old pictures (their mean) and the new ones (the shell's discs) by the loss's operators; the end states of
the tag's L97 and P97 drawn on the same lattice, their render terms (silhouette, shading; w_hole 2, w_spray 1) against
both; and the offset of each state's exterior (and of the own target sample's) from the shell, sd - c, split by the same
Gaussian means over the discs' own same-side graph into the part above 10.8 pitches (the shell the pictures ask for
beyond the relief) and below (the relief), rms in pitches."""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as nnf
from scipy.spatial import cKDTree

from physmorph.losses.silhouette import set_kernel
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views
from physmorph.render.exterior import Lattice, ZhuBridson
from physmorph.render.mesh_shell import MeshDistance, closed_outward, mesh_shell_discs
from physmorph.sampling.mesh import draws_in_frame, load_normalized

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
set_kernel("cic")
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
REPO = os.environ.get("D158_REPO", "/data/relcfd/chayo/physmorph_v2/repo_d159")
VIEWS = make_views(6, (0.0, 0.5, -0.5))
RES = 188


def pitch_of(x):
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    d = cKDTree(sub).query(sub, 9)[0][:, 8]
    return .708 * float(np.median(d)) * (20000 / len(x)) ** (1 / 3)


def bands(p, n, v, a):
    """(rms of v, of its part above 10.8 pitches, of its part below) over the discs' same-side 16-neighbour graph."""
    d, nb = cKDTree(p).query(p, 17)
    side = (n[nb] * n[:, None]).sum(-1) > 0.5
    s = float(np.median(d[:, 1]))
    w = np.exp(-0.5 * (d / s) ** 2) * side
    w[:, 0] = 1.0
    w /= w.sum(1, keepdims=True)
    var = float(np.median((w * d ** 2).sum(1))) / 2.0
    g = v.copy()
    for _ in range(max(1, int(round((2.0 * a) ** 2 / var)))):
        g = (w * g[nb]).sum(1)
    rms = lambda t: float(np.sqrt((t ** 2).mean()) / a)          # noqa: E731
    return rms(v - v.mean()), rms(g - v.mean()), rms(v - g)


def main():
    out, meshes = sys.argv[1], sys.argv[2:]
    os.chdir(REPO)
    for mesh in meshes:
        t0 = time.time()
        zl = np.load(f"{G}/d145t_b1/{mesh}300k_L97_end1.npz", allow_pickle=True)
        zp = np.load(f"{G}/d145t_b1/{mesh}300k_P97_end1.npz", allow_pickle=True)
        tgt = np.asarray(zl["tgt"], np.float32)
        n = len(tgt)
        _, v_src = load_normalized("assets/isosphere.obj", n, 97, return_volume=True, sample="stratified", fill={})
        frame = {}
        tg, _ = load_normalized(f"assets/{mesh}.obj", n, 98, match_volume=v_src, sample="stratified", return_volume=True,
                                frame=frame, fill={})
        same = float(np.abs(tg - tgt).max())
        samples = [tgt] + [np.asarray(d, np.float32) for d in draws_in_frame(f"assets/{mesh}.obj", frame, n,
                                                                              [98 + 100 * k for k in range(1, 8)])]
        a = pitch_of(tgt)
        extent = float(np.abs(tgt).max()) * 1.25
        tt = torch.as_tensor(tgt)
        center = tt.mean(0)
        lattice = Lattice(center, 2.8 * float((tt - center).norm(dim=1).max()))
        h = min(extent / RES, 0.92 * a)

        def discs(x):
            p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(x), a), h, refine=False)
            return p, nnf.normalize(g, dim=1)

        refs, per = [], []
        for s in samples:
            p, nn_ = discs(s)
            refs.append(p)
            per.append(exterior_targets(p, nn_, VIEWS, RES, extent, 1.5, 0.25))
        old = ([torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])],
               [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])])
        m = np.load(f"{G}/d154/meshes/{mesh}_97.npz")
        ps, ns, c, info = mesh_shell_discs(lattice, h, m["V"], m["F"], a, refs, 3.0 * a)
        new = exterior_targets(ps, ns, VIEWS, RES, extent, 1.5, 0.25)
        msq = lambda A, B: float(torch.stack([(x - y).square().mean() for x, y in zip(A, B)]).mean())   # noqa: E731
        Vc, Fc, _, _ = closed_outward(m["V"], m["F"])
        dist = MeshDistance(Vc, Fc)
        row = dict(mesh=mesh, tgt_reproduced_maxdiff=same, pitch=a, extent=extent, res=RES, lattice_a=h / a, c_a=c / a,
                   c_sd_a=info["c_sd"] / a, c_p01_a=info["c_p01"] / a, c_p50_a=info["c_p50"] / a, c_p99_a=info["c_p99"] / a,
                   refs_sd_mean_a=info["refs_sd_mean"] / a, refs_sd_median_a=info["refs_sd_median"] / a,
                   shell_discs=info["discs"], sample_discs=[len(r) for r in refs],
                   watertight=info["watertight"], loops=info["loops_closed"],
                   pictures_msq=dict(sil=msq(new[0], old[0]), pbr=msq(new[1], old[1])), states={})
        for lab, x in (("own", tgt), ("L97", np.asarray(zl["frames"][-1], np.float32)),
                       ("P97", np.asarray(zp["frames"][-1], np.float32))):
            p, nn_ = discs(x)
            t_old = [float(t) for t in d_exterior(p, nn_, old[0], old[1], VIEWS, RES, extent, 1.5)]
            t_new = [float(t) for t in d_exterior(p, nn_, new[0], new[1], VIEWS, RES, extent, 1.5)]
            off = dist.signed(p.double().numpy()) - c
            tot, low, rel = bands(p.double().numpy(), nn_.double().numpy(), off, a)
            row["states"][lab] = dict(discs=len(p), terms_old=t_old, terms_new=t_new, offset_mean_a=float(off.mean() / a),
                                      offset_rms_a=tot, low_rms_a=low, relief_rms_a=rel)
        row["seconds"] = time.time() - t0
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
