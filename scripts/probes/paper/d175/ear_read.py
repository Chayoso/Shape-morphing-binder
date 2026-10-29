"""ear_read.py OUT_MD RUNDIR -- D175 forensic of W97's ear-tip piece (the P3 miss of D175d), frame by frame, no definition
change. On a card (the term gradients read the final target's pack and the exterior).

1. The set: the particles of every non-largest set of >= 10 particles (1.5-pitch link, the reader's) in W97's kept frames;
   their history from frame 0: each kept frame's component containing them (its size, whether it is the body), their gap to
   the body in pitches, their height; their depth below the source's hull at frame 0 (surface particles or not).
2. The same particle ids (the seed's source sample is shared) in D97, T97 and X97: the largest separate set they form and
   its gap, per frame.
3. Which term pulls them off: at W97's kept frames 552-876 (every 24), each objective term's position gradient at the
   state (window/objective.active_set_record's: scaled transport, surface proximity, near band, spray cleanup, the weighted
   render), read on the set as the mean component along the outward direction (from the body's nearest particle to the
   set's particle; positive = away from the body), against the same on the body particles within 3 pitches of the set."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d175")
import physmorph  # noqa: E402,F401  (before torch)
import torch  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

from d175_read import Ctx  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer  # noqa: E402
from physmorph.pipeline.run.state import fragment_mask  # noqa: E402
from physmorph.pipeline.run.storyboard import Storyboard  # noqa: E402
from physmorph.pipeline.target import calibrate_units  # noqa: E402
from physmorph.pipeline.window import StartState  # noqa: E402
from physmorph.pipeline.window.solve import WindowOptimizer  # noqa: E402


def labels(x, a):
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    return connected_components(g, directed=False)[1]


def main():
    out, rd = sys.argv[1], sys.argv[2]
    zw = np.load(f"{rd}/W_97_frames12.npz")
    fr, raws, tgt = zw["frames"], np.asarray(zw["raws"]), np.asarray(zw["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
    S = set()
    for x in fr:
        lab = labels(np.asarray(x, np.float64), a)
        sz = np.bincount(lab)
        big = np.argmax(sz)
        for c in np.flatnonzero(sz >= 10):
            if c != big:
                S |= set(np.flatnonzero(lab == c).tolist())
    S = np.array(sorted(S))
    x0 = np.asarray(fr[0], np.float64)
    h = ConvexHull(x0)
    dep0 = np.min(-(x0 @ h.equations[:, :3].T + h.equations[:, 3]), axis=1) / a
    L = [f"# D175 forensic: W97's ear-tip piece (pitch {a:.4f} wu)", "",
         f"The set: {len(S)} particles {S.tolist()}; depth below the source's hull at frame 0: "
         f"{np.round(dep0[S], 2).tolist()} pitches (the body's median {np.median(dep0):.2f}).", "",
         "## 1-2. History: per run, the largest separate set of these ids (size, gap to the body in pitches) and their mean "
         "height above the body's centre", ""]
    rows = {}
    for name in ("W_97", "D_97", "T_97", "X_97"):
        try:
            z = np.load(f"{rd}/{name}_frames12.npz")
        except OSError:
            continue
        rr, F = np.asarray(z["raws"]), z["frames"]
        line = []
        for i in range(0, len(rr)):
            if rr[i] % 48 and not (552 <= rr[i] <= 876):
                continue
            x = np.asarray(F[i], np.float64)
            lab = labels(x, a)
            sz = np.bincount(lab)
            big = np.argmax(sz)
            cs = [c for c in set(lab[S].tolist()) if c != big]
            best = max(cs, key=lambda c: sz[c]) if cs else None
            gap = (cKDTree(x[lab == big]).query(x[lab == best])[0].min() / a) if best is not None else 0.0
            line.append((int(rr[i]), int(sz[best]) if best is not None else 0, round(float(gap), 2),
                         round(float(x[S, 1].mean() - x[:, 1].mean()), 3), int(sum(sz[c] for c in cs))))
        rows[name] = line
        L.append(f"- {name}: " + "; ".join(f"{r}: {s} ({g})" for r, s, g, _, _ in line if s > 0 or r % 240 == 0))
        L.append(f"  - largest separate set of these ids over the run: {max((s for _, s, _, _, _ in line), default=0)}; "
                 f"frames with one >= 10: {[r for r, s, _, _, _ in line if s >= 10]}")
    # 3. the terms at W97's frames
    ctx = Ctx(97, "assets/bunny.obj", f"{Path(rd).parent}/assets/kf_s97.npz")
    cfg, f, prep = ctx.cfg, ctx.final, ctx.prep
    src = gpu.tensor(np.asarray(prep.src, np.float32))
    calibrate_units(f, src, cfg)
    N = src.shape[0]
    Fp = torch.eye(3, device=gpu.DEVICE).repeat(N, 1, 1)
    vol0 = compute_rest_volumes(src, 1.0, prep.prm, cfg.device)
    coh = gpu.knn(src, cfg.coh_k + 1)[1][:, 1:]
    bal = LambdaBalancer(cfg.lambda_auto, 0.0, None, cap_rel=20.0)
    Storyboard.calibrate(StartState(x=src.clone(), Fp=Fp), prep.prm, cfg, f,
                         bal, vol0, (coh, (src[coh] - src[:, None, :]).norm(dim=2), fragment_mask(src, prep.prm).float()))
    lam = float(bal.lam)
    cfg2 = dataclasses.replace(cfg, term_dump="ear")
    L += ["", f"## 3. The terms' outward pull on the set (positive = away from the body; lambda {lam:.4g} held), mean over the "
          "set / over the body particles within 3 pitches of it (for those, positive = toward the set)", "",
          "| frame | set separate (size) | transport | proximity | near band | spray | render |", "|---|---|---|---|---|---|---|"]
    for i in range(len(raws)):
        if not (552 <= raws[i] <= 876) or raws[i] % 24:
            continue
        x = gpu.tensor(np.asarray(fr[i], np.float32))
        xs = x.cpu().numpy().astype(np.float64)
        lab = labels(xs, a)
        big = np.argmax(np.bincount(lab))
        body = np.flatnonzero(lab == big)
        bt = cKDTree(xs[body])
        d, j = bt.query(xs[S])
        outw = (xs[S] - xs[body[j]]) / np.maximum(d[:, None], 1e-12)
        near = np.setdiff1d(np.unique(np.concatenate(cKDTree(xs).query_ball_point(xs[S], 3 * a))), S)
        dn, jn = cKDTree(xs[S]).query(xs[near])
        outn = (xs[S][jn] - xs[near]) / np.maximum(dn[:, None], 1e-12)     # toward the set
        b2 = (coh, (x[coh] - x[:, None, :]).norm(dim=2), fragment_mask(x, prep.prm).float())
        opt = WindowOptimizer(StartState(x=x.clone(), Fp=Fp), prep.prm, cfg2, f, bal, vol0, b2, 1.0, None, lambda *_: None)
        tg = opt.obj.active_set_record(x, lam)["term_grads"]
        cells = []
        for k in ("ot", "surf", "near", "spray", "rend"):
            g = -tg[k].detach().cpu().numpy().astype(np.float64)                   # the descent direction
            cells.append(f"{(g[S] * outw).sum(1).mean():+.2e} / {(g[near] * outn).sum(1).mean():+.2e}")
        sep = [c for c in set(lab[S].tolist()) if c != big]
        L.append(f"| {int(raws[i])} | {len(sep) > 0} ({max((int((lab == c).sum()) for c in sep), default=0)}) | "
                 + " | ".join(cells) + " |")
        del opt
        torch.cuda.empty_cache()
    Path(out).write_text("\n".join(L) + "\n")
    json.dump(rows, open(str(out).replace(".md", ".json"), "w"))
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
