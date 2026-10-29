"""pull_meas.py OUT_MD DUMP_DIR KF.npz -- D175 measurement (a) of tmp/d175/MEAS_a.md: where each keyframe comparison
(current, blur sigma, pyramid, sdf) puts the render's pull, (1) in position space on S97's dumped segment states and
(2) through the dynamics at the source state (the method's segment gradient with that comparison, Adam's first step,
one rollout), against a 3D-target reference. On a card (the exterior operators and the rollouts are CUDA)."""
import glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d175")
import physmorph  # noqa: E402,F401  (before torch)
import torch  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

from d175_read import Ctx  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.losses.silhouette import soft_silhouette_multi  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_exterior  # noqa: E402
from physmorph.pipeline.run.state import fragment_mask  # noqa: E402
from physmorph.pipeline.run.storyboard import Storyboard  # noqa: E402
from physmorph.pipeline.target import _gauss_blur, build_target, calibrate_units, keyframe_pictures  # noqa: E402
from physmorph.pipeline.window import StartState  # noqa: E402
from physmorph.pipeline.window.rollout import graph_terms  # noqa: E402
from physmorph.pipeline.window.solve import WindowOptimizer, _norm, pcgrad  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402

LAYERS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)]
NAMES = ["current", "blur2", "blur4", "blur8", "pyramid", "sdf"]


def depth_pitch(x):
    h = ConvexHull(x)
    d = np.min(-(x @ h.equations[:, :3].T + h.equations[:, 3]), axis=1)
    a = 0.708 * float(np.median(cKDTree(x).query(x, 9)[0][:, 8]))
    return d / a


def aspect(x):
    v = ((x - x.mean(0)) ** 2).mean(0)
    return float(np.sqrt(v[1] / (0.5 * (v[0] + v[2]))))


def cos(a, b):
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def profile(v, x, dep, c, A):
    """Shares of |v|^2 per layer, cos with the stretch (all / per layer / below 3), linear fit share and cos(M, A - I)."""
    e = (v ** 2).sum(1)
    st = (x - c) @ (A - np.eye(3)).T
    sh = [float(e[(dep >= lo) & (dep < hi)].sum() / max(e.sum(), 1e-30)) for lo, hi in LAYERS]
    cl = [cos(v[(dep >= lo) & (dep < hi)], st[(dep >= lo) & (dep < hi)]) for lo, hi in LAYERS]
    deep = dep >= 3
    r = x - c
    M, *_ = np.linalg.lstsq(r, v, rcond=None)
    expl = 1.0 - float(((v - r @ M) ** 2).sum() / max(e.sum(), 1e-30))
    return dict(share=sh, cos_layer=cl, cos_all=cos(v, st), cos_deep=cos(v[deep], st[deep]),
                deep_share=float(e[deep].sum() / max(e.sum(), 1e-30)), lin=expl, cos_M=cos(M.T, A - np.eye(3)))


class Compare:
    """The comparisons on the exterior discs against the keyframe's soft pictures."""

    def __init__(self, ctx, sils_k, shade_k):
        self.ctx, self.cfg, self.f = ctx, ctx.cfg, ctx.final
        self.sils_k, self.shade_k = sils_k, shade_k
        self.Ak = torch.stack(list(sils_k))
        B = (self.Ak > 0.5).cpu().numpy()
        sdf = np.stack([distance_transform_edt(~b) - distance_transform_edt(b) for b in B]) / B.shape[-1]
        self.sdf = gpu.tensor(sdf.astype(np.float32))

    def discs(self, xt, ext):
        return tracked(make_field(ext.field, xt.detach(), ext.pitch, radius=ext.radius, offset=ext.offset,
                                  device_field=True), ext.lattice, ext.h, ext.skin)

    def value(self, name, p, n):
        cfg, f = self.cfg, self.f
        if name == "current":
            sil, pbr = d_exterior(p, n, self.sils_k, self.shade_k, f.views, cfg.render_res, f.extent, cfg.sil_k,
                                  cfg.w_hole, cfg.w_spray, cfg.pbr_ambient)
            return sil + cfg.w_pbr * pbr
        a = soft_silhouette_multi(p, f.views, cfg.render_res, f.extent, cfg.sil_k)
        if name.startswith("blur"):
            s = float(name[4:])
            return (_gauss_blur(a, s) - _gauss_blur(self.Ak, s)).pow(2).mean()
        if name == "pyramid":
            terms = [(a - self.Ak).pow(2).mean()] + [(_gauss_blur(a, s) - _gauss_blur(self.Ak, s)).pow(2).mean()
                                                     for s in (1.0, 2.0, 4.0, 8.0)]
            return sum(t / t.detach().clamp_min(1e-30) for t in terms)
        if name == "sdf":
            return (a * self.sdf).mean()
        raise ValueError(name)


def main():
    out, dump, kfp = sys.argv[1], sys.argv[2], sys.argv[3]
    ctx = Ctx(97, "assets/bunny.obj", kfp)
    cfg, f, prep = ctx.cfg, ctx.final, ctx.prep
    kf = np.load(kfp)
    A, c = np.asarray(kf["A"]), np.asarray(kf["centre"])
    smp = list(np.asarray(kf["samples"], np.float32))
    sils_k, shade_k = keyframe_pictures(f, smp, cfg)
    C = Compare(ctx, sils_k, shade_k)
    ext_k = replace(f.ext, sils=sils_k, shade=shade_k)
    L = ["# D175 (a): where each keyframe comparison puts the render's pull (criteria: tmp/d175/MEAS_a.md)", "",
         "Layers = pitches below the body's convex hull: 0-1 / 1-2 / 2-3 / 3-5 / > 5. Stretch = the keyframe's (A - I)(x - c).",
         "", "## 1. Position pull on S97's dumped segment states", "",
         "| window | comparison | share of pull^2 per layer | below 3 | cos(-pull, stretch) all / below 3 | linear share, cos(M, A-I) | cos with the dump's g_rend |",
         "|---|---|---|---|---|---|---|"]
    for fz in sorted(glob.glob(f"{dump}/terms_*.npz")):
        z = np.load(fz)
        x = np.asarray(z["x"], np.float64)
        dep = depth_pitch(x)
        gd = np.asarray(z["g_rend"], np.float64)
        for name in NAMES:
            xt = gpu.tensor(x.astype(np.float32)).requires_grad_(True)
            p, n, _ = C.discs(xt, ext_k).read(xt)
            g = torch.autograd.grad(C.value(name, p, n), xt)[0].detach().cpu().numpy().astype(np.float64)
            r = profile(-g, x, dep, c, A)
            L.append(f"| {fz[-8:-4]} | {name} | " + " / ".join(f"{s:.3f}" for s in r["share"]) + f" | {r['deep_share']:.3f} | "
                     f"{r['cos_all']:+.3f} / {r['cos_deep']:+.3f} | {r['lin']:.3f}, {r['cos_M']:+.3f} | {cos(g, gd):+.3f} |")
    # 2. through the dynamics at the source state
    src = gpu.tensor(np.asarray(prep.src, np.float32))
    prm = prep.prm
    calibrate_units(f, src, cfg)
    N = src.shape[0]
    Fp = torch.eye(3, device=gpu.DEVICE).repeat(N, 1, 1)
    vol0 = compute_rest_volumes(src, 1.0, prm, cfg.device)
    coh = gpu.knn(src, cfg.coh_k + 1)[1][:, 1:]
    bonds = (coh, (src[coh] - src[:, None, :]).norm(dim=2), fragment_mask(src, prm).float())
    bal = LambdaBalancer(cfg.lambda_auto, 0.0, None, cap_rel=20.0)
    Storyboard.calibrate(StartState(x=src.clone(), Fp=Fp), prm, cfg, f, bal, vol0, bonds, print)
    lam = float(bal.lam)
    f.ext, f.geometry, f.keyframe, f.relief = ext_k, False, 0, None
    rows = []

    def one_step(opt, render):
        win = opt.win
        e = graph_terms(win, opt.obj, opt.leaves[0], opt.u)
        gp = torch.autograd.grad(opt.obj.phys_core(e), opt.leaves, retain_graph=render)
        if render:
            gr = [r.detach().clone() for r in torch.autograd.grad(e.lr, opt.leaves)]
            g = [a + lam * b for a, b in zip(gp, pcgrad(gp, gr))]
        else:
            g = [a.detach() for a in gp]
        sc = max(cfg.min_alpha_scale, min(1.0, win.target_norm_eff / max(_norm(g), 1e-30)))
        with torch.no_grad():
            for p_, gi in zip(opt.leaves, g):
                p_.copy_(-cfg.alpha * sc * gi / (gi.abs() + win.eps_eff))
            nn_ = opt.dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
            opt.dFc *= (cfg.dfc_clip / nn_.clamp_min(1e-8)).clamp(max=1.0)
            opt.u.clamp_(-win.sp0, win.sp0)
        opt.free_of_rigid()
        x1 = opt.eval().xT.detach().cpu().numpy().astype(np.float64)
        with torch.no_grad():
            for p_ in opt.leaves:
                p_.zero_()
        return x1

    opt = WindowOptimizer(StartState(x=src.clone(), Fp=Fp), prm, cfg, f, bal, vol0, bonds, 1.0, None, lambda *_: None)
    x0 = opt.eval().xT.detach().cpu().numpy().astype(np.float64)
    dep0 = depth_pitch(x0)
    own = opt.obj.render_terms
    for name in NAMES:
        if name == "current":
            opt.obj.render_terms = own
        else:
            def rt(xT, name=name):
                p, n, _ = C.discs(xT, ext_k).read(xT)
                return C.value(name, p, n), xT.sum() * 0.0
            opt.obj.render_terms = rt
        x1 = one_step(opt, True)
        rows.append((name, profile(x1 - x0, x0, dep0, c, A), aspect(x1) - aspect(x0)))
    opt.obj.render_terms = own
    del opt
    # the reference: the keyframe sample as a 3D target (transport + proximity), physics alone
    kp = build_target(smp[0], prm, cfg)
    calibrate_units(kp, src, cfg)
    opt3 = WindowOptimizer(StartState(x=src.clone(), Fp=Fp), prm, cfg, kp, LambdaBalancer(cfg.lambda_auto, 0.0, None),
                           vol0, bonds, 1.0, None, lambda *_: None)
    x0b = opt3.eval().xT.detach().cpu().numpy().astype(np.float64)
    x1 = one_step(opt3, False)
    rows.append(("3D target (reference)", profile(x1 - x0b, x0b, depth_pitch(x0b), c, A), aspect(x1) - aspect(x0b)))
    cur = next(r for n_, r, _ in rows if n_ == "current")
    L += ["", f"## 2. Through the dynamics at the source (Adam's first step, lambda {lam:.4g} held, one rollout)", "",
          "| comparison | share of dx^2 per layer | below 3 | cos(dx, stretch) all / below 3 | linear share, cos(M, A-I) | "
          "aspect gain x1e3 | REACH | ALIGN |", "|---|---|---|---|---|---|---|---|"]
    passing = []
    for name, r, da in rows:
        reach = r["deep_share"] >= max(0.25, 2 * cur["deep_share"])
        align = r["cos_deep"] >= 0.3 and r["cos_deep"] >= cur["cos_deep"] + 0.1
        if name in NAMES and name != "current" and reach and align:
            passing.append((da, name))
        L.append(f"| {name} | " + " / ".join(f"{s:.3f}" for s in r["share"]) + f" | {r['deep_share']:.3f} | "
                 f"{r['cos_all']:+.3f} / {r['cos_deep']:+.3f} | {r['lin']:.3f}, {r['cos_M']:+.3f} | {da * 1e3:+.2f} | "
                 f"{'yes' if reach else 'no'} | {'yes' if align else 'no'} |")
    L += ["", "VERDICT (a): " + (f"{max(passing)[1]} reaches the bulk (largest one-step aspect gain among "
                                 f"{[n_ for _, n_ in passing]}): pre-register its arm" if passing
                                 else "no blurred comparison reaches the bulk (REACH and ALIGN): no arm")]
    Path(out).write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
