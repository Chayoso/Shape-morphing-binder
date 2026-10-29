"""seg_read.py OUT_DIR KF.npz -- PIPELINE ARGS -- D175: the one-step reading at the start of every keyframe-segment
window of a live S97 rerun. Measurement only: the run is pipeline_run.py with its own flags; window/optimize_window is
wrapped so that, before each segment window is optimised, its full start state is saved
(OUT_DIR/state_w<k>.npz: x, F, v, C, Fp, J, the bonds' rest lengths and fragment mask, the warm start, the step scale)
and read, then the window runs exactly as it would.

The reading at a window start (zero control, as the window's first gradient):
- the method's segment step with a given render term: g = g_p + lambda PCGrad(g_p, g_r), with g_r rescaled to the full
  render term's norm so every term has the same authority; Adam's first step g / (|g| + eps), the run's length and clip;
  one rollout. Delta x = x_T(step) - x_T(0): its share below 3 pitches, cos with the keyframe's stretch (A - I)(x - c),
  and the one-step aspect gain.
- The render terms on the exterior, an exact split of the run's term: hole_core = w_hole (a_k - a)+^2 off the keyframe's
  edge band, area_core = w_spray (a - a_k)+^2 off it, edge = both on the band (0.05 <= a_k <= 0.95), shading.
  Two candidate pictures as well: cloud_sil (the silhouette of the particles themselves, render_loss.d_render's operator,
  against the keyframe samples' cloud silhouettes) and column (the projected particle count, the X-ray picture, against
  the keyframe samples').
- Per term: |g_term| / |g_render| and cos(g_term, g_render) on dF_c (raw render gradients).
- Pictures at the window start: the pole pixels (keyframe a_k > 0.5 where the source sphere has a < 0.5), the share of them
  covered (a > 0.5), the body's projected particle count there over the keyframe's, each term's value."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d175")
import physmorph  # noqa: E402,F401  (before torch)
import torch  # noqa: E402

import physmorph.pipeline.run.runner as RN  # noqa: E402
import scripts.pipeline_run as PR  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.losses.silhouette import _view_basis, soft_silhouette_multi, splat_terms  # noqa: E402
from physmorph.pipeline.render_loss import d_render, shaded_discs  # noqa: E402
from physmorph.pipeline.window.rollout import graph_terms  # noqa: E402
from physmorph.pipeline.window.solve import WindowOptimizer, _norm, pcgrad  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402
from pull_meas import aspect, depth_pitch, profile  # noqa: E402

TERMS = ["total", "hole_core", "area_core", "edge", "shading", "cloud_sil", "column"]
G = {"rows": [], "calls": 0, "src_alpha": None}


def cover(x, views, res, extent):
    """The CIC count of the points per pixel, (V, res, res): soft_silhouette_multi's coverage before its exp."""
    right, up = _view_basis(x, views)
    V = right.shape[0]
    p = torch.stack([x @ right.T, x @ up.T], -1)
    rel = (p + extent) / (2 * extent) * res
    voff = (torch.arange(V, device=x.device) * (res * res)).view(1, V)
    img = x.new_zeros(V * res * res)
    for ii, jj, w in splat_terms(rel):
        valid = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
        idx = (voff + ii * res + jj).clamp(0, V * res * res - 1)
        img = img.index_add(0, idx.reshape(-1), torch.where(valid, w, torch.zeros_like(w)).reshape(-1))
    return img.reshape(V, res, res)


def kf_pictures(tgt, cfg, smp):
    """The keyframe's cloud silhouettes and projected counts, the mean over its samples."""
    with torch.no_grad():
        ss = [gpu.tensor(s) for s in smp]
        a = torch.stack([soft_silhouette_multi(s, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k) for s in ss]).mean(0)
        c = torch.stack([cover(s, tgt.views, cfg.render_res, tgt.extent) for s in ss]).mean(0)
    return a, c


def term_fn(name, tgt, cfg, K):
    e = tgt.ext
    At = torch.stack(list(e.sils))
    St = torch.stack(list(e.shade))
    band = ((At >= 0.05) & (At <= 0.95)).float()

    def discs(xT):
        d = tracked(make_field(e.field, xT.detach(), e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                    e.lattice, e.h, e.skin)
        return d.read(xT)[:2]

    def f(xT):
        z = xT.sum() * 0.0
        if name == "cloud_sil":
            return d_render(xT, list(K["a_cloud"]), tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                            cfg.w_spray), z
        if name == "column":
            c = cover(xT, tgt.views, cfg.render_res, tgt.extent)
            return (c - K["cov"]).pow(2).mean() / K["cov"].pow(2).mean(), z
        p, n = discs(xT)
        if name == "shading":
            s, _ = shaded_discs(p, n, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.pbr_ambient)
            return z, (s - St).pow(2).mean(dim=(1, 2)).mean()
        a = soft_silhouette_multi(p, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k)
        hole = cfg.w_hole * torch.clamp(At - a, min=0.0).pow(2)
        area = cfg.w_spray * torch.clamp(a - At, min=0.0).pow(2)
        m = {"hole_core": hole * (1 - band), "area_core": area * (1 - band), "edge": (hole + area) * band}[name]
        return m.mean(dim=(1, 2)).mean(), z
    return f


def read(k, start, prm, cfg, tgt, balancer, vol0, bonds, K):
    opt = WindowOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, 1.0, None, lambda *_: None)
    win, lam = opt.win, float(opt.lam_r)
    own = opt.obj.render_terms
    x0t = opt.eval().xT.detach()
    x0 = x0t.cpu().numpy().astype(np.float64)
    dep0 = depth_pitch(x0)
    A, c = K["A"], K["c"]
    row = {"window": k, "lam": lam, "aspect_start": aspect(start.x.cpu().numpy().astype(np.float64)), "terms": {}}

    def step(render, ref_norm=None):
        e = graph_terms(win, opt.obj, opt.leaves[0], opt.u)
        gp = torch.autograd.grad(opt.obj.phys_core(e), opt.leaves, retain_graph=True)
        raw = [r.detach().clone() for r in torch.autograd.grad(e.lr, opt.leaves)]
        gr = list(raw)
        val = float(e.lr)
        if ref_norm is not None:
            s = ref_norm / max(_norm(gr), 1e-30)
            gr = [r * s for r in gr]
        g = [a_ + lam * b for a_, b in zip(gp, pcgrad(gp, gr))]
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
        return x1, raw, val

    ref = None
    for name in TERMS:
        opt.obj.render_terms = own if name == "total" else term_fn(name, tgt, cfg, K)
        x1, raw, val = step(True, None if name == "total" else ref["norm"])
        if name == "total":
            ref = {"g": raw[0], "norm": _norm(raw)}
        pr = profile(x1 - x0, x0, dep0, c, A)
        gd = raw[0]                                          # the term's raw dF_c gradient
        row["terms"][name] = dict(value=val, deep=pr["deep_share"], cos_deep=pr["cos_deep"], cos_all=pr["cos_all"],
                                  share=pr["share"], aspect_gain=aspect(x1) - aspect(x0),
                                  g_ratio=_norm([gd]) / max(_norm([ref["g"]]), 1e-30),
                                  g_cos=float((gd * ref["g"]).sum() / max(_norm([gd]) * _norm([ref["g"]]), 1e-30)))
    opt.obj.render_terms = own
    # the pictures at the window start
    e = tgt.ext
    with torch.no_grad():
        xs = start.x
        d = tracked(make_field(e.field, xs, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                    e.lattice, e.h, e.skin)
        p, _ = d.read(xs)[:2]
        a = soft_silhouette_multi(p, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k)
        At = torch.stack(list(e.sils))
        if G["src_alpha"] is None:
            G["src_alpha"] = a.clone()
        pole = (At > 0.5) & (G["src_alpha"] < 0.5)
        cb = cover(xs, tgt.views, cfg.render_res, tgt.extent)
        hole = cfg.w_hole * torch.clamp(At - a, min=0.0).pow(2)
        row.update(pole_px=int(pole.sum()), pole_covered=float((a[pole] > 0.5).float().mean()) if pole.any() else None,
                   pole_mass=float(cb[pole].sum() / K["cov"][pole].sum().clamp_min(1e-30)) if pole.any() else None,
                   pole_alpha=float(a[pole].mean()) if pole.any() else None,
                   hole_pole_share=float(hole[pole].sum() / hole.sum().clamp_min(1e-30)),
                   y_q50=float(np.quantile(np.abs(xs[:, 1].cpu().numpy() - c[1]), .5)),
                   y_q999=float(np.quantile(np.abs(xs[:, 1].cpu().numpy() - c[1]), .999)))
    del opt
    torch.cuda.empty_cache()
    return row


def main():
    out, kfp = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    argv = sys.argv[sys.argv.index("--") + 1:]
    kf = np.load(kfp)
    K = {"A": np.asarray(kf["A"]), "c": np.asarray(kf["centre"]), "smp": list(np.asarray(kf["samples"], np.float32))}
    orig = RN.optimize_window

    def wrapped(start, prm, cfg, tgt, balancer, vol0, bonds, dfc_init=None, alpha_scale=1.0, on_iter=None, log=print):
        G["calls"] += 1
        k = G["calls"]
        if getattr(tgt, "keyframe", None) is not None:
            if "cov" not in K:
                K["a_cloud"], K["cov"] = kf_pictures(tgt, cfg, K["smp"])
            h = lambda t: None if t is None else t.detach().cpu().numpy()  # noqa: E731
            np.savez(out / f"state_w{k}.npz", x=h(start.x), F=h(start.F), v=h(start.v), C=h(start.C), Fp=h(start.Fp),
                     J=h(start.J), bond_rest=h(bonds[1]), frag=h(bonds[2]), dfc_init=h(dfc_init), alpha_scale=alpha_scale,
                     lam=balancer.lam)
            try:
                G["rows"].append(read(k, start, prm, cfg, tgt, balancer, vol0, bonds, K))
            except Exception as ex:  # noqa: BLE001  (the reading must not cost the run)
                traceback.print_exc()
                G["rows"].append({"window": k, "error": repr(ex)})
            json.dump(G["rows"], open(out / "rows.json", "w"), indent=1, default=float)
        return orig(start, prm, cfg, tgt, balancer, vol0, bonds, dfc_init=dfc_init, alpha_scale=alpha_scale,
                    on_iter=on_iter, log=log)

    RN.optimize_window = wrapped
    sys.argv = ["pipeline_run.py", *argv]
    PR.main()
    write(out)


def write(out):
    rows = G["rows"]
    L = ["# D175: the one-step reading at the start of each keyframe-segment window (S97 rerun, 40k)", "",
         "Per term: share of |Delta x|^2 below 3 pitches / cos(Delta x, stretch) below 3 / aspect gain x1e3 of Adam's first "
         "step with that render term alone (its gradient at the full term's norm); g ratio and cos = the term's raw dF_c "
         "gradient against the full term's.", ""]
    for r in rows:
        if "error" in r:
            L.append(f"window {r['window']}: {r['error']}")
            continue
        L += [f"## Window {r['window']} (aspect at start {r['aspect_start']:.3f}; |y| q50 {r['y_q50']:.3f}, q99.9 "
              f"{r['y_q999']:.3f}; pole pixels {r['pole_px']}: covered {r['pole_covered']:.3f}, mean alpha "
              f"{r['pole_alpha']:.3f}, particle count over the keyframe's {r['pole_mass']:.3f}; the pole's share of the hole "
              f"term {r['hole_pole_share']:.3f})", "",
              "| term | value | below 3 | cos below 3 / all | aspect gain x1e3 | g ratio | g cos |", "|---|---|---|---|---|---|---|"]
        for n, t in r["terms"].items():
            L.append(f"| {n} | {t['value']:.4e} | {t['deep']:.3f} | {t['cos_deep']:+.3f} / {t['cos_all']:+.3f} | "
                     f"{t['aspect_gain'] * 1e3:+.2f} | {t['g_ratio']:.3f} | {t['g_cos']:+.3f} |")
        L.append("")
    (out / "seg_read.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
