"""thin_visibility.py LABEL=ARCHIVE.npz ... — what keeps thin features uncovered (T2, docs/experiments.md).

At the delivered end state, the outer target points of the thin bins (local thickness < 1 and 1-2 MPM cells) that
are farther than 1.5 target spacings from the body. For each, over the training views (make_views of the config,
extent 1.25 max|target|, the CIC soft silhouette with k = sil_k):
  deficit   clamp(a_target - a_body, 0) read through the point's own CIC footprint, at 64, 96 (training) and 256 px;
  exposed   every target point in its pixel lies within two MPM cells of it in depth (the pixel would be empty
            without the feature), at 64 and 256 px;
  front     it lies within two cells of the nearest or farthest target depth in its pixel (the shading can see it).
Classes: seen (deficit > 0.5 at a training resolution in some view), resolution-limited (exposed at 256 px in some
view, deficit <= 0.2 at 64 and 96 px, > 0.5 at 256 px), occluded (exposed in no view at 256 px), other.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import json
import sys

import numpy as np
import torch

from physmorph import gpu
from physmorph.losses.silhouette import _view_basis, set_kernel, soft_silhouette_multi
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.render_loss import make_views

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from thin_regions import local_thickness, npz_member, outer_mask  # noqa: E402

RES = (64, 96, 256)


def lattice(p, right, up, res, extent):
    uv = torch.stack([p @ right.T, p @ up.T], -1)                  # (M,V,2)
    return (uv + extent) / (2 * extent) * res


def cic_read(img, p, right, up, res, extent):
    """(M,V): per-view images (V,res,res) read through each point's CIC footprint."""
    rel = lattice(p, right, up, res, extent)
    base = torch.floor(rel).long()
    frac = rel - base.to(rel.dtype)
    V = img.shape[0]
    vi = torch.arange(V, device=p.device).view(1, V).expand(rel.shape[0], V)
    out = torch.zeros(rel.shape[:2], device=p.device, dtype=img.dtype)
    for ox in (0, 1):
        wx = frac[..., 0] if ox else 1 - frac[..., 0]
        for oy in (0, 1):
            wy = frac[..., 1] if oy else 1 - frac[..., 1]
            ii = (base[..., 0] + ox).clamp(0, res - 1)
            jj = (base[..., 1] + oy).clamp(0, res - 1)
            out += wx * wy * img[vi, ii, jj]
    return out


def depth_tests(tgt, q, views, res, extent, R):
    """(M,V) exposed and front flags of points q against the target's depth range in their pixel."""
    right, up = _view_basis(tgt, views)
    th = tgt.new_tensor([float(t) for t, _ in views])
    ph = tgt.new_tensor([float(p) for _, p in views])
    dvec = torch.stack([torch.cos(ph) * torch.sin(th), torch.sin(ph), torch.cos(ph) * torch.cos(th)], 1)
    V = len(views)
    off = (torch.arange(V, device=tgt.device) * res * res).view(1, V)

    def flat(p):
        ij = torch.floor(lattice(p, right, up, res, extent)).long().clamp(0, res - 1)
        return off + ij[..., 0] * res + ij[..., 1]
    ft, dt = flat(tgt).reshape(-1), (tgt @ dvec.T).reshape(-1)
    dmin = torch.full((V * res * res,), float("inf"), device=tgt.device).scatter_reduce(0, ft, dt, "amin")
    dmax = torch.full((V * res * res,), float("-inf"), device=tgt.device).scatter_reduce(0, ft, dt, "amax")
    fq, dq = flat(q), q @ dvec.T
    lo, hi = dq - dmin[fq], dmax[fq] - dq
    return (lo <= R) & (hi <= R), (lo <= R) | (hi <= R)


def analyse(label, path):
    js = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    arm = js["arms"]["render_full_dt_iso_nn"]
    prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in js["provenance"]["mpm"].items()
                       if k in MPMParams.__dataclass_fields__})
    cfg = PipelineConfig()
    set_kernel("cic")
    z = np.load(path, allow_pickle=True)
    frames = npz_member(path, "frames")
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
    x = gpu.tensor(np.asarray(frames[dn - 1], np.float32))
    tgt = gpu.tensor(np.asarray(z["tgt"], np.float32))
    sp = gpu.median(gpu.knn(tgt, 2)[0][:, 1])
    h = local_thickness(tgt, sp) / prm.dx
    out_t = outer_mask(tgt, sp)
    far = gpu.KNN(x).query(tgt, 1)[0][:, 0] > 1.5 * sp
    views = make_views(cfg.render_views, cfg.render_elevs)
    extent = float(tgt.abs().max()) * 1.25
    right, up = _view_basis(tgt, views)
    R = 2.0 * prm.dx
    print(f"\n== {label}: N {len(x)}, {len(views)} views, extent {extent:.2f} wu, pixel at 64 px "
          f"{2 * extent / 64:.3f} wu = {2 * extent / 64 / sp:.1f} spacings, R = 2 cells = {R:.3f} wu")
    rows = []
    for name, lo, hi in (("<1", 0.0, 1.0), ("1-2", 1.0, 2.0)):
        sel = out_t & (h >= lo) & (h < hi)
        U = tgt[sel & far]
        if not len(U):
            continue
        dmax = {}
        for res in RES:
            with torch.no_grad():
                D = (soft_silhouette_multi(tgt, views, res, extent, cfg.sil_k)
                     - soft_silhouette_multi(x, views, res, extent, cfg.sil_k)).clamp_min(0.0)
            dmax[res] = cic_read(D, U, right, up, res, extent).max(1).values
        exp64, front64 = depth_tests(tgt, U, views, 64, extent, R)
        exp256, front256 = depth_tests(tgt, U, views, 256, extent, R)
        seen = (dmax[64] > .5) | (dmax[96] > .5)
        occl = ~exp256.any(1)
        reslim = ~seen & ~occl & (dmax[64] <= .2) & (dmax[96] <= .2) & (dmax[256] > .5)
        other = ~seen & ~occl & ~reslim
        f = lambda m: 100.0 * float(m.float().mean())  # noqa: E731
        r = dict(bin=name, n_uncovered=int(len(U)), n_outer=int(sel.sum()),
                 seen=f(seen), resolution_limited=f(reslim), occluded=f(occl), other=f(other),
                 occluded_front_visible=f(occl & front256.any(1)) / max(f(occl), 1e-9) * 100.0 if bool(occl.any()) else 0.0,
                 exposed64=f(exp64.any(1)), exposed256=f(exp256.any(1)),
                 deficit_median={res: float(dmax[res].median()) for res in RES})
        rows.append(r)
        print(f"  bin {name:>3s}: {r['n_uncovered']} of {r['n_outer']} outer points uncovered | seen {r['seen']:.1f} %  "
              f"resolution-limited {r['resolution_limited']:.1f} %  occluded {r['occluded']:.1f} % "
              f"(front-visible {r['occluded_front_visible']:.0f} % of them)  other {r['other']:.1f} %")
        print(f"            exposed in some view: {r['exposed64']:.1f} % at 64 px, {r['exposed256']:.1f} % at 256 px; "
              f"median of the largest deficit: " + ", ".join(f"{res} px {v:.2f}" for res, v in r["deficit_median"].items()))
    return dict(label=label, rows=rows)


if __name__ == "__main__":
    out = [analyse(*a.split("=", 1)) for a in sys.argv[1:] if not a.startswith("--")]
    js = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--json=")]
    if js:
        json.dump(out, open(js[0], "w"), indent=1)
