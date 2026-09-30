"""thin_regions.py LABEL=ARCHIVE.npz ... — end-state defects by the target's local feature thickness.

The target's local thickness h (twice the radius of the largest ball inside the target that contains the point,
Hildebrand & Ruegsegger 1997) is measured on a voxel grid at the target spacing; every body particle takes the h of
its nearest target point. Per thickness bin, in MPM cells (<1, 1-2, 2-4, >=4; the loss cell of a run whose loss
grid is the MPM grid, so runs with a finer loss grid bin the same features):
  target side   outer target points farther than 1.5 target spacings from the body (uncovered share), and farther
                than 1.5 spacings of a 40k sampling (uncovered_40k: one world distance for every N);
  body side     sparsity (8th-neighbour distance / the target's median), stretch (largest singular value of F),
                anisotropy (largest / smallest), det F;
  render        silhouette pixels of the bin's target points not covered by the body (24 views, 256 px);
  gradients     per-particle position gradients of the settled objective at the end state (velocity 0): the
                transport part and the local-support part of E + E wB / (E + wB), the lambda-weighted render
                term, the cleanup terms; the support penalty of the body and of the TARGET itself.
The JSON run file next to the archive (<tag>.json) supplies the grid, the loss resolution and lambda.
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import argparse
import json
import math
import struct
import zipfile

import numpy as np
import torch

from physmorph import gpu
from physmorph.losses.grid_ot import GridSinkhornLoss
from physmorph.losses.volumetric import (d_nn_band, d_vol_density, d_w1, isolation_gate, nn_band_assign,
                                         rasterize_mass)
from physmorph.metrics import _splat_body
from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.render_loss import d_pbr, d_render, make_views
from physmorph.pipeline.target import build_target, calibrate_units
from physmorph.render.knn_gpu import knn_self_torch

BINS = (0.0, 1.0, 2.0, 4.0, math.inf)


def npz_member(path, name):
    """A member of an uncompressed .npz as a read-only memmap (the 300k archives hold GBs of frames)."""
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(name + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        return np.load(path, allow_pickle=True)[name]
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        n, m = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def local_thickness(tgt, sp):
    """h at every target point: max-ball thickness on a voxel grid of the target spacing."""
    lo = tgt.min(0).values - 3 * sp
    dims = [int(math.ceil(float(v))) for v in ((tgt.max(0).values + 3 * sp - lo) / sp)]
    axes = [lo[i] + sp * (torch.arange(dims[i], device=tgt.device) + 0.5) for i in range(3)]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tree = gpu.KNN(tgt)
    d = torch.cat([tree.query(c, 1)[0][:, 0] for c in centers.split(2_000_000)])
    # inside the sampled volume: within the sampling's covering radius of a target point (0.75 x the median
    # 8th-neighbour distance, about 1.06 fill pitches on a jittered lattice; the median nearest-neighbour
    # distance is shorter than the pitch and leaves the volume full of holes)
    r_cover = 0.75 * gpu.median(knn_self_torch(tgt, 9)[0][:, 8])
    occ = (d <= r_cover).reshape(dims)
    # interior voxels the jittered sampling happens to miss are enclosed: fill them, or the inside distance
    # (and so h) is capped by the distance to the nearest such hole, a fixed number of voxels at every N
    occ = gpu.fill_holes(occ)
    D = gpu.edt(occ)                                           # voxels to the outside
    thick = torch.zeros(dims, dtype=torch.float64, device=tgt.device)
    for r in range(1, int(D.max()) + 1):                       # balls of radius r cover thickness >= 2r
        core = D >= r
        if not bool(core.any()):
            break
        thick[(gpu.edt(~core) <= r) & occ] = 2.0 * r * sp
    ijk = ((tgt - lo) / sp).long()
    for i in range(3):
        ijk[:, i] = ijk[:, i].clamp(0, dims[i] - 1)
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return torch.where(h > 0, h, torch.full_like(h, 2 * sp))  # a point off the closed volume: thinnest


def outer_mask(P, sp):
    d, _ = knn_self_torch(P, 41)
    c = (d[:, 1:] < 2.0 * sp).sum(1).float()
    return c < 0.6 * c.median()


def per_point_support(sup, x):
    return sup.penalty_per_point(x)


def analyse(label, path):
    js = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    arm = js["arms"]["render_full_dt_iso_nn"]
    mpm = js["provenance"]["mpm"]
    prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mpm.items()
                       if k in MPMParams.__dataclass_fields__})
    cfg0 = arm["config"]
    cfg = PipelineConfig(loss_res=int(cfg0["loss_res"]), unit_ref_res=int(cfg0.get("unit_ref_res", 64)),
                         nn_berth_k=float(cfg0.get("nn_berth_k", 1.0)),
                         support_target_ref=bool(cfg0.get("support_target_ref", False)),
                         support_form=str(cfg0.get("support_form", "log")))
    z = np.load(path, allow_pickle=True)
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else None
    frames = npz_member(path, "frames")
    dn = dn or len(frames)
    x = gpu.tensor(np.asarray(frames[dn - 1], np.float32))
    tgt_np, src_np = np.asarray(z["tgt"], np.float32), np.asarray(z["src"], np.float32)
    tgt = gpu.tensor(tgt_np)
    idx = list(z["F_sample_idx"]); F = gpu.tensor(z["F_samples"][idx.index(dn - 1)] if dn - 1 in idx else z["F_samples"][-1])
    ldx = prm.dx * prm.nx / cfg.loss_res
    sp_t = gpu.median(gpu.knn(tgt, 2)[0][:, 1])
    h_t = local_thickness(tgt, sp_t) / prm.dx                # in MPM cells (the loss cell of a D/26 run)
    tree_t = gpu.KNN(tgt)
    h_x = h_t[tree_t.query(x, 1)[1][:, 0]]
    # target side: uncovered outer points
    out_t = outer_mask(tgt, sp_t)
    d_body = gpu.KNN(x).query(tgt, 1)[0][:, 0]
    far = d_body > 1.5 * sp_t
    far40 = d_body > 1.5 * sp_t * (len(tgt) / 40000.0) ** (1.0 / 3.0)
    # body side
    d8 = knn_self_torch(x, 9)[0][:, 8].double() / gpu.median(knn_self_torch(tgt, 9)[0][:, 8])
    sv = torch.linalg.svdvals(F.reshape(-1, 3, 3).double())
    J = sv.prod(1)
    # render: silhouette pixels of each bin's target points left uncovered
    ext = float(tgt.norm(dim=1).max()) * 1.15
    views = make_views(8, (0.0, 0.5, -0.5))
    body_m = [_splat_body(x, 256, th, ph, ext) for th, ph in views]
    # gradients of the settled objective at the end state
    pack = build_target(tgt_np, prm, cfg)
    src = gpu.tensor(src_np)
    calibrate_units(pack, src, cfg)
    ot = GridSinkhornLoss(pack.grid, pack.lgmin, pack.ldx, pack.ldims, eps=pack.ldx ** 2, iters=cfg.ot_iters,
                          tol=min(cfg.ot_tol, 1e-3), mass_total=float(pack.m.sum()), cuda_blocks=True)
    horizon = cfg.T * prm.dt

    def E_of(q):
        return ot.state_energy(q, pack.m, torch.zeros_like(q), horizon)
    qs = src.clone().requires_grad_(True)
    gd = torch.autograd.grad(d_vol_density(qs, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims, pack.m_ref,
                                           pack.n_support), qs)[0].norm()
    qs2 = src.clone().requires_grad_(True)
    ot_scale = float(gd / torch.autograd.grad(E_of(qs2), qs2)[0].norm())
    q = x.clone().requires_grad_(True)
    E = E_of(q); gE = torch.autograd.grad(E, q)[0]
    q2 = x.clone().requires_grad_(True)
    B = pack.support.penalty(q2); gB = torch.autograd.grad(B, q2)[0]
    w = cfg.support_weight; Ev, Bv = float(E), float(B)
    c_tr = ot_scale * (1 + (w * Bv / (Ev + w * Bv)) ** 2); c_sup = ot_scale * w * (Ev / (Ev + w * Bv)) ** 2
    lam = [r for r in arm["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= dn]
    lam = float(lam[-1]["lambda"] or 0.0) if lam else 0.0
    q3 = x.clone().requires_grad_(True)
    Lr = (d_render(q3, pack.sils, pack.views, cfg.render_res, pack.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)
          + cfg.w_pbr * d_pbr(q3, pack.shade, pack.views, cfg.render_res, pack.extent, pack.pgmin, pack.pdx,
                              pack.pdims, cfg.sil_k, cfg.pbr_ambient, pack.pblur))
    gR = lam * torch.autograd.grad(Lr, q3)[0]
    q4 = x.clone().requires_grad_(True)
    wu = 1.0 / pack.unit_ratio
    m_dt = pack.m * isolation_gate(x, cfg.dt_iso_lo, cfg.dt_iso_hi)
    nn_idx, nn_el = nn_band_assign(x, pack.knn, pack.nn_spacing, cfg.nn_berth_k, cfg.nn_far_k)
    Lc = wu * (cfg.w_dt * d_w1(q4, m_dt, pack.dt3, pack.dtgmin, pack.dtdx, pack.dtdims)
               + cfg.w_nn * d_nn_band(q4, pack.m, pack.pts, nn_idx, nn_el, cfg.nn_berth_k * pack.nn_spacing))
    gC = torch.autograd.grad(Lc, q4)[0]
    Bx, Bt = per_point_support(pack.support, x), per_point_support(pack.support, tgt)
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        bt, bx = (h_t >= lo) & (h_t < hi), (h_x >= lo) & (h_x < hi)
        bto = bt & out_t
        mask_holes = []
        for (th, ph), bm in zip(views, body_m):
            tm = _splat_body(tgt[bt], 256, th, ph, ext) if bool(bt.any()) else None
            if tm is not None and int(tm.sum()):
                mask_holes.append(float((tm & ~bm).sum()) / float(tm.sum()))
        f = lambda v, sel: float(v[sel].double().mean()) if bool(sel.any()) else float("nan")  # noqa: E731
        rows.append(dict(bin=f"{lo:g}-{hi:g}", n_tgt=int(bt.sum()), n_body=int(bx.sum()),
                         uncovered=f(far.float(), bto), uncovered_40k=f(far40.float(), bto),
                         holes=float(np.mean(mask_holes)) if mask_holes else float("nan"),
                         sparsity=f(d8, bx), sparse_frac=f((d8 > 1.5).float(), bx), stretch_p90=(
                             float(torch.quantile(sv[bx, 0], 0.9)) if bool(bx.any()) else float("nan")),
                         aniso_p90=(float(torch.quantile(sv[bx, 0] / sv[bx, 2].clamp_min(1e-9), 0.9))
                                    if bool(bx.any()) else float("nan")), J=f(J, bx),
                         g_transport=f(c_tr * gE.norm(dim=1), bx), g_support=f(c_sup * gB.norm(dim=1), bx),
                         g_render=f(gR.norm(dim=1), bx), g_cleanup=f(gC.norm(dim=1), bx),
                         B_body=f(Bx, bx), B_target=f(Bt, bt), B_target_pos=f((Bt > 0).float(), bt)))
    head = dict(label=label, frames=dn, loss_cell=ldx, target_spacing=sp_t, cell_over_spacing=ldx / sp_t,
                E=Ev, B=Bv, support_coef=c_sup / ot_scale, lam=lam, ot_scale=ot_scale)
    print(f"\n== {label}: loss cell {ldx:.4f} wu = {ldx / sp_t:.2f} target spacings; E {Ev:.3e}, wB {w * Bv:.3e}, "
          f"support gradient weight w(E/(E+wB))^2 = {c_sup / ot_scale:.3e}, lambda {lam:.3g}")
    cols = ("bin", "n_tgt", "n_body", "uncovered", "uncovered_40k", "holes", "sparsity", "sparse_frac", "stretch_p90", "aniso_p90", "J",
            "g_transport", "g_support", "g_render", "g_cleanup", "B_body", "B_target", "B_target_pos")
    print(" ".join(f"{c:>11s}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]:>11}" if isinstance(r[c], (str, int)) else f"{r[c]:>11.4g}" for c in cols))
    return dict(head=head, rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=ARCHIVE.npz")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = [analyse(*r.split("=", 1)) for r in a.runs]
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
