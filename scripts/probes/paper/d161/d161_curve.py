"""d161_curve.py OUT_JSONL MESH N [LAMBDA_JSON] -- D161's response curve: for a relief of wavelength w and amplitude delta
(pitches), what each term of the objective does to it at an arrived state, against what the render asks of it.

A measurement on a card with the pipeline's own operators (repo_d161 = freeze-2026-10-15-rt1 + D156's --render_tgt).
MESH: sphere (assets/isosphere.obj; zonal patterns cos(m theta), one exact wavelength) or bunny (assets/bunny.obj;
stripes cos(k y.e) along the body inside a window of 1.8 wu on the flank). The run's TargetPack, scales and units are
rebuilt from the frozen recipe's flags at N, seed 97 (scripts/pipeline_run.prepared / run_config, target.build_target,
calibrate_units, the transport scale at the source as the window objective sets it).

The arrived state X0: an independent sample of the smooth target at seed + 2 (the evaluation's own reference). A relief
state X(ds) = X0 + ds a Phi, Phi the pattern's incompressible field (d161_lib.displacement), a the field's pitch.
Per wavelength w:
  physics  the scaled transport L_T (ot_scale x the debiased grid Sinkhorn at one loss cell's blur, no fine term), the
           surface proximity L_P (ot_scale x SurfaceProximity), the near band L_nn and the spray cleanup L_sp (wu x
           their weights; the selection merit's current-state forms) at ds in DS pitches, and their derivatives along
           Phi (per pitch) at ds = 0 and 1;
  field    the exterior's relief per unit relief of the particles (the discs of X(1) read against P: read_relief), and
           of the outer layer (the pipeline's layer rule) itself;
  window   the outer-layer relaxation of one window (T driven steps, s = -(1/T)(d - dbar - ref), rigid modes removed,
           ref = TargetRelief from the smooth mesh's n/4 surface points, exactly k_layer_resid / relax / project with
           no MPM motion): R = the layer's (and the discs') relief after over before, from X(1); G = the relief at the
           window's end when u = 1 pitch x P (rigid-free, as free_of_rigid) is written from X0;
  render   for each target amplitude dt in DT: the render's target pictures drawn from the target sample and its seven
           draws displaced by dt a Phi (render_x / draws of build_target, as --render_tgt would draw a relief mesh's
           samples); L_r(X(ds)) = silhouette + w_pbr shading of the state's exterior discs against them, at every ds;
           its derivative along Phi at ds = 0 and its gradient norm; the same of X0 against the smooth pictures (the
           floor and the reference norm).
  totals   E(ds) = L_T + L_P + L_nn + L_sp + lambda L_r(ds) with lambda the tag's late windows' (LAMBDA_JSON: the
           median of the last ten committed windows of the tag's render arm at this N) and lambda_adapt = that x
           |grad L_r(X0; smooth)| / |grad L_r(X0; relief)| (the rule lambda |g_r| = lambda_auto |g_p| holds the
           render's push; a larger render gradient takes a smaller lambda); the minimisers by a parabola through the
           three lowest points; the position-space PCGrad share: the part of the render's push along Phi that the
           one-sided projection on the physics gradient removes at X(ds_star).
One JSON line per (w, dt), plus a head line (constants, units, lambda, the step of the tag's late windows)."""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import physmorph  # noqa: E402,F401  (before torch: CuPy's CUDA 12 compiler loads first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from physmorph import gpu  # noqa: E402

if os.environ.get("D161_CPU") == "1":
    # the same operators on the CPU (torch), for when no card is free: the device helpers' three CuPy pieces (the KD-tree,
    # the distance transform, the layer's hash-grid kNN) by scipy (PHYSMORPH_KNN=cpu for the last); float rounding only
    from scipy.ndimage import distance_transform_edt as _edt
    from scipy.spatial import cKDTree as _Tree
    assert os.environ.get("PHYSMORPH_KNN") == "cpu", "set PHYSMORPH_KNN=cpu"
    gpu.DEVICE = "cpu"

    class _KNN:
        def __init__(self, points):
            self.n = int(points.shape[0])
            self.tree = _Tree(points.detach().double().cpu().numpy())

        def query(self, queries, k=1):
            k = int(min(k, self.n))
            d, i = self.tree.query(queries.detach().double().cpu().numpy(), k=k, workers=int(os.environ.get("OMP_NUM_THREADS", "8")))
            return (torch.as_tensor(d, dtype=torch.float64).reshape(-1, k),
                    torch.as_tensor(i, dtype=torch.int64).reshape(-1, k))

    gpu.KNN = _KNN
    gpu.edt = lambda outside: torch.as_tensor(_edt(outside.cpu().numpy())).double()
from physmorph.losses.grid_ot import GridSinkhornLoss  # noqa: E402
from physmorph.losses.volumetric import d_nn_band_current, d_vol_density, d_w1, isolation_gate  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior  # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units, target_relief  # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame  # noqa: E402
from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d161_lib import Stripes, Surface, Zonal, displacement, read_relief, target_frame  # noqa: E402

torch.set_grad_enabled(True)
out_path, mesh, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
lam_json = sys.argv[4] if len(sys.argv) > 4 else None
SEED = 97
TGT = {"sphere": "assets/isosphere.obj", "bunny": "assets/bunny.obj"}[mesh]
WS = [4, 6, 8, 11, 16, 22, 32, 45] if mesh == "sphere" else [4, 6, 8, 11, 16, 22]
DS = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
DT = [0.25, 0.5, 1.0, 2.0]
out = open(out_path, "a")


def emit(row):
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row)[:400], flush=True)


t0 = time.time()
args = parse_args(["--tgt", TGT, "--n", str(N), "--seed", str(SEED)])
cfg0, prep = prepared(args, SEED)
cfg = run_config(args, cfg0, prep)
prm = prep.prm
frame, x_t = target_frame(args.src, args.tgt, N, SEED)
assert np.abs(x_t - prep.tgt).max() < 1e-6, "the frame does not reproduce the run's target sample"
tgt0 = build_target(prep.tgt, prm, cfg, draws=prep.tgt_draws)
relief_ref = target_relief(tgt0.pts, prep.tgt_surface, cfg)
src_t = gpu.tensor(prep.src)
calibrate_units(tgt0, src_t, cfg)
wu = 1.0 / tgt0.unit_ratio
ot_T = GridSinkhornLoss(tgt0.grid, tgt0.lgmin, tgt0.ldx, tgt0.ldims, eps=float(tgt0.ldx) ** 2, iters=cfg.ot_iters,
                        tol=min(cfg.ot_tol, 1e-3), mass_total=float(tgt0.m.sum()), cuda_blocks=True, support=None)
ot_F = GridSinkhornLoss(tgt0.grid, tgt0.lgmin, tgt0.ldx, tgt0.ldims, eps=float(tgt0.ldx) ** 2, iters=cfg.ot_iters,
                        tol=min(cfg.ot_tol, 1e-3), mass_total=float(tgt0.m.sum()), cuda_blocks=True,
                        support=tgt0.support)
xs = src_t.clone().requires_grad_(True)
gd = torch.autograd.grad(d_vol_density(xs, tgt0.m, tgt0.grid, tgt0.lgmin, tgt0.ldx, tgt0.ldims, tgt0.m_ref,
                                       tgt0.n_support), xs)[0].norm()
xs = src_t.clone().requires_grad_(True)
gt = torch.autograd.grad(ot_F.state_energy(xs, tgt0.m), xs)[0].norm()
ot_scale = float(gd / gt.clamp_min(1e-30))
berth = cfg.nn_berth_k * tgt0.nn_spacing
a = float(tgt0.ext.pitch)
ones = torch.ones_like(tgt0.m)

lam_tag, alpha_tag = None, None
if lam_json:
    js = json.load(open(lam_json))
    hist = [h for h in list(js["arms"].values())[0]["history"] if h.get("frame_end") and not h.get("null_commit")]
    lam_tag = float(np.median([h["lambda"] for h in hist[-10:]]))
    alpha_tag = float(np.median([h["alpha_last"] for h in hist[-10:]]))

X0n = draws_in_frame(args.tgt, frame, N, [SEED + 2])[0].astype(np.float64)
surf = Surface(args.tgt, frame, 1_000_000 if N > 100000 else 400_000, prep.tgt.astype(np.float64))
foot0 = surf.foot(X0n)
footT = surf.foot(prep.tgt.astype(np.float64))
footD = [surf.foot(d.astype(np.float64)) for d in prep.tgt_draws]
X0 = gpu.tensor(X0n)
sp0 = layer_spacing(X0)
center = surf.P.mean(0)
R_mean = float(np.linalg.norm(surf.P - center, axis=1).mean())
if mesh == "bunny":
    xc = prep.tgt.mean(0)
    ext = prep.tgt.max(0) - prep.tgt.min(0)
    lat = np.eye(3)[0] if ext[0] < ext[2] else np.eye(3)[2]          # the narrower horizontal axis: the flank's normal
    fwd = np.eye(3)[2] if ext[0] < ext[2] else np.eye(3)[0]
    band = np.abs(surf.P[:, 1] - xc[1]) < 0.5
    c_f = surf.P[band][np.argmax((surf.P[band] - xc) @ lat)]
emit(dict(head=True, mesh=mesh, N=N, pitch=a, sp0_layer=sp0, ldx=float(tgt0.ldx), loss_cell_pitch=float(tgt0.ldx) / a,
          mpm_dx_pitch=float(prm.dx) / a, render_res=cfg.render_res, pixel_pitch=2 * tgt0.extent / cfg.render_res / a,
          ext_radius=tgt0.ext.radius, ext_offset=tgt0.ext.offset, ot_scale=ot_scale, unit_ratio=tgt0.unit_ratio, wu=wu,
          berth_pitch=berth / a, nn_spacing_pitch=tgt0.nn_spacing / a, sup_radius_pitch=tgt0.support.radius / a,
          lam_tag=lam_tag, alpha_tag=alpha_tag, u_window_tag_pitch=(cfg.iters * alpha_tag / a) if alpha_tag else None,
          R_mean=R_mean, center=center.tolist(), flank=(c_f.tolist() if mesh == "bunny" else None),
          setup_s=time.time() - t0))


def phys_terms(x, grad_dir=None):
    """dict of the physics terms at x (and, with grad_dir (N,3), each one's derivative along it)."""
    xg = x.detach().clone().requires_grad_(grad_dir is not None)
    terms = {}
    with torch.set_grad_enabled(grad_dir is not None):
        terms["T"] = ot_scale * ot_T.state_energy(xg, tgt0.m)
        terms["P"] = ot_scale * tgt0.support.penalty(xg)
        terms["nn"] = wu * cfg.w_nn * d_nn_band_current(xg, tgt0.m, tgt0.pts, ones, berth, tgt0.knn,
                                                         far=float(tgt0.ldx))
        terms["sp"] = wu * cfg.w_dt * d_w1(xg, tgt0.m * isolation_gate(xg, cfg.dt_iso_lo, cfg.dt_iso_hi), tgt0.dt3,
                                           tgt0.dtgmin, tgt0.dtdx, tgt0.dtdims)
    val = {k: float(v.detach()) for k, v in terms.items()}
    der, grads = {}, {}
    if grad_dir is not None:
        for k, v in terms.items():
            g = torch.autograd.grad(v, xg, retain_graph=True, allow_unused=True)[0]
            g = torch.zeros_like(xg) if g is None else g.detach()
            grads[k] = g
            der[k] = float((g * grad_dir).sum())
    return val, der, grads


def render_term(x, pack, grad_dir=None):
    e = pack.ext
    discs = tracked(make_field(e.field, x.detach(), e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                    e.lattice, e.h, e.skin)
    xg = x.detach().clone().requires_grad_(grad_dir is not None)
    with torch.set_grad_enabled(grad_dir is not None):
        p, n, _ = discs.read(xg)
        sil, pbr = d_exterior(p, n, e.sils, e.shade, pack.views, cfg.render_res, pack.extent, cfg.sil_k, cfg.w_hole,
                              cfg.w_spray, cfg.pbr_ambient)
        L = sil + cfg.w_pbr * pbr
    res = dict(L=float(L.detach()), sil=float(sil.detach()), pbr=float(pbr.detach()))
    g = None
    if grad_dir is not None:
        g = torch.autograd.grad(L, xg)[0].detach()
        res.update(der=float((g * grad_dir).sum()), gnorm=float(g.norm()))
    return res, g, p.detach()


def relax_window(x0, u_amp, P_layer=None):
    """One window of the outer-layer relaxation (T steps, 1/T a step) with u (rigid-free) written at u_amp x P on the
    layer, as k_layer_resid / k_layer_relax / k_layer_project do it with no MPM motion. Returns (end state, mask)."""
    sp = layer_spacing(x0)
    mask, nrm, nbr, w = layer_relax_data(x0, sp, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    ref = relief_ref.at(x0, mask, nrm, nbr, w)
    m = mask
    xd = x0.double()
    rn = torch.linalg.cross(xd - xd.mean(0), nrm.double(), dim=1) * m.double()[:, None]
    modes = torch.cat((nrm.double() * m.double()[:, None], rn), 1)
    Minv = torch.linalg.pinv(modes.T @ modes).float()
    rn = rn.float()
    u = torch.zeros(len(x0), device=x0.device)
    if u_amp != 0.0 and P_layer is not None:
        u = (u_amp * P_layer * m).float()
        A = torch.cat((m[:, None] * nrm, m[:, None] * torch.linalg.cross(x0 - x0.mean(0), nrm, dim=1)), 1).double()
        u = u - (A @ (torch.linalg.pinv(A.T @ A) @ (A.T @ u.double()))).float()
    T = cfg.T
    x = x0.clone()
    for _ in range(T):
        c = (w[..., None] * x[nbr]).sum(1)
        d = (nrm * (x - c)).sum(1) * m
        dbar = (w * d[nbr]).sum(1)
        s = -(1.0 / T) * (d - dbar - ref) * m
        b0, b1 = (s[:, None] * nrm).sum(0), (s[:, None] * rn).sum(0)
        ct = Minv[:3, :3] @ b0 + Minv[:3, 3:] @ b1
        cr = Minv[3:, :3] @ b0 + Minv[3:, 3:] @ b1
        spp = s - nrm @ ct - rn @ cr
        x = x + (((1.0 / T) * u + spp) * m)[:, None] * nrm
    return x, mask > 0.5


def layer_read(x, Pv, Wv):
    sp = layer_spacing(x)
    mask, _, _, _ = layer_relax_data(x, sp, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    q = x[mask > 0.5].detach().cpu().numpy().astype(np.float64)
    return read_relief(q, surf, Pv, Wv, reach=3 * a)


def disc_read(p, Pv, Wv):
    return read_relief(p.cpu().numpy().astype(np.float64), surf, Pv, Wv, reach=3 * a)


def parabola_min(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    j = int(np.argmin(ys))
    if j == 0 or j == len(xs) - 1:
        return float(xs[j]), float(ys[j]), False
    x3, y3 = xs[j - 1:j + 2], ys[j - 1:j + 2]
    c = np.polyfit(x3, y3, 2)
    if c[0] <= 0:
        return float(xs[j]), float(ys[j]), False
    xm = -c[1] / (2 * c[0])
    return float(xm), float(np.polyval(c, xm)), True


# the floor: X0 against the smooth pictures
r00, g00, _ = render_term(X0, tgt0, grad_dir=torch.zeros_like(X0))
for w in WS:
    t1 = time.time()
    if mesh == "sphere":
        m_int = max(1, int(round(2 * math.pi * R_mean / (w * a))))
        pat = Zonal(center, np.array([0.0, 0.0, 1.0]), R_mean, m_int)
    else:
        pat = Stripes(fwd, 2 * math.pi / (w * a), c_f, 1.8)
    Pv, Gv, Wv = pat.at(surf)
    k = pat.k
    w_true = pat.wavelength / a
    Phi0 = gpu.tensor(displacement(X0n, surf, Pv, Gv, k, foot0))
    PhiT = displacement(prep.tgt.astype(np.float64), surf, Pv, Gv, k, footT)
    PhiD = [displacement(d.astype(np.float64), surf, Pv, Gv, k, f) for d, f in zip(prep.tgt_draws, footD)]
    dirv = Phi0 * a                                          # the derivative along Phi per pitch of relief
    # physics along ds
    phys = {}
    for ds in DS:
        v, dd, _ = phys_terms(X0 + ds * a * Phi0, grad_dir=dirv if ds in (0.0, 1.0) else None)
        phys[ds] = dict(val=v, der=dd)
    # the field and the layer at X(1); the relaxation and u over one window
    X1 = X0 + a * Phi0
    _, _, p1 = render_term(X1, tgt0)
    field = dict(disc=disc_read(p1, Pv, Wv), layer=layer_read(X1, Pv, Wv), layer0=layer_read(X0, Pv, Wv))
    xr, _ = relax_window(X1, 0.0)
    _, _, pr = render_term(xr, tgt0)
    win = dict(R_layer=layer_read(xr, Pv, Wv), R_disc=disc_read(pr, Pv, Wv))
    Pl = gpu.tensor(Pv[foot0[0]])
    xu, _ = relax_window(X0, a, Pl)
    _, _, pu = render_term(xu, tgt0)
    win.update(G_layer=layer_read(xu, Pv, Wv), G_disc=disc_read(pu, Pv, Wv))
    xu0, _ = relax_window(X0, 0.0)
    win.update(G0_layer=layer_read(xu0, Pv, Wv))
    # render against the relief pictures, per target amplitude
    for dt in DT:
        rx = (prep.tgt + dt * a * PhiT).astype(np.float32)
        dr = [(d + dt * a * p).astype(np.float32) for d, p in zip(prep.tgt_draws, PhiD)]
        pack = build_target(prep.tgt, prm, cfg, draws=dr, render_x=rx)
        rend = {}
        for ds in DS:
            gdir = dirv if ds in (0.0, 1.0) else None
            r, g, _ = render_term(X0 + ds * a * Phi0, pack, grad_dir=gdir)
            rend[ds] = r
        r0s, _, _ = render_term(X0, pack)
        Lr = [rend[ds]["L"] for ds in DS]
        ds_r, Lr_min, ok_r = parabola_min(DS, Lr)
        lam_adapt = (lam_tag * g00.norm().item() / max(rend[0.0]["gnorm"], 1e-30)) if lam_tag else None
        tot = {}
        for name, lam in (("tag", lam_tag), ("adapt", lam_adapt)):
            if lam is None:
                continue
            E = [sum(phys[ds]["val"].values()) + lam * rend[ds]["L"] for ds in DS]
            Ep = [sum(phys[ds]["val"].values()) for ds in DS]
            dmin, Emin, ok = parabola_min(DS, E)
            tot[name] = dict(lam=lam, ds_star=dmin, ok=ok, E=E, E_phys=Ep,
                             push_render=-lam * rend[0.0]["der"],               # > 0: the render asks for the relief
                             hold_phys_at1=sum(phys[1.0]["der"].values()))     # > 0: the physics resists 1 pitch
        # PCGrad (position space) at X(1): the share of the render's push along Phi the projection removes
        pc = None
        if lam_tag is not None:
            _, _, gp = phys_terms(X1, grad_dir=dirv)
            gps = sum(gp.values())
            _, gr, _ = render_term(X1, pack, grad_dir=dirv)
            dot = float((gps * gr).sum())
            along = float((gr * dirv).sum())
            if dot < 0:
                gr2 = gr - (dot / float((gps * gps).sum())) * gps
                pc = dict(cos=dot / float(gps.norm() * gr.norm()), along=along, along_after=float((gr2 * dirv).sum()))
            else:
                pc = dict(cos=dot / float(gps.norm() * gr.norm()), along=along, along_after=along)
        emit(dict(mesh=mesh, N=N, w_nominal=w, w=w_true, m=getattr(pat, "m", None), dt=dt, ds=DS,
                  phys={str(k_): v for k_, v in phys.items()}, field=field, window=win,
                  render={str(k_): v for k_, v in rend.items()}, floor_smooth=r00["L"], X0_vs_relief=r0s["L"],
                  render_only=dict(ds_star=ds_r, L_min=Lr_min, ok=ok_r), totals=tot, pcgrad=pc,
                  secs=time.time() - t1))
        del pack
        torch.cuda.empty_cache()
emit(dict(done=True, secs=time.time() - t0))
