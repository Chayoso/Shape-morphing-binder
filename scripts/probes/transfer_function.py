"""Probe: spatial transfer function of the MPM adjoint (docs/root_analysis.md §2).

Root claim under test: the render covector's unique HIGH-FREQUENCY content is attenuated
by the MPM adjoint (cubic B-spline P2G/G2P + F-smoothing) before it reaches dFc.  Nobody
had measured it.  This script measures, on a WARM flagship state (4 accepted commits,
promoted exactly like scripts/dress_bench.py: x/F/v/C, Fp = I, no gauss):

  B. gain ||J^T g|| of the x_T -> dFc adjoint for unit covectors g at controlled spatial
     scales (white field low-passed by k rounds of 16-NN mean smoothing), for a pure
     translation field, and for the real render / physics covectors; swept over window
     length T in {1,5,20} and F-smoothing s in {0.955, 0.0, 1.0}.
  C. band decomposition (successive smoothing differences) of the real silhouette
     covector g_r = d D_render/d x_T and the physics covector g_p = d(D_vol+w_kin kin)/d x_T:
     energy per band, adjoint gain per band, transmitted-energy share per band.
  D. gain of the F_T -> dFc path for the same bands (the gauss/covariance channel).

NOTE on s (physmorph/mpm/kernels.py k_update): F_out = (1-s) F_new + s F_in.  s is a
TEMPORAL blend, not a spatial filter; s=1.0 FREEZES F (dFc never enters F), s=0.0
disables the smoothing.  Both are measured.

Run (hyde06, GPU 0 only):
  cd /tmp/pm31 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=0 $PY scripts/probes/transfer_function.py
Results: JSON at --out (default /tmp/pm31_tf_results.json) + tables on stdout.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from physmorph.losses.volumetric import d_vol  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm_full  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.optimizer import optimize_window  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_render  # noqa: E402
from physmorph.pipeline.runner import build_target  # noqa: E402
from physmorph.sampling import load_normalized as load  # noqa: E402

BANDS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]   # rounds of kNN-mean smoothing.  At
# N=20000 the 16-NN radius is ~0.1 wu, so 32 rounds only reach ~1 dx; the sweep must
# cross the cubic B-spline support (2 dx = 1.0 wu), hence the extension to 256.
REF = BANDS[-1]                                  # normaliser: the smoothest band
DEV = "cuda"


# ── helpers ──────────────────────────────────────────────────────────────────
def gpu_mem_mib() -> float:
    """This process's GPU memory (MiB) as nvidia-smi sees it (torch + warp)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            p, m = [t.strip() for t in line.split(",")]
            if int(p) == os.getpid():
                return float(m)
    except Exception:
        pass
    return float("nan")


def knn_index(x: np.ndarray, k: int) -> torch.Tensor:
    idx = cKDTree(x).query(x, k=k + 1, workers=-1)[1][:, 1:]
    return torch.as_tensor(np.ascontiguousarray(idx), device=DEV)


def smooth(u: torch.Tensor, idx: torch.Tensor, rounds: int) -> torch.Tensor:
    """`rounds` applications of the (self + kNN) mean on a (N,C) field."""
    for _ in range(rounds):
        u = (u + u[idx].sum(1)) / float(idx.shape[1] + 1)
    return u


def levels(u: torch.Tensor, idx: torch.Tensor, bands=BANDS) -> dict:
    """{k: S^k u} for every k in bands (incremental)."""
    out, cur, done = {}, u, 0
    for k in bands:
        cur = smooth(cur, idx, k - done)
        done = k
        out[k] = cur
    return out


def band_split(u: torch.Tensor, idx: torch.Tensor, bands=BANDS) -> dict:
    """Successive-difference bands: [k_i, k_{i+1}) = S^{k_i}u - S^{k_{i+1}}u, plus the
    residual S^{k_last} u (lowest band).  Keys: 'k0-1', ..., 'k32+'."""
    lv = levels(u, idx, bands)
    out = {}
    for a, b in zip(bands[:-1], bands[1:]):
        out[f"k{a}-{b}"] = lv[a] - lv[b]
    out[f"k{bands[-1]}+"] = lv[bands[-1]]
    return out


def unit(u: torch.Tensor) -> torch.Tensor:
    return u / u.norm().clamp_min(1e-30)


def band_radii(x_t: torch.Tensor, idx: torch.Tensor, bands=BANDS, n_probe=64, seed=1):
    """RMS radius (world units) of S^k applied to a one-hot particle field — the
    honest spatial scale of each band.  Averaged over n_probe random particles."""
    N = x_t.shape[0]
    g = torch.Generator().manual_seed(seed)
    pj = torch.randint(0, N, (n_probe,), generator=g).to(DEV)
    E = torch.zeros(N, n_probe, device=DEV)
    E[pj, torch.arange(n_probe, device=DEV)] = 1.0
    d2 = ((x_t[:, None, :] - x_t[pj][None, :, :]) ** 2).sum(-1)      # (N, n_probe)
    out, cur, done = {}, E, 0
    for k in bands:
        cur = smooth(cur, idx, k - done)
        done = k
        r = torch.sqrt((cur * d2).sum(0) / cur.sum(0).clamp_min(1e-30))
        out[k] = float(r.mean())
    return out


def adj_norm(outs, dfc, seeds) -> tuple[float, torch.Tensor]:
    """||J^T seed|| through the dFc adjoint (one tape backward); returns (norm, grad)."""
    g = torch.autograd.grad(outs, dfc, grad_outputs=seeds, retain_graph=True)[0]
    return float(g.norm()), g


# ── A. warm state ────────────────────────────────────────────────────────────
def warm_state(n: int, commits: int, log):
    prm = MPMParams()
    src = load("assets/isosphere.obj", n, 1)
    tgt_x = load("assets/bunny.obj", n, 2)
    cfg = PipelineConfig(T=20, iters=8, animations=8, loss_res=64)
    cfg.lambda_auto, cfg.w_kin = 0.5, 5.0
    cfg.w_dt, cfg.w_nn, cfg.w_jvol = 0.2, 0.2, 50.0
    cfg.assim_iso = True                       # (no assimilation is run here: Fp = I)
    tgt = build_target(tgt_x, prm, cfg)
    vol0 = compute_rest_volumes(src, 1.0, prm, cfg.device)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    x, st = src.copy(), {"F": None, "v": None, "C": None}
    Fp = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    rec, acc, tries = [], 0, 0
    while acc < commits and tries < 3 * commits:
        t0 = time.perf_counter()
        fr, F_seq, end, _s, whist, stats = optimize_window(
            x, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            log=lambda *_: None, vol0=vol0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tries += 1
        if whist:
            x = np.ascontiguousarray(fr[-1], np.float32)
            st = {"F": end["F"], "v": end["v"], "C": end["C"]}
            acc += 1
            w = whist[-1]
            rec.append({"commit": acc, "secs": dt, "d_vol": w["d_vol"], "kin": w["kin"],
                        "d_render": w["d_render"], "lambda": w["lambda"],
                        "g_phys_norm": stats.get("g_phys_norm"),
                        "g_rend_norm": stats.get("g_rend_norm"),
                        "g_cos": stats.get("g_cos"),
                        "accepted": stats["accepted"], "rejected": stats["rejected"]})
            log(f"[warm] commit {acc}: {dt:.1f}s d_vol={w['d_vol']:.3f} "
                f"d_render={w['d_render']:.5f} lam={w['lambda']:.3g} "
                f"|g_p|={stats.get('g_phys_norm'):.4g} |g_r|={stats.get('g_rend_norm'):.4g} "
                f"cos={stats.get('g_cos'):.3f} acc/rej={stats['accepted']}/{stats['rejected']}")
        else:
            log(f"[warm] try {tries}: no accepted step ({dt:.1f}s)")
    if acc < commits:
        raise RuntimeError(f"only {acc}/{commits} accepted commits")
    vmag = np.linalg.norm(st["v"], axis=1)
    detF = np.linalg.det(st["F"])
    log(f"[warm] state: |v| mean={vmag.mean():.4f} max={vmag.max():.4f} "
        f"detF min/med/max={detF.min():.3f}/{np.median(detF):.3f}/{detF.max():.3f} "
        f"gpu={gpu_mem_mib():.0f}MiB")
    return prm, cfg, tgt, vol0, x, st, Fp, rec, {"v_mean": float(vmag.mean()),
                                                 "v_max": float(vmag.max()),
                                                 "detF_min": float(detF.min()),
                                                 "detF_med": float(np.median(detF))}


# ── B/C/D. one (T, s) combo ──────────────────────────────────────────────────
def measure_combo(T: int, s: float, x, st, Fp, vol0, lam0, mu0, prm_base: MPMParams,
                  cfg, tgt, idx, fields, log):
    N = x.shape[0]
    prm = MPMParams(dx=prm_base.dx, dt=prm_base.dt, drag=prm_base.drag, smoothing=s,
                    v_max=prm_base.v_max, eta_sym=prm_base.eta_sym,
                    eta_mode=prm_base.eta_mode, f_ext=prm_base.f_ext,
                    grid_min=prm_base.grid_min, nx=prm_base.nx, ny=prm_base.ny,
                    nz=prm_base.nz)
    spec = RolloutSpec(x0=x, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T, F0=st["F"], Fp=Fp,
                       v0=st["v"], C0=st["C"], device=DEV, vol0=vol0)
    dfc = torch.zeros(T, N, 3, 3, device=DEV, requires_grad=True)
    t0 = time.perf_counter()
    xT, FT, vT = warp_mpm_full(dfc, spec, None, None)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    R = {"T": T, "s": s, "t_fwd": t_fwd, "nbwd": 0}
    tb = [0.0]

    def A(outs, seeds):
        t1 = time.perf_counter()
        n, g = adj_norm(outs, dfc, seeds)
        torch.cuda.synchronize()
        tb[0] += time.perf_counter() - t1
        R["nbwd"] += 1
        return n, g

    # B. x-path: white -> low-passed unit covectors
    R["x_band_gain"] = {str(k): A(xT, fields["x_unit"][k])[0] for k in BANDS}
    # translation (DC) covectors, one per axis
    tr = []
    for ax in range(3):
        e = torch.zeros(N, 3, device=DEV)
        e[:, ax] = 1.0 / np.sqrt(N)
        tr.append(A(xT, e)[0])
    R["x_translation_gain"] = tr
    # real covectors at THIS combo's x_T
    lr = d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k,
                  cfg.w_hole, cfg.w_spray)
    lv = d_vol(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
    lk = vT.pow(2).sum(1).mean()
    g_r = torch.autograd.grad(lr, xT, retain_graph=True)[0].detach()
    g_p = torch.autograd.grad(lv + cfg.w_kin * lk, xT, retain_graph=True)[0].detach()
    g_pv = torch.autograd.grad(cfg.w_kin * lk, vT, retain_graph=True)[0].detach()
    R["d_render"], R["d_vol"], R["kin"] = float(lr), float(lv), float(lk)
    R["g_r_norm"], R["g_p_norm"], R["g_pv_norm"] = float(g_r.norm()), float(g_p.norm()), float(g_pv.norm())
    R["cos_x_rp"] = float((g_r * g_p).sum() / (g_r.norm() * g_p.norm()).clamp_min(1e-30))
    nr, Jr = A(xT, unit(g_r))
    npx, Jp = A(xT, unit(g_p))
    gpv_all = torch.cat([g_p.flatten(), g_pv.flatten()]).norm()
    npv, Jpv = A((xT, vT), (g_p / gpv_all, g_pv / gpv_all))
    R["render_gain"], R["phys_gain"], R["phys_xv_gain"] = nr, npx, npv
    R["cos_dfc_rp"] = float((Jr * Jp).sum() / (Jr.norm() * Jp.norm()).clamp_min(1e-30))
    R["cos_dfc_r_pxv"] = float((Jr * Jpv).sum() / (Jr.norm() * Jpv.norm()).clamp_min(1e-30))
    # balancer-style lambda this window would get (raw ratio, alpha=0.5, no EMA; the
    # real gp also carries w_ctrl/w_jvol/box terms — indicative only)
    R["lambda_est"] = 0.5 * (npv * float(gpv_all)) / max(nr * float(g_r.norm()), 1e-30)
    # per-particle sparsity of the covectors (how many particles carry 90% of energy)
    for name, g in (("g_r", g_r), ("g_p", g_p)):
        e = g.pow(2).sum(1)
        cs = torch.cumsum(torch.sort(e, descending=True)[0], 0) / e.sum().clamp_min(1e-30)
        R[f"{name}_n90"] = int((cs < 0.9).sum()) + 1
    del Jr, Jp, Jpv

    # C. band decomposition of the real covectors (successive differences; NOT
    # orthogonal, so E need not sum to 1) + the monotone low-pass curve: energy kept
    # and adjoint gain of S^k g — what render_gs_iters-style pre-smoothing would do
    for name, g in (("render", g_r), ("phys", g_p)):
        bands = band_split(g, idx)
        e_tot = float(g.pow(2).sum())
        rows = {}
        for key, b in bands.items():
            bn = float(b.norm())
            gain = A(xT, b / max(bn, 1e-30))[0] if bn > 0 else 0.0
            rows[key] = {"E": bn * bn / e_tot, "gain": gain,
                         "out2": (gain * bn) ** 2}
        tot_out = sum(r["out2"] for r in rows.values())
        for r in rows.values():
            r["share"] = r["out2"] / max(tot_out, 1e-30)
        R[f"{name}_bands"] = rows
        R[f"{name}_band_E_sum"] = sum(r["E"] for r in rows.values())
        R[f"{name}_sum_out2_over_direct"] = tot_out / max(
            (R["render_gain" if name == "render" else "phys_gain"] * float(g.norm())) ** 2,
            1e-30)
        lp = {}
        for k, gk in levels(g, idx).items():
            gn = float(gk.norm())
            lp[str(k)] = {"E_lp": gn * gn / e_tot, "gain_lp": A(xT, gk / max(gn, 1e-30))[0],
                          "cos_lp": float((gk * g).sum() / max(gn * float(g.norm()), 1e-30))}
        R[f"{name}_lowpass"] = lp

    # D. F-path bands (+ DC seed = identity on every particle)
    R["F_band_gain"] = {str(k): A(FT, fields["F_unit"][k])[0] for k in BANDS}
    eyeF = torch.eye(3, device=DEV).reshape(1, 9).repeat(N, 1)
    R["F_identity_gain"] = A(FT, unit(eyeF))[0]
    R["F_render_like_gain"] = A(FT, unit(g_r.repeat(1, 3)))[0]   # render-sparse support on F

    R["t_bwd_total"], R["t_bwd_mean"] = tb[0], tb[0] / max(R["nbwd"], 1)
    R["gpu_mib"] = gpu_mem_mib()
    R["torch_max_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    log(f"[combo T={T} s={s}] fwd {t_fwd:.2f}s, {R['nbwd']} bwd @ {R['t_bwd_mean']:.2f}s, "
        f"gpu={R['gpu_mib']:.0f}MiB  x-gain k0={R['x_band_gain']['0']:.4g} "
        f"k32={R['x_band_gain']['32']:.4g} trans={max(tr):.3g} "
        f"render={nr:.4g} phys={npx:.4g} cos_x={R['cos_x_rp']:.3f} cos_dfc={R['cos_dfc_rp']:.3f}")
    del xT, FT, vT, dfc, g_r, g_p, g_pv, lr, lv, lk
    gc.collect()
    torch.cuda.empty_cache()
    return R


# ── tables ───────────────────────────────────────────────────────────────────
def print_tables(res: dict, log):
    combos = res["combos"]
    keys = [(c["T"], c["s"]) for c in combos]
    hdr = "| band k | r_rms (wu) | " + " | ".join(f"T={T},s={s}" for T, s in keys) + " |"
    ref = str(REF)
    log(f"\n### x-path gain, relative to the k={REF} band (1.0 = no attenuation vs smoothest)")
    log(hdr)
    log("|" + "---|" * (len(keys) + 2))
    for k in BANDS:
        row = [f"{c['x_band_gain'][str(k)] / max(c['x_band_gain'][ref], 1e-30):.3f}" for c in combos]
        log(f"| {k} | {res['band_radius'][str(k)]:.3f} | " + " | ".join(row) + " |")
    log("| translation | inf | " + " | ".join(
        f"{max(c['x_translation_gain']) / max(c['x_band_gain'][ref], 1e-30):.2e}" for c in combos) + " |")
    log("| render g_r | - | " + " | ".join(
        f"{c['render_gain'] / max(c['x_band_gain'][ref], 1e-30):.3f}" for c in combos) + " |")
    log("| phys g_p | - | " + " | ".join(
        f"{c['phys_gain'] / max(c['x_band_gain'][ref], 1e-30):.3f}" for c in combos) + " |")
    log(f"\n### F-path gain, relative to the k={REF} band")
    log(hdr)
    log("|" + "---|" * (len(keys) + 2))
    for k in BANDS:
        row = [f"{c['F_band_gain'][str(k)] / max(c['F_band_gain'][ref], 1e-30):.3f}" for c in combos]
        log(f"| {k} | {res['band_radius'][str(k)]:.3f} | " + " | ".join(row) + " |")
    log("\n### absolute gains ||J^T g|| for unit g")
    log(hdr)
    log("|" + "---|" * (len(keys) + 2))
    for k in BANDS:
        log(f"| x k={k} | {res['band_radius'][str(k)]:.3f} | " + " | ".join(
            f"{c['x_band_gain'][str(k)]:.4g}" for c in combos) + " |")
    log("| x translation | inf | " + " | ".join(f"{max(c['x_translation_gain']):.3g}" for c in combos) + " |")
    log("| x render | - | " + " | ".join(f"{c['render_gain']:.4g}" for c in combos) + " |")
    log("| x phys | - | " + " | ".join(f"{c['phys_gain']:.4g}" for c in combos) + " |")
    log("| x+v phys | - | " + " | ".join(f"{c['phys_xv_gain']:.4g}" for c in combos) + " |")
    for k in BANDS:
        log(f"| F k={k} | {res['band_radius'][str(k)]:.3f} | " + " | ".join(
            f"{c['F_band_gain'][str(k)]:.4g}" for c in combos) + " |")
    log("| F identity | inf | " + " | ".join(f"{c['F_identity_gain']:.4g}" for c in combos) + " |")
    log("\n### covector alignment / lambda")
    log("| combo | cos_x(g_r,g_p) | cos_dFc(J^T g_r, J^T g_p) | cos_dFc(r, p_xv) | lambda_est | |g_r| | |g_p| | n90 g_r | n90 g_p |")
    log("|---|---|---|---|---|---|---|---|---|")
    for c in combos:
        log(f"| T={c['T']},s={c['s']} | {c['cos_x_rp']:.3f} | {c['cos_dfc_rp']:.3f} | "
            f"{c['cos_dfc_r_pxv']:.3f} | {c['lambda_est']:.3g} | {c['g_r_norm']:.3g} | "
            f"{c['g_p_norm']:.3g} | {c['g_r_n90']} | {c['g_p_n90']} |")
    for c in combos:
        log(f"\n### band decomposition at T={c['T']}, s={c['s']}  (E = energy fraction, "
            f"gain = ||J^T b||/||b||, share = fraction of transmitted energy)")
        log("| band | render E | render gain | render share | phys E | phys gain | phys share |")
        log("|---|---|---|---|---|---|---|")
        for key in c["render_bands"]:
            r, p = c["render_bands"][key], c["phys_bands"][key]
            log(f"| {key} | {r['E']:.3f} | {r['gain']:.4g} | {r['share']:.3f} | "
                f"{p['E']:.3f} | {p['gain']:.4g} | {p['share']:.3f} |")
        log(f"| sum E | {c['render_band_E_sum']:.3f} | | | {c['phys_band_E_sum']:.3f} | | |")
        log(f"\n### low-pass curve at T={c['T']}, s={c['s']}: S^k g (E_lp = energy kept, "
            f"gain_lp = ||J^T unit(S^k g)||, cos = cos(S^k g, g))")
        log("| k | r_rms | render E_lp | render gain_lp | render cos | phys E_lp | phys gain_lp | phys cos |")
        log("|---|---|---|---|---|---|---|---|")
        for k in BANDS:
            r, p = c["render_lowpass"][str(k)], c["phys_lowpass"][str(k)]
            log(f"| {k} | {res['band_radius'][str(k)]:.3f} | {r['E_lp']:.4f} | {r['gain_lp']:.4g} | "
                f"{r['cos_lp']:.3f} | {p['E_lp']:.4f} | {p['gain_lp']:.4g} | {p['cos_lp']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--commits", type=int, default=4)
    ap.add_argument("--T", type=str, default="1,5,20")
    ap.add_argument("--s", type=str, default="0.955,0.0,1.0")
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="/tmp/pm31_tf_results.json")
    args = ap.parse_args()
    Ts = [int(t) for t in args.T.split(",")]
    Ss = [float(t) for t in args.s.split(",")]

    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t_start = time.perf_counter()
    prm, cfg, tgt, vol0, x, st, Fp, warm_rec, warm_stats = warm_state(args.n, args.commits, log)
    lam0, mu0 = lame(cfg.young, cfg.poisson)
    N = x.shape[0]

    # kNN topology on the warm state (window-start positions) — shared by every combo so
    # the band fields are identical across the (T, s) sweep
    idx = knn_index(x, args.knn)
    dnn = cKDTree(x).query(x, k=args.knn + 1, workers=-1)[0]
    x_t = torch.as_tensor(x, device=DEV)
    radii = band_radii(x_t, idx)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    wx = torch.randn(N, 3, device=DEV, generator=gen)
    wF = torch.randn(N, 9, device=DEV, generator=gen)
    fields = {"x_unit": {k: unit(v) for k, v in levels(wx, idx).items()},
              "F_unit": {k: unit(v) for k, v in levels(wF, idx).items()}}
    res = {"n": N, "commits": args.commits, "knn": args.knn, "seed": args.seed,
           "dx": prm.dx, "dt": prm.dt, "grid": [prm.nx, prm.ny, prm.nz],
           "loss_res": cfg.loss_res, "render_res": cfg.render_res, "views": len(tgt.views),
           "extent": tgt.extent, "nn_spacing_med": float(np.median(dnn[:, 1])),
           "knn_radius_med": float(np.median(dnn[:, -1])),
           "band_radius": {str(k): v for k, v in radii.items()},
           "warm": warm_rec, "warm_state": warm_stats, "combos": []}
    log(f"[bands] NN spacing med={res['nn_spacing_med']:.4f} wu, {args.knn}-NN radius med="
        f"{res['knn_radius_med']:.4f} wu, dx={prm.dx}; band rms radius (wu): " +
        ", ".join(f"k{k}={r:.3f}" for k, r in radii.items()))

    for T in Ts:
        for s in Ss:
            R = measure_combo(T, s, x, st, Fp, vol0, lam0, mu0, prm, cfg, tgt, idx, fields, log)
            res["combos"].append(R)
            res["elapsed"] = time.perf_counter() - t_start
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
    print_tables(res, log)
    res["elapsed"] = time.perf_counter() - t_start
    res["gpu_mib_end"] = gpu_mem_mib()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    log(f"[done] {res['elapsed']:.0f}s total, results -> {args.out}")


if __name__ == "__main__":
    main()
