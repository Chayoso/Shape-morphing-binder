"""Oscillation triage probe (docs/oscillation_triage.md): decide, from a finished run's
archive, which of the pre-registered drivers an observed vibration belongs to:
  A  VOLUME     J = det F breathing (the volumetric spring lambda/2 (J-1)^2),
  B  STIFFNESS  elastic ringing at the body's elastic period / a CFL violation,
  C  CONTROL    window-boundary stop-and-go from the per-window terminal-rest objective,
  D  none       sub-spacing noise (Addendum 7 of docs/oscillation.md decides VISIBLE).
Measurement only: pre-registered decision rules, no fix; nothing here is imported by the
pipeline. numpy + scipy.spatial only; matplotlib is imported lazily behind --png.

usage: python scripts/probes/oscillation_triage.py --npz RUN_<arm>.npz [--json RUN.json --arm ARM]
         [--T 20] [--dt 1/240] [--dx 0.5] [--young 1.4e5 --poisson 0.2]
         [--out triage.json] [--png triage.png] [--tail 40]

Archive contract (scripts/pipeline_run.py, physmorph/pipeline/runner.py): frames[0] is the
source; every ACCEPTED commit appends its T rollout states (fr[1:-1], then the promoted end
state) and records frame_end = len(frames); held / null commits append ONE duplicated frame
and carry no frame_end; outer-rejected candidates roll the archive back and append nothing.
Only `frames` and T are required — every other input degrades to "not measured".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULTS = dict(dt=1.0 / 240.0, dx=0.5, young=1.4e5, poisson=0.2, smoothing=0.955,
                drag=0.9, grid_min=(-16.0, -16.0, -16.0))
# Pre-registered thresholds (docs/oscillation_triage.md "Decision rules"). Do not tune
# them on the run being triaged.
RULES = dict(visible_frac=0.01, visible_sp=0.5, sag=0.5, modulation=2.0, jump_hi=2.0, jump_lo=0.5,
             tortuosity=1.5, j_rms=0.05,
             # lock band 6 % (REFUTE-2 F2: at 15 % the F-smoothing time dt/(1-s) = 22.2
             # steps and the elastic harmonics tau_e/6, tau_e/7 fell inside the band; the
             # measured periods are 20.00-20.03 with 0.19-step bins, so 6 % keeps them)
             lock_main=0.06, lock_div=0.10, ring_tol=0.25, cfl_b=0.3, cfl_violation=0.5,
             j_p2p=0.02, j_corr=0.5, kin_end=0.05, reversal_cos=-0.2, tail=40)


def lame(young: float, poisson: float) -> tuple[float, float]:
    """docs/method.md eq (1)."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    return float(lam), float(mu)


# ----------------------------------------------------------------------------- helpers
def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _jsonable(o):
    """numpy -> python; non-finite floats -> None (json has no inf/nan)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _fraction(s):
    """'1/240' or '0.004167' -> float."""
    if s is None:
        return None
    s = str(s)
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def _nanstat(fn, v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) | np.isinf(v)] if v.size else v
    return float(fn(v)) if v.size and np.isfinite(v).any() else None


# ----------------------------------------------------------------------------- loading
def _guess_json(npz_path):
    """RUN_<arm>.npz -> RUN.json: try every '_' split from the right."""
    p = Path(npz_path); stem = p.stem
    for i in range(len(stem) - 1, 0, -1):
        if stem[i] == "_":
            cand = p.with_name(stem[:i] + ".json")
            if cand.exists():
                return str(cand)
    return None


def _guess_arm(npz_path, json_path, arms):
    stem = Path(npz_path).stem; pre = Path(json_path).stem
    if stem.startswith(pre + "_") and stem[len(pre) + 1:] in arms:
        return stem[len(pre) + 1:]
    return next(iter(arms), None)


def load_archive(npz_path, json_path=None, arm=None):
    """-> (arrays, history|None, provenance|None, arm_config|None, arm|None, json_used)."""
    z = np.load(npz_path)
    keys = ("src", "tgt", "frames", "deliver_n", "F_samples", "F_sample_idx")
    arrays = {k: z[k] for k in z.files if k in keys}
    history = prov = cfg = None
    jp = json_path or _guess_json(npz_path)
    if jp and Path(jp).exists():
        with open(jp) as f:
            d = json.load(f)
        prov = d.get("provenance")
        arms = d.get("arms") or {}
        if arm is None:
            arm = _guess_arm(npz_path, jp, arms)
        rec = arms.get(arm) if arm else None
        if isinstance(rec, dict):
            history, cfg = rec.get("history"), rec.get("config")
    else:
        jp = None
    return arrays, history, prov, cfg, arm, jp


def resolve_params(prov=None, cfg=None, T=None, dt=None, dx=None, young=None, poisson=None):
    """CLI > run json (provenance / arm config) > DEFAULTS; every value keeps its source
    (AGENTS.md rule 4: the discretisation travels with the numbers)."""
    prov = prov or {}; cfg = cfg or {}; mpm = prov.get("mpm") or {}

    def pick(cli, cands, default=None):
        if cli is not None:
            return cli, "cli"
        for tag, d, key in cands:
            if isinstance(d, dict) and d.get(key) is not None:
                return d[key], tag
        return default, "default"
    p, src = {}, {}
    p["T"], src["T"] = pick(T, [("json.provenance", prov, "T"), ("json.config", cfg, "T")])
    p["dt"], src["dt"] = pick(dt, [("json.mpm", mpm, "dt")], DEFAULTS["dt"])
    p["dx"], src["dx"] = pick(dx, [("json.mpm", mpm, "dx")], DEFAULTS["dx"])
    p["smoothing"], src["smoothing"] = pick(None, [("json.mpm", mpm, "smoothing")], DEFAULTS["smoothing"])
    p["drag"], src["drag"] = pick(None, [("json.mpm", mpm, "drag")], DEFAULTS["drag"])
    p["young"], src["young"] = pick(young, [("json.config", cfg, "young"), ("json.provenance", prov, "young")], DEFAULTS["young"])
    p["poisson"], src["poisson"] = pick(poisson, [("json.config", cfg, "poisson"), ("json.provenance", prov, "poisson")], DEFAULTS["poisson"])
    p["grid_min"], src["grid_min"] = pick(None, [("json.mpm", mpm, "grid_min")], DEFAULTS["grid_min"])
    p["w_kin"], src["w_kin"] = pick(None, [("json.provenance", prov, "w_kin"), ("json.config", cfg, "w_kin")])
    for k in ("dt", "dx", "smoothing", "drag", "young", "poisson"):
        if p[k] is not None:
            p[k] = float(p[k])
    if p["T"] is not None:
        p["T"] = int(p["T"])
    p["source"] = src
    return p


# ----------------------------------------------------------------------------- 1. speed
def speed_series(frames, dn, dt, chunk=128):
    """s_t = mean_p ||x_{t+1}-x_t|| / dt for t < dn-1 (chunked: frames may be GBs), plus
    the exact-duplicate mask (held / null commits archive one copied frame)."""
    n = int(min(dn, len(frames))); m = max(n - 1, 0)
    s = np.zeros(m); zero = np.zeros(m, bool)
    for i in range(0, m, chunk):
        j = min(i + chunk, m)
        d = np.asarray(frames[i + 1:j + 1], np.float32) - np.asarray(frames[i:j], np.float32)
        zero[i:j] = ~np.any(d != 0, axis=(1, 2))
        s[i:j] = np.linalg.norm(d.astype(np.float64), axis=2).mean(1) / dt
    return s, zero


def accepted_records(history):
    if not history:
        return []
    acc = [r for r in history if isinstance(r, dict) and r.get("frame_end")
           and not r.get("null_commit") and not r.get("held")]
    return sorted(acc, key=lambda r: int(r["frame_end"]))


def infer_T(acc):
    """Mode of consecutive frame_end differences (held frames add +1 occasionally)."""
    fe = np.array([int(r["frame_end"]) for r in acc])
    d = np.diff(fe); d = d[d > 0]
    if d.size == 0:
        return None
    v, c = np.unique(d, return_counts=True)
    return int(v[np.argmax(c)])


def segment(s, zero, acc, T):
    """Window step ranges. With history: the T transitions ending at frame_end-1 (the
    first one starts at the previous end state or its held copy — same positions). Without
    history: drop duplicate steps, then every T steps. Returns the compressed speed series,
    the held-step mask (raw), windows in compressed indices, and the keep mask (raw)."""
    n = len(s)
    if acc:
        raw = []
        for r in acc:
            t1 = int(r["frame_end"]) - 1; t0 = t1 - T
            if t0 >= 0 and t1 <= n:
                raw.append((t0, t1))
        inwin = np.zeros(n, bool)
        for a, b in raw:
            inwin[a:b] = True
        held = zero & ~inwin              # a zero step INSIDE a window is real (kept)
    else:
        held = zero.copy(); raw = None
    keep = ~held
    pos = np.cumsum(keep) - 1
    s_c = s[keep]
    if raw is not None:
        win = [(int(pos[a]), int(pos[b - 1]) + 1) for a, b in raw]
    else:
        win = [(w * T, (w + 1) * T) for w in range(len(s_c) // T)]
    return s_c, held, win, keep


def stop_and_go(s_c, win, acc_ok=None, dt=None):
    """acc_ok/dt: accepted records aligned with `win` (same filter as segment) — enables
    the TORTUOSITY statistic path/net per window (REFUTE-2 CNR-1): path = sum_t s_t dt,
    net = history.move (mean per-particle |x_end - x_start|); 1 = straight, >>1 = push
    and return. Convention-free, unlike the spectral power fraction (F4)."""
    sag, jump, endfrac = [], [], []
    for k, (a, b) in enumerate(win):
        seg = s_c[a:b]
        if seg.size == 0:
            continue
        m = float(seg.max())
        sag.append((m - seg[-1]) / m if m > 0 else np.nan)
        endfrac.append((seg[-1] / m) ** 2 if m > 0 else np.nan)   # kin_end ~ s_end^2 / s_max^2
        if k + 1 < len(win):
            a2 = win[k + 1][0]
            jump.append(s_c[a2] / seg[-1] if seg[-1] > 0 else (np.inf if s_c[a2] > 0 else np.nan))
    # intra-window MODULATION: max/min speed inside a window. A window-locked limit
    # cycle whose turning point sits mid-window (hyde06 2026-09-15: 0.47 -> 0.10 -> 0.45,
    # continuous across the boundary) has sag ~ 0 and jump ~ 1 yet modulation ~ 4.
    mod = []
    for (a, b) in win:
        seg = s_c[a:b]
        if seg.size >= 3 and seg.min() > 0:
            mod.append(float(seg.max() / seg.min()))
    sag = np.array(sag); jump = np.array(jump); endfrac = np.array(endfrac); mod = np.array(mod)
    tort = []
    if acc_ok is not None and dt and len(acc_ok) == len(win):
        for r, (a, b) in zip(acc_ok, win):
            net = r.get("move")
            if net and net > 0 and b > a:
                tort.append(float(s_c[a:b].sum() * dt / net))
    tort = np.array(tort)
    return dict(
        n_windows=int(len(sag)),
        tortuosity_median=_nanstat(np.nanmedian, tort) if tort.size else None,
        modulation_median=_nanstat(np.nanmedian, mod) if mod.size else None,
        modulation_frac_gt_rule=(_nanstat(np.nanmean, (mod > RULES["modulation"]).astype(float))
                                 if mod.size else None),
        sag_median=_nanstat(np.nanmedian, sag),
        sag_frac_gt_half=_nanstat(np.nanmean, (sag > RULES["sag"]).astype(float)) if sag.size else None,
        jump_median=_nanstat(np.nanmedian, jump),
        jump_p10=_nanstat(lambda v: np.nanpercentile(v, 10), jump),
        jump_p90=_nanstat(lambda v: np.nanpercentile(v, 90), jump),
        kin_end_frac=_nanstat(np.nanmean, (endfrac < RULES["kin_end"]).astype(float)) if endfrac.size else None,
    )


def dominant_period(x, top=3):
    """Linear detrend + Hann + rFFT. Period of the strongest local maximum (refined on an
    8x zero-padded spectrum), power fraction = peak +-1 raw bin / total above 2 cycles."""
    x = np.asarray(x, float); n = len(x)
    empty = dict(period=None, power_frac=None, power_frac_all=None, n=int(n), peaks=[])
    if n < 8 or not np.isfinite(x).all():
        return empty
    t = np.arange(n)
    y = x - np.polyval(np.polyfit(t, x, 1), t)
    if np.allclose(y, 0):
        return empty
    y = y * np.hanning(n)
    P = np.abs(np.fft.rfft(y)) ** 2; fr = np.fft.rfftfreq(n)
    valid = fr >= 2.0 / n                    # at least two cycles in the record
    if not valid.any() or P[valid].sum() <= 0:
        return empty
    tot = P[valid].sum()
    tot_all = float(P[fr > 0].sum()) or float(tot)   # REFUTE-2 F4: second convention
    loc = [k for k in range(1, len(P) - 1) if valid[k] and P[k] >= P[k - 1] and P[k] >= P[k + 1]]
    if not loc:
        loc = [int(np.argmax(np.where(valid, P, -1.0)))]
    loc.sort(key=lambda k: -P[k])
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    Pz = np.abs(np.fft.rfft(y, nfft)) ** 2; frz = np.fft.rfftfreq(nfft)
    peaks = []
    for k in loc[:top]:
        sel = np.where((frz >= fr[k] - 1.0 / n) & (frz <= fr[k] + 1.0 / n) & (frz > 0))[0]
        kz = sel[np.argmax(Pz[sel])]
        peaks.append(dict(period=float(1.0 / frz[kz]),
                          power_frac=float(P[max(k - 1, 0):k + 2].sum() / tot),
                          power_frac_all=float(P[max(k - 1, 0):k + 2].sum() / tot_all)))
    return dict(period=peaks[0]["period"], power_frac=peaks[0]["power_frac"],
                power_frac_all=peaks[0]["power_frac_all"], n=int(n), peaks=peaks)


def window_lock(P, T, tol_main=RULES["lock_main"], tol_div=RULES["lock_div"]):
    """|P-T| <= 15% T, or P within 10% of T/m for an integer m >= 2 (T/m >= 2 substeps)."""
    if P is None or not T:
        return False
    if abs(P - T) <= tol_main * T:
        return True
    m = 2
    while T / m >= 2.0:
        if abs(P - T / m) <= tol_div * (T / m):
            return True
        m += 1
    return False


# ----------------------------------------------------------------------------- 2. volume
def speed_at_frames(s, keep, idx):
    """Speed at a frame instant = mean of the adjacent kept steps (NaN if none)."""
    n = len(s); out = np.full(len(idx), np.nan)
    for q, i in enumerate(idx):
        c = [s[j] for j in (i - 1, i) if 0 <= j < n and keep[j]]
        if c:
            out[q] = float(np.mean(c))
    return out


def nn_volume_proxy(frames, idx, k=8, max_pts=4000, max_samples=400, seed=0):
    """CIC-free volume proxy: median k-NN spacing cubed on a fixed random subset."""
    N = frames.shape[1]; rng = np.random.default_rng(seed)
    sub = rng.choice(N, max_pts, replace=False) if N > max_pts else np.arange(N)
    idx = np.asarray(idx, int)
    if len(idx) > max_samples:
        idx = idx[np.linspace(0, len(idx) - 1, max_samples).astype(int)]
    kk = max(1, min(k, len(sub) - 1))
    v = []
    for i in idx:
        x = np.asarray(frames[i], np.float64)[sub]
        d = cKDTree(x).query(x, k=kk + 1)[0][:, 1:]
        v.append(float(np.median(d.mean(1)) ** 3))
    return np.array(v), idx


def volume_series(F_samples, F_sample_idx, frames, dn, s, keep, seed=0):
    out = dict(n_samples=0, meanJ_p2p=None, corr_J_speed=None)
    if F_samples is None or F_sample_idx is None:
        return out
    idx = np.asarray(F_sample_idx, int); sel = idx < dn
    idx = idx[sel]
    if idx.size == 0:
        return out
    Fs = np.asarray(F_samples)[sel].astype(np.float64)
    K = len(idx); out["n_samples"] = int(K)
    J = np.linalg.det(Fs)                                    # (K, N)
    meanJ = J.mean(1)
    run = np.cumsum(J, 0) / np.arange(1, K + 1)[:, None]     # per-particle running mean
    rms_run = np.sqrt(((J - run) ** 2).mean(1))
    t = np.arange(K)
    Jd = meanJ - np.polyval(np.polyfit(t, meanJ, 1), t) if K >= 3 else meanJ - meanJ.mean()
    sp_at = speed_at_frames(s, keep, idx)
    c_signed = _corr(Jd, sp_at); c_abs = _corr(np.abs(Jd), sp_at)
    cands = [c for c in (c_signed, c_abs) if c is not None]
    corr = max(cands, key=abs) if cands else None
    proxy, pidx = nn_volume_proxy(frames, idx, seed=seed)
    prel = proxy / proxy[0] if proxy.size and proxy[0] > 0 else proxy
    tp = np.arange(len(prel))
    pd = prel - np.polyval(np.polyfit(tp, prel, 1), tp) if len(prel) >= 3 else prel - prel.mean()
    Jp = np.interp(pidx, idx, meanJ) if len(pidx) != len(idx) else meanJ
    out.update(
        sample_idx=idx.tolist(), meanJ_series=meanJ.tolist(), rms_running_series=rms_run.tolist(),
        speed_at_samples=sp_at.tolist(),
        meanJ_first=float(meanJ[0]), meanJ_last=float(meanJ[-1]),
        meanJ_min=float(meanJ.min()), meanJ_max=float(meanJ.max()),
        Jmin_particle=float(J.min()), J_std_last=float(J[-1].std()),
        meanJ_p2p=float(Jd.max() - Jd.min()), rms_running_max=float(rms_run.max()),
        corr_J_speed_signed=c_signed, corr_absJ_speed=c_abs, corr_J_speed=corr,
        proxy_sample_idx=pidx.tolist(), proxy_rel_series=prel.tolist(),
        proxy_rel_p2p=float(pd.max() - pd.min()) if pd.size else None,
        corr_J_proxy=_corr(Jp, prel),
    )
    return out


# ----------------------------------------------------------------------------- 3. stiffness
def stiffness_scales(src, lam, mu, dt, dx, grid_min=None, mass=1.0):
    """c = sqrt((lam+2mu)/rho), rho = N m / V_cic, V_cic = occupied CIC nodes x dx^3 (an
    over-count by one surface layer; V_bbox reported alongside), CFL = c dt / dx,
    tau_e = 2 L / c with L = source bbox diagonal."""
    x = np.asarray(src, np.float64); N = len(x)
    ext = x.max(0) - x.min(0); L = float(np.linalg.norm(ext))
    g0 = np.asarray(grid_min, float) if grid_min is not None else x.min(0) - 2 * dx
    base = np.floor((x - g0) / dx).astype(np.int64)
    offs = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    nodes = (base[:, None, :] + offs[None]).reshape(-1, 3) - base.min(0)   # CIC 2^3 support
    key = (nodes[:, 0] * (1 << 21) + nodes[:, 1]) * (1 << 21) + nodes[:, 2]
    n_occ = int(np.unique(key).size)
    V_cic = n_occ * dx ** 3; V_bbox = float(np.prod(ext))
    rho = N * mass / V_cic
    c = math.sqrt((lam + 2.0 * mu) / rho)
    tau = 2.0 * L / c
    return dict(N=int(N), L=L, n_occupied=n_occ, V_cic=float(V_cic), V_bbox=V_bbox,
                rho=float(rho), c=float(c), cfl=float(c * dt / dx), tau_e=float(tau),
                tau_e_steps=float(tau / dt), lam=float(lam), mu=float(mu), dt=float(dt), dx=float(dx))


def ringing(P, tau_steps, window_locked, tol=RULES["ring_tol"]):
    if P is None or not tau_steps:
        return False
    return (not window_locked) and any(abs(P - q) <= tol * q for q in (tau_steps, tau_steps / 2.0))


# ----------------------------------------------------------------------------- 4. control
def control_stats(history, acc):
    out = dict(has_history=bool(history))
    if not history:
        return out
    out.update(n_records=len(history), n_accepted=len(acc),
               n_null=sum(1 for r in history if isinstance(r, dict) and r.get("null_commit")),
               n_held=sum(1 for r in history if isinstance(r, dict) and r.get("held")),
               n_outer_rejected=sum(1 for r in history if isinstance(r, dict) and r.get("outer_rejected")))
    rc = np.array([r["reversal_cos"] for r in acc if r.get("reversal_cos") is not None], float)
    out["reversal_n"] = int(rc.size)
    out["reversal_frac_lt_thr"] = float(np.mean(rc < RULES["reversal_cos"])) if rc.size else None
    out["reversal_cos_median"] = float(np.median(rc)) if rc.size else None
    pairs = [(r["lambda"], r["d_vol"]) for r in acc
             if r.get("lambda") is not None and r.get("d_vol") is not None]
    out["corr_dlambda_ddvol"] = None
    if len(pairs) >= 4:
        lam, dv = np.array(pairs, float).T
        out["corr_dlambda_ddvol"] = _corr(np.diff(lam), np.diff(dv))
    dfc = np.array([r["dfc_absmax"] for r in acc if r.get("dfc_absmax") is not None], float)
    out["dfc_absmax"] = (dict(n=int(dfc.size), median=float(np.median(dfc)), max=float(dfc.max()),
                              last=float(dfc[-1]), frac_zero=float(np.mean(dfc < 1e-9)))
                         if dfc.size else None)
    mv = np.array([r["move"] for r in acc if r.get("move") is not None], float)
    out["move_sign_flip_rate"] = None
    if mv.size >= 3:
        d = np.diff(mv); d = d[d != 0]
        if d.size >= 2:
            out["move_sign_flip_rate"] = float(np.mean(np.sign(d[1:]) != np.sign(d[:-1])))
    out["move_median"] = float(np.median(mv)) if mv.size else None
    kin = np.array([r["kin"] for r in acc if r.get("kin") is not None], float)
    out["kin_median"] = float(np.median(kin)) if kin.size else None
    out["kin_last"] = float(kin[-1]) if kin.size else None
    dv = np.array([r["d_vol"] for r in acc if r.get("d_vol") is not None], float)
    out["d_vol_first"] = float(dv[0]) if dv.size else None
    out["d_vol_last"] = float(dv[-1]) if dv.size else None
    return out


# ----------------------------------------------------------------------------- 5. visibility
def target_spacing(tgt, src=None):
    p = tgt if tgt is not None and len(tgt) > 1 else src
    if p is None or len(p) < 2:
        return None
    p = np.asarray(p, np.float64)
    return float(np.median(cKDTree(p).query(p, k=2)[0][:, 1]))


def commit_end_frames(acc, T, dn, dup_frame):
    """Frame indices of accepted commit end states (delivered slice only)."""
    if acc:
        return [int(r["frame_end"]) - 1 for r in acc if int(r["frame_end"]) <= dn]
    kept = np.where(~dup_frame[:dn])[0]
    return [int(kept[j]) for j in range(T, len(kept), T)]


def visibility(frames, dn, ends, sp, tail=RULES["tail"]):
    """docs/oscillation.md Addendum 7: per-particle peak-to-peak excursion about the linear
    drift over the last `tail` accepted commits, and per-commit increments, in target
    spacings (sp)."""
    out = dict(n_tail_commits=max(len(ends) - 1, 0), sp=sp, frac_excursion_gt_half_sp=None,
               frac_move_gt_half_sp=None)
    if len(ends) < 2 or not sp:
        return out
    sel = ends[-(tail + 1):]
    X = np.asarray(frames[sel], np.float64)                 # (n_t, N, 3)
    u = (np.arange(len(X)) / max(len(X) - 1, 1))[:, None, None]
    drift = X[0][None] + (X[-1] - X[0])[None] * u
    p2p = np.linalg.norm(X - drift, axis=2).max(0) / sp     # (N,)
    inc = np.linalg.norm(np.diff(X, axis=0), axis=2)        # (n_t-1, N)
    un = np.diff(X, axis=0) / np.maximum(inc, 1e-12)[..., None]
    cos = (un[1:] * un[:-1]).sum(2) if len(X) > 2 else np.zeros((0, X.shape[1]))
    inc_sp = inc / sp
    hist, edges = np.histogram(np.clip(p2p, 0, 2.0), bins=20, range=(0.0, 2.0))
    out.update(
        n_tail_commits=int(len(X) - 1), tail_frames=[int(i) for i in sel],
        excursion_median=float(np.median(p2p)), excursion_p99=float(np.percentile(p2p, 99)),
        excursion_max=float(p2p.max()),
        frac_excursion_gt_half_sp=float(np.mean(p2p > RULES["visible_sp"])),
        excursion_hist=hist.tolist(), excursion_hist_edges=edges.tolist(),
        move_median=float(np.median(inc_sp)), move_p99=float(np.percentile(inc_sp, 99)),
        frac_move_gt_half_sp=float(np.mean(inc_sp > RULES["visible_sp"])),
        rev_cos_median=float(np.median(cos)) if cos.size else None,
        frac_reversing=float(np.mean(cos < 0)) if cos.size else None,
    )
    return out


# ----------------------------------------------------------------------------- decision
def decide(speed, vol, stiff, vis):
    R = RULES
    fe = vis.get("frac_excursion_gt_half_sp"); fm = vis.get("frac_move_gt_half_sp")
    visible = bool((fe is not None and fe > R["visible_frac"]) or (fm is not None and fm > R["visible_frac"]))
    wl = bool(speed.get("window_locked")); sag = speed.get("sag_median"); jm = speed.get("jump_median")
    md = speed.get("modulation_median"); tq = speed.get("tortuosity_median")
    # WINDOW-locked driver (REFUTE-2 F1: the rule reads the speed series only, so the
    # label says what is measured — locked to the optimisation window — not "control";
    # the T-variation and --assim 0 discriminators attribute it, docs/oscillation_triage.md)
    C = wl and ((sag is not None and sag > R["sag"])
                or (jm is not None and (jm > R["jump_hi"] or jm < R["jump_lo"]))
                or (md is not None and md > R["modulation"])
                # path/net is noise-dominated for sub-spacing motion: gate on visibility
                or (tq is not None and visible and tq > R["tortuosity"]))
    cfl = stiff.get("cfl")
    B = bool(stiff.get("stiffness_ringing")) and cfl is not None and cfl > R["cfl_b"]
    cfl_violation = cfl is not None and cfl > R["cfl_violation"]
    p2p = vol.get("meanJ_p2p"); cj = vol.get("corr_J_speed")
    A = ((p2p is not None and p2p > R["j_p2p"]) and (cj is not None and abs(cj) > R["j_corr"])) \
        or (vol.get("rms_running_max") is not None and vol["rms_running_max"] > R["j_rms"])  # F18
    drivers = [n for n, f in (("A_volume", A), ("B_stiffness", B), ("C_window", C)) if f]
    verdict = ("VISIBLE" if visible else "INVISIBLE (sub-spacing)") + ": " + \
              (", ".join(drivers) if drivers else "no identifiable driver") + \
              (" [CFL violation]" if cfl_violation else "")
    return dict(visible=visible, driver_A=bool(A), driver_B=bool(B), driver_C=bool(C),
                cfl_violation=bool(cfl_violation), drivers=drivers, verdict=verdict)


def triage(arrays, history=None, prov=None, cfg=None, T=None, dt=None, dx=None,
           young=None, poisson=None, tail=RULES["tail"], seed=0):
    """Full report (JSON-safe dict) from in-memory arrays + optional history/provenance."""
    frames = arrays["frames"]; M = int(len(frames))
    dn = int(arrays["deliver_n"]) if arrays.get("deliver_n") is not None else M
    dn = max(2, min(dn, M))
    prm = resolve_params(prov, cfg, T, dt, dx, young, poisson)
    acc = accepted_records(history)
    T_inf = infer_T(acc) if acc else None
    if prm["T"] is None:
        prm["T"], prm["source"]["T"] = T_inf, "history.frame_end"
    if prm["T"] is None:
        raise ValueError("T unknown: pass --T or a run json with provenance/history")
    Tn = int(prm["T"]); lam, mu = lame(prm["young"], prm["poisson"])

    s, zero = speed_series(frames, dn, prm["dt"])
    s_c, held, win, keep = segment(s, zero, acc, Tn)
    acc_ok = [r for r in acc if (int(r["frame_end"]) - 1 - Tn) >= 0
              and int(r["frame_end"]) - 1 <= len(s)] if acc else None
    spec = dominant_period(s_c)
    wl = window_lock(spec["period"], Tn)
    speed = dict(n_steps=int(len(s)), n_held_steps=int(held.sum()), n_used=int(len(s_c)),
                 boundaries="history.frame_end" if acc else f"every T={Tn} (no history)",
                 T_inferred=T_inf, T_mismatch=bool(T_inf is not None and T_inf != Tn),
                 s_mean=float(s_c.mean()) if s_c.size else None,
                 s_median=float(np.median(s_c)) if s_c.size else None,
                 s_max=float(s_c.max()) if s_c.size else None,
                 **stop_and_go(s_c, win, acc_ok, prm["dt"]), period=spec["period"],
                 power_frac=spec["power_frac"], power_frac_all=spec.get("power_frac_all"),
                 peaks=spec["peaks"], window_locked=wl, series=s_c.tolist(), windows=win)

    src = arrays.get("src", frames[0])
    stiff = stiffness_scales(src, lam, mu, prm["dt"], prm["dx"], prm["grid_min"])
    stiff["stiffness_ringing"] = ringing(spec["period"], stiff["tau_e_steps"], wl)
    ratio = Tn * prm["dt"] / stiff["tau_e"] if stiff["tau_e"] > 0 else None
    stiff["window_over_tau"] = ratio
    stiff["window_ends_mid_oscillation"] = bool(ratio is not None and abs((ratio % 1.0) - 0.5) < 0.15)

    vol = volume_series(arrays.get("F_samples"), arrays.get("F_sample_idx"), frames, dn, s, keep, seed)
    ctrl = control_stats(history, acc)
    sp = target_spacing(arrays.get("tgt"), src)
    ends = commit_end_frames(acc, Tn, dn, np.r_[False, held])
    states = ([0] if not ends or ends[0] != 0 else []) + ends   # frame 0 precedes commit 1
    vis = visibility(frames, dn, states, sp, tail)
    dec = decide(speed, vol, stiff, vis)
    rep = dict(provenance=prm, archive=dict(n_frames=M, deliver_n=dn, N=int(frames.shape[1]),
                                            n_F_samples=vol.get("n_samples", 0),
                                            n_accepted_records=len(acc)),
               speed=speed, volume=vol, stiffness=stiff, control=ctrl, visibility=vis,
               decision=dec, rules=dict(RULES))
    return _jsonable(rep)


# ----------------------------------------------------------------------------- reporting
def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def summarize_markdown(rep):
    p = rep["provenance"]; d = rep["decision"]; s = rep["speed"]; v = rep["volume"]
    k = rep["stiffness"]; c = rep["control"]; z = rep["visibility"]; a = rep["archive"]
    tag = (f"dt={_fmt(p['dt'], 6)} dx={_fmt(p['dx'])} T={p['T']} smoothing={_fmt(p['smoothing'])} "
           f"drag={_fmt(p['drag'])} E={_fmt(p['young'])} nu={_fmt(p['poisson'])}")
    src = ", ".join(f"{q}:{p['source'][q]}" for q in ("T", "dt", "dx", "smoothing", "young"))
    L = [f"# Oscillation triage: {d['verdict']}",
         f"provenance: {tag}  [{src}]",
         f"archive: N={a['N']} frames={a['n_frames']} delivered={a['deliver_n']} "
         f"accepted_records={a['n_accepted_records']} F_samples={a['n_F_samples']}", ""]

    def sec(title, rows):
        L.append(f"## {title}  @ {tag}"); L.append("| quantity | value |"); L.append("|---|---|")
        L.extend(f"| {q} | {_fmt(val)} |" for q, val in rows); L.append("")
    sec("1. speed / windows (driver C evidence)", [
        ("steps used / held", f"{s['n_used']} / {s['n_held_steps']}"), ("boundaries", s["boundaries"]),
        ("T inferred from frame_end", s["T_inferred"]), ("windows", s["n_windows"]),
        ("stop-and-go median", s["sag_median"]), ("frac windows sag>0.5", s["sag_frac_gt_half"]),
        ("intra-window speed modulation max/min median", s.get("modulation_median")),
        ("tortuosity path/net per window (median)", s.get("tortuosity_median")),
        ("peak power fraction, all non-DC bins (F4 second convention)", s.get("power_frac_all")),
        ("boundary jump ratio median (p10/p90)", f"{_fmt(s['jump_median'])} ({_fmt(s['jump_p10'])}/{_fmt(s['jump_p90'])})"),
        ("kin_end<5% of peak s^2 (frac windows)", s["kin_end_frac"]),
        ("dominant speed period P_s [substeps]", s["period"]), ("peak power fraction", s["power_frac"]),
        ("next peaks", ", ".join(f"{_fmt(q['period'])} ({_fmt(q['power_frac'], 2)})" for q in s["peaks"][1:]) or "n/a"),
        ("window_locked", s["window_locked"])])
    sec("2. volume (driver A evidence)", [
        ("F samples", v.get("n_samples")), ("mean J first -> last", f"{_fmt(v.get('meanJ_first'))} -> {_fmt(v.get('meanJ_last'))}"),
        ("detrended mean-J peak-to-peak", v.get("meanJ_p2p")), ("Jmin (any particle, any sample)", v.get("Jmin_particle")),
        ("max RMS(J - running mean)", v.get("rms_running_max")),
        ("corr(J~, speed) signed / corr(|J~|, speed)", f"{_fmt(v.get('corr_J_speed_signed'))} / {_fmt(v.get('corr_absJ_speed'))}"),
        ("corr_J_speed (rule input)", v.get("corr_J_speed")),
        ("8-NN spacing^3 proxy: rel p2p / corr with J", f"{_fmt(v.get('proxy_rel_p2p'))} / {_fmt(v.get('corr_J_proxy'))}")])
    sec("3. stiffness (driver B evidence)", [
        ("lambda / mu", f"{_fmt(k['lam'])} / {_fmt(k['mu'])}"), ("V_cic / V_bbox [wu^3]", f"{_fmt(k['V_cic'])} / {_fmt(k['V_bbox'])}"),
        ("rho", k["rho"]), ("c [wu/s]", k["c"]), ("CFL = c dt/dx", k["cfl"]), ("L (bbox diag)", k["L"]),
        ("tau_e [s] / [substeps]", f"{_fmt(k['tau_e'])} / {_fmt(k['tau_e_steps'])}"),
        ("window T dt / tau_e", k["window_over_tau"]), ("window ends mid-oscillation", k["window_ends_mid_oscillation"]),
        ("stiffness_ringing", k["stiffness_ringing"])])
    if c.get("has_history"):
        dfc = c.get("dfc_absmax") or {}
        sec("4. gradient / control (history)", [
            ("records: accepted / null / held / outer-rejected", f"{c['n_accepted']} / {c['n_null']} / {c['n_held']} / {c['n_outer_rejected']}"),
            ("frac reversal_cos < -0.2 (n)", f"{_fmt(c['reversal_frac_lt_thr'])} ({c['reversal_n']})"),
            ("reversal_cos median", c["reversal_cos_median"]), ("corr(d lambda, d d_vol)", c["corr_dlambda_ddvol"]),
            ("dfc_absmax median / max / last / frac zero", f"{_fmt(dfc.get('median'))} / {_fmt(dfc.get('max'))} / {_fmt(dfc.get('last'))} / {_fmt(dfc.get('frac_zero'))}"),
            ("move sign-flip rate", c["move_sign_flip_rate"]), ("kin median / last", f"{_fmt(c['kin_median'])} / {_fmt(c['kin_last'])}"),
            ("d_vol first -> last", f"{_fmt(c['d_vol_first'])} -> {_fmt(c['d_vol_last'])}")])
    else:
        L.append(f"## 4. gradient / control  @ {tag}\n\nno history (json missing) — not measured\n")
    sec(f"5. visibility (Addendum 7, last {z['n_tail_commits']} commits, sp={_fmt(z.get('sp'))})", [
        ("excursion p2p median / p99 / max [sp]", f"{_fmt(z.get('excursion_median'))} / {_fmt(z.get('excursion_p99'))} / {_fmt(z.get('excursion_max'))}"),
        ("frac particles excursion > 0.5 sp", z.get("frac_excursion_gt_half_sp")),
        ("per-commit move median / p99 [sp]", f"{_fmt(z.get('move_median'))} / {_fmt(z.get('move_p99'))}"),
        ("frac particle-commits move > 0.5 sp", z.get("frac_move_gt_half_sp")),
        ("tail rev-cos median / frac reversing", f"{_fmt(z.get('rev_cos_median'))} / {_fmt(z.get('frac_reversing'))}")])
    L.append(f"## decision  @ {tag}\n")
    L.append(f"visible={d['visible']}  A_volume={d['driver_A']}  B_stiffness={d['driver_B']}  "
             f"C_window={d['driver_C']}  cfl_violation={d['cfl_violation']}\n\n**{d['verdict']}**")
    return "\n".join(L)


def save_png(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = rep["speed"]; v = rep["volume"]; k = rep["stiffness"]; z = rep["visibility"]; p = rep["provenance"]
    y = np.array(s["series"], float)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]; a.plot(y, lw=0.6)
    for _, b in s["windows"]:
        a.axvline(b, color="k", alpha=0.12, lw=0.5)
    a.set_title(f"mean speed per substep; P_s={_fmt(s['period'])} locked={s['window_locked']}"); a.set_xlabel("substep")
    a = ax[0, 1]
    if len(y) >= 8:
        t = np.arange(len(y)); yd = (y - np.polyval(np.polyfit(t, y, 1), t)) * np.hanning(len(y))
        P = np.abs(np.fft.rfft(yd)) ** 2; fr = np.fft.rfftfreq(len(y)); m = fr > 0
        a.semilogy(1.0 / fr[m], P[m], lw=0.8)
        for q, lab in ((p["T"], "T"), (k["tau_e_steps"], "tau_e"), (k["tau_e_steps"] / 2, "tau_e/2")):
            a.axvline(q, ls="--", lw=0.8, label=f"{lab}={_fmt(float(q), 3)}")
        a.set_xscale("log"); a.legend(fontsize=8)
    a.set_title("speed spectrum (period, substeps)")
    a = ax[1, 0]
    if v.get("meanJ_series"):
        a.plot(v["sample_idx"], v["meanJ_series"], "o-", ms=3, label="mean J")
        a.plot(v["proxy_sample_idx"], v["proxy_rel_series"], "s--", ms=3, label="NN spacing^3 (rel)")
        a2 = a.twinx(); a2.plot(v["sample_idx"], v["speed_at_samples"], color="gray", lw=0.6, label="speed")
        a.legend(fontsize=8, loc="upper left")
    a.set_title(f"volume: J p2p={_fmt(v.get('meanJ_p2p'))} corr(J,speed)={_fmt(v.get('corr_J_speed'))}")
    a = ax[1, 1]
    if z.get("excursion_hist"):
        e = np.array(z["excursion_hist_edges"]); a.bar(e[:-1], z["excursion_hist"], width=np.diff(e), align="edge")
        a.axvline(RULES["visible_sp"], color="r", ls="--")
    a.set_title(f"tail excursion [sp]; frac>0.5sp={_fmt(z.get('frac_excursion_gt_half_sp'))}")
    fig.suptitle(f"{rep['decision']['verdict']}   dt={_fmt(p['dt'], 6)} dx={_fmt(p['dx'])} T={p['T']} s={_fmt(p['smoothing'])}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- synthetic
def make_synthetic_archive(kind, n=200, T=10, windows=30, side=2.0, cfl=0.1, dx=0.5,
                           young=DEFAULTS["young"], poisson=DEFAULTS["poisson"], seed=0,
                           held_every=0, amp=None):
    """In-memory archive in the pipeline_run layout for the tests. kinds:
    'stopgo'    per window: speed ramps up then decays to ~0 (u(1-u)^2 profile), direction
                alternating in x with a y drift (driver C, VISIBLE);
    'stiffness' uniaxial mode at the elastic period tau_e computed from (lam, mu, rho, L)
                for the chosen CFL (driver B when cfl > 0.3);
    'volume'    uniform breathing x <- c + (x-c)(1 + 0.05 sin), F = (1 + 0.05 sin) I (driver A);
    'drift'     smooth monotone drift + iid sub-spacing noise (no driver, INVISIBLE).
    dt is chosen as cfl * dx / c (unit particle mass). Returns (arrays, history, prov, cfg)."""
    rng = np.random.default_rng(seed)
    src = (rng.random((n, 3)) - 0.5) * side
    tgt = (rng.random((n, 3)) - 0.5) * side + np.array([3.0, 0.0, 0.0])
    lam, mu = lame(young, poisson)
    # same grid origin as the provenance below, so triage() recovers exactly this CFL
    st = stiffness_scales(src, lam, mu, 1.0, dx, DEFAULTS["grid_min"])   # c is dt-free
    dt = cfl * dx / st["c"]
    steps = windows * T; tt = np.arange(steps + 1, dtype=float)
    c0 = src.mean(0); scale = np.ones(steps + 1)
    if kind == "stopgo":
        amp = 0.3 if amp is None else amp
        u = (np.arange(1, T + 1) - 0.5) / T
        g = np.cumsum(u * (1 - u) ** 2); g /= g[-1]        # ramp up, decay to ~0 (never exactly)
        X = [src]; x = src.copy()
        for w in range(windows):
            d = np.array([0.8 * (1 if w % 2 == 0 else -1), 0.2, 0.0]); d /= np.linalg.norm(d)
            X.extend(x + amp * d * g[j] for j in range(T)); x = X[-1]
        X = np.stack(X)
    elif kind == "stiffness":
        amp = 0.03 if amp is None else amp
        P = stiffness_scales(src, lam, mu, dt, dx, DEFAULTS["grid_min"])["tau_e_steps"]
        mode = np.zeros_like(src); mode[:, 0] = src[:, 0] - c0[0]
        X = src[None] + amp * np.sin(2 * np.pi * tt / P)[:, None, None] * mode[None]
    elif kind == "volume":
        amp = 0.05 if amp is None else amp
        scale = 1.0 + amp * np.sin(2 * np.pi * tt / (4 * T))
        X = c0[None, None] + (src - c0)[None] * scale[:, None, None]
    elif kind == "drift":
        amp = 0.02 if amp is None else amp                  # per-commit drift (wu)
        X = src[None] + (amp / T) * tt[:, None, None] * np.array([0.0, 1.0, 0.0])[None, None] \
            + rng.normal(0.0, 0.002, (steps + 1, n, 3))
    else:
        raise ValueError(kind)
    frames, tof, history = [X[0]], [0], []
    for w in range(windows):
        t0 = w * T
        frames.extend(X[t0 + 1:t0 + T + 1]); tof.extend(range(t0 + 1, t0 + T + 1))
        e = len(frames); xe, xp = X[t0 + T], X[t0]
        v_end = (X[t0 + T] - X[t0 + T - 1]) / dt
        disp = (xe - xp).reshape(-1)
        prev = (X[t0] - X[t0 - T]).reshape(-1) if w > 0 else None
        rc = (float(disp @ prev / max(np.linalg.norm(disp) * np.linalg.norm(prev), 1e-12))
              if prev is not None else None)
        history.append(dict(animation=w, frame_end=e, loss=100.0 * math.exp(-w / 10.0),
                            d_vol=100.0 * math.exp(-w / 10.0) + 0.5 * rng.random(),
                            kin=float((v_end ** 2).sum(1).mean()),
                            v_mean=float(np.linalg.norm(v_end, axis=1).mean()),
                            v_absmax=float(np.abs(v_end).max()),
                            move=float(np.linalg.norm(xe - xp, axis=1).mean()),
                            Jmin=float(scale[t0 + T] ** 3), Jmin_traj=float(scale[t0:t0 + T + 1].min() ** 3),
                            **{"lambda": 1000.0 + (500.0 if (kind == "stopgo" and w % 2) else 0.0)},
                            dfc_absmax=0.01, accepted=8, rejected=0, reversal_cos=rc,
                            outer_gain=1.0, improved=1, stale=0, anneal=1.0))
        if held_every and (w + 1) % held_every == 0:
            frames.append(frames[-1].copy()); tof.append(tof[-1])
            history.append(dict(animation=w, held=1))
    frames = np.stack(frames).astype(np.float32)
    idx = sorted({0, len(frames) - 1} | {int(r["frame_end"]) - 1 for r in history if r.get("frame_end")})
    F_samples = np.stack([scale[tof[i]] * np.eye(3) for i in idx]).astype(np.float32)[:, None].repeat(n, 1)
    arrays = dict(src=src.astype(np.float32), tgt=tgt.astype(np.float32), frames=frames,
                  deliver_n=np.int64(len(frames)), F_samples=F_samples, F_sample_idx=np.array(idx))
    prov = dict(T=T, w_kin=0.5, synthetic=kind,
                mpm=dict(dx=dx, dt=dt, smoothing=0.955, drag=0.9, nx=64, ny=64, nz=64,
                         grid_min=list(DEFAULTS["grid_min"])))
    cfg = dict(T=T, young=young, poisson=poisson, w_kin=0.5)
    return arrays, history, prov, cfg


def save_archive(arrays, history, prov, cfg, prefix, arm="synthetic"):
    """Write <prefix>_<arm>.npz + <prefix>.json in the pipeline_run layout."""
    npz = f"{prefix}_{arm}.npz"; js = f"{prefix}.json"
    np.savez(npz, **arrays)
    with open(js, "w") as f:
        json.dump({"provenance": prov, "arms": {arm: {"history": history, "config": cfg}}}, f)
    return npz, js


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--json", default=None, help="run json (default: guessed from the npz name)")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--dt", type=_fraction, default=None, help="e.g. 1/240")
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--young", type=float, default=None)
    ap.add_argument("--poisson", type=float, default=None)
    ap.add_argument("--tail", type=int, default=RULES["tail"])
    ap.add_argument("--out", default=None, help="JSON report path")
    ap.add_argument("--png", default=None, help="4-panel figure (matplotlib)")
    a = ap.parse_args(argv)
    arrays, history, prov, cfg, arm, jp = load_archive(a.npz, a.json, a.arm)
    rep = triage(arrays, history, prov, cfg, T=a.T, dt=a.dt, dx=a.dx, young=a.young,
                 poisson=a.poisson, tail=a.tail)
    rep["inputs"] = dict(npz=str(a.npz), json=jp, arm=arm)
    print(summarize_markdown(rep))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"\n[triage] report -> {a.out}")
    if a.png:
        save_png(rep, a.png)
        print(f"[triage] figure -> {a.png}")
    return rep


if __name__ == "__main__":
    main()
