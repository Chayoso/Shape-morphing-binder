"""settle_diag.py OUT_DIR -- B9b step 1b (a diagnostic of step 1's reading; pre-registered in docs/experiments.md, B9b,
"Step 1b", before its run): why the loaded zero-control continuation does not settle in 8 s. The same run as
settle_probe.py (the assim_load definition's 40k bunny L97, re-run to 6 s, unchanged); at the committed windows 25, 30
and 35 (t 4.0, 4.83, 5.67 s) the two continuations are chained for 48 windows as in settle_probe.py, twice: with the
integrator as it is, and with its F smoothing off in the continuations only (MPMParams.smoothing 0: the stored F takes
each step's whole deformation increment; kernels.k_update keeps 1 - s = 4.5 % of it as it is). Recorded at chosen
chain lengths, for each continuation: the RMS speed split into its kinds (the centroid's speed, the rigid rotation's
|omega| R, the RMS of the rest, the deformation), the centroid's drop from the window's start, the elastic stretch
against Fp (p50 / p90 / p99) and the largest F change over the last window; and the read with each L. A diagnostic
only: the smoothing is the frozen integrator's, and nothing here proposes to change it. CUDA; run on hyde06."""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import settle_probe as SP  # noqa: E402  (imports physmorph first)
import torch  # noqa: E402

from physmorph.pipeline.scene import runner as RN  # noqa: E402
from physmorph.pipeline.scene import window as WN  # noqa: E402

WINDOWS = (25, 30, 35)
KS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
STATE = {"win": None, "a": 0, "rows": [], "out": None}


def motion(x, v, x_start):
    """The motion's kinds: |v_c| (wu/s), |omega| R (wu/s, R the largest distance from the centroid), the RMS of what is
    left (the deformation's speed), the centroid's drop since the window's start (wu)."""
    x, v = x.double(), v.double()
    c = x.mean(0)
    r = x - c
    vc = v.mean(0)
    vr = v - vc
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    inertia = (r * r).sum() * eye - r.T @ r
    w = torch.linalg.solve(inertia, torch.cross(r, vr, dim=1).sum(0))
    vd = vr - torch.cross(w.expand_as(r), r, dim=1)
    return {"v_c": float(vc.norm()), "omega_R": float(w.norm() * r.norm(dim=1).max()),
            "v_def_rms": float(vd.square().sum(1).mean().sqrt()), "v_rms": float(v.square().sum(1).mean().sqrt()),
            "drop_wu": float(x_start.double().mean(0)[1] - c[1])}


def chain(win, spec, Fc, Fp, smoothing):
    prm_l = win.prm if smoothing is None else dataclasses.replace(win.prm, smoothing=smoothing)
    prm_f = dataclasses.replace(prm_l, f_ext=(0.0, 0.0, 0.0), floor_y=-1.0e9)
    sl, sf = SP.state_of(spec), SP.state_of(spec)
    spl, spf = dataclasses.replace(spec, prm=prm_l), dataclasses.replace(spec, prm=prm_f)
    Fl_prev, Ff_prev = sl["F"].reshape(-1, 3, 3), sf["F"].reshape(-1, 3, 3)
    out = []
    for k in range(1, max(KS) + 1):
        if k > 1:
            spl, spf = SP.respec(win, 0, spec, sl, prm_l), SP.respec(win, 0, spec, sf, prm_f)
        sl, sf = SP.roll(win, 0, spl, True), SP.roll(win, 0, spf, False)
        if k in KS:
            L = sl["F"] @ torch.linalg.inv(sf["F"])
            out.append({"k": k, "read": RN.stretch_record(SP.LOAD_FREE_F(Fc, sl["F"], sf["F"]), Fp),
                        "L": RN.stretch_record(L, SP.eye_like(L)),
                        "loaded_strain": RN.stretch_record(sl["F"], Fp), "free_strain": RN.stretch_record(sf["F"], Fp),
                        "dF_loaded": float((sl["F"] - Fl_prev).abs().max()),
                        "dF_free": float((sf["F"] - Ff_prev).abs().max()),
                        "loaded": motion(sl["x"], sl["v"], spec.x0), "free": motion(sf["x"], sf["v"], spec.x0)})
        Fl_prev, Ff_prev = sl["F"], sf["F"]
    return out


def measure(win, F_ctrl, F_zl, F_zf):
    a = STATE["a"]
    if a + 1 not in WINDOWS:
        return
    t_clock = time.time()
    spec = win.specs[0]
    Fp = spec.Fp.reshape(-1, 3, 3).float()
    Fc = F_ctrl.reshape(-1, 3, 3).float()
    row = {"window": a + 1, "t_start": a * win.T * win.prm.dt, "smoothing": win.prm.smoothing,
           "frozen": RN.stretch_record(Fc, Fp), "one": RN.stretch_record(SP.LOAD_FREE_F(Fc, F_zl, F_zf), Fp),
           "as_is": chain(win, spec, Fc, Fp, None), "smoothing_0": chain(win, spec, Fc, Fp, 0.0)}
    row["seconds"] = time.time() - t_clock
    STATE["rows"].append(row)
    for name in ("as_is", "smoothing_0"):
        e = row[name][-1]
        m = e["loaded"]
        print(f"[diag] window {a + 1} {name}: k={e['k']} read p99 {e['read']['p99']:.4f} (frozen {row['frozen']['p99']:.4f}); "
              f"L p99 {e['L']['p99']:.4f}; loaded: v_rms {m['v_rms']:.3e}, v_c {m['v_c']:.3e}, omega R {m['omega_R']:.3e}, "
              f"v_def {m['v_def_rms']:.3e}, drop {m['drop_wu']:.4f} wu, dF {e['dF_loaded']:.2e}; free v_rms "
              f"{e['free']['v_rms']:.3e}", flush=True)
    (STATE["out"] / "diag_partial.json").write_text(json.dumps({"rows": STATE["rows"]}, indent=1))
    torch.cuda.empty_cache()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    STATE["out"] = out
    orig_free, orig_load = WN.SceneWindow.free_zero_end_F, RN.load_free_F

    def free_zero_end_F(self):
        STATE["win"] = self
        return orig_free(self)

    def load_free_F(F_ctrl, F_zl, F_zf):
        try:
            measure(STATE["win"], F_ctrl, F_zl, F_zf)
        finally:
            STATE["a"] += 1
        return orig_load(F_ctrl, F_zl, F_zf)

    WN.SceneWindow.free_zero_end_F = free_zero_end_F
    RN.load_free_F = load_free_F
    from scripts import scene_run
    windows = max(WINDOWS)
    sys.argv = ["scene_run.py", "--scene", "scripts/probes/b9b/specs/b9d_bunny.json", "--tgt", "assets/bunny.obj",
                "--n", "40000", "--seed", "97", "--save_stride", "12", "--windows", str(windows), "--out", str(out / "run")]
    t0 = time.time()
    scene_run.main()
    res = {"windows": WINDOWS, "KS": KS, "minutes": (time.time() - t0) / 60, "rows": STATE["rows"]}
    (out / "diag.json").write_text(json.dumps(res, indent=1))
    print(f"DIAG done: {len(STATE['rows'])} windows; saved {out / 'diag.json'}")


if __name__ == "__main__":
    main()
