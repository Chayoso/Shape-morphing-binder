"""settle_probe.py OUT_DIR [T0 T1 K] -- B9b step 1 (the coordinator's go, 2026-10-22; pre-registered in
docs/experiments.md, B9b, "Step 1: the one-window load effect against the settled one"): what the assimilation would
read if the loads' effect were taken from the settled continuations, not from one window's.

The 40k bunny L97 run of the assim_load definition (specs/b9d_bunny.json: standing frame, g 0.65, the floor, drag 0.9)
is run again to the end of the last measured window (the run itself unchanged: its assimilation reads the definition's
one-window value, as in the gate). At each committed window whose start lies in [T0, T1) (default 4-6 s), before the
assimilation, it reads (body 0):
  frozen   the committed end F against the window's Fp (what the frozen rule assimilates);
  one      the definition's reading, F_zf F_zl^-1 F_ctrl (zero-control continuations with and without the loads, one window);
  chain k  the same two continuations chained window after window from the window's start (zero control and zero u;
           with the loads: gravity, the floor and its position constraint; without: alone, no gravity, no floor), each
           window rebuilt from the previous end as the scene builds a window (the layer's data, the relief, the minimum
           spacing's neighbours from the new positions; Fp, the bonds, the volumes the window's), for k = 1 .. K windows
           (default 48, 8 s); at chosen k the reading F_f,k F_l,k^-1 F_ctrl, the load effect L_k = F_l,k F_f,k^-1 itself,
           each continuation's elastic stretch against Fp, its largest F change over its last window and its RMS speed;
  carried  step 2's candidate: the pair settled at the first measured window, then advanced one window per committed
           window with that window's Fp, read against the freshly settled one.
Readings are each particle's largest |log sigma| of F_e = F Fp^-1: p50 / p90 / p99 (runner.stretch_record). A check: the
chain's first window equals the definition's own continuations (largest F difference). Writes OUT_DIR/settle.json and
prints one line a window. The scene's run output goes to OUT_DIR/run*. CUDA; run on hyde06."""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import physmorph  # noqa: E402,F401  (before torch)
import torch  # noqa: E402
import warp as wp  # noqa: E402

from physmorph import gpu  # noqa: E402
from physmorph.mpm.bodies import Bodies, body_trajectory  # noqa: E402
from physmorph.pipeline.scene import runner as RN  # noqa: E402
from physmorph.pipeline.scene import window as WN  # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402

KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48)
LOAD_FREE_F = RN.load_free_F            # the definition's reading, kept before main() wraps the runner's name
STATE = {"win": None, "a": 0, "rows": [], "carried": None, "t0": 4.0, "t1": 6.0, "K": 48, "out": None}


def state_of(spec) -> dict:
    return {"x": spec.x0, "F": spec.F0, "v": spec.v0, "C": spec.C0, "J": spec.J0}


def respec(win, b, spec, st, prm, Fp=None):
    """The window spec rebuilt at a new state, as SceneWindow builds it (the layer's data, the relief and the minimum
    spacing's neighbours from the new positions); the u gate, the bonds and the volumes are the window's."""
    cfg, body = win.cfg, win.bodies[b]
    x0 = st["x"]
    y0, c0, R0 = (t.detach() for t in body.frame(x0))
    sp_b = layer_spacing(x0)
    lmask, lnrm, lnbr, lw = layer_relax_data(x0, sp_b, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    ref = None if body.tgt.relief is None else body.tgt.relief.at(y0, lmask, lnrm @ R0, lnbr, lw)
    layer = (lmask, lnrm, lnbr, lw, 1.0 / float(cfg.T), spec.layer[5], ref, cfg.T if cfg.relax_driven else None)
    spacing = spec.spacing
    if spacing is not None:
        snbr = gpu.knn(x0, 17)[1][:, 1:]
        spacing = (gpu.symmetric_rows(snbr) if cfg.spacing_symmetric else snbr, spacing[1])
    return dataclasses.replace(spec, x0=x0, F0=st["F"], v0=st["v"], C0=st["C"], J0=st["J"], layer=layer,
                               spacing=spacing, prm=prm, Fp=spec.Fp if Fp is None else Fp)


def roll(win, b, spec, loaded: bool) -> dict:
    """One window of zero control and zero u from spec's start: with the loads as window.zero_end_F builds it (the
    floor's position constraint), or without them as window.free_zero_end_F. The end state."""
    body, T = win.bodies[b], win.T
    dz = torch.zeros(T, body.n, 3, 3, device=win.dc_buf.device)
    uz = torch.zeros(body.n, device=win.u_buf.device)
    seq = [wp.from_torch(dz[t], dtype=wp.mat33) for t in range(T)]
    uw = wp.from_torch(uz, dtype=wp.float32)
    tr = body_trajectory(spec, seq, uw, requires_grad=False, persistent=False, **({"vp": body.vp} if loaded else {}))
    Bodies([tr], [body.vp], win.mu).rollout()
    out = {"x": wp.to_torch(tr.x[T]).clone(), "F": wp.to_torch(tr.F[T]).reshape(-1, 3, 3).clone(),
           "v": wp.to_torch(tr.v[T]).clone(), "C": wp.to_torch(tr.C[T]).reshape(-1, 3, 3).clone(),
           "J": wp.to_torch(tr.J[T]).clone() if getattr(tr, "volume_exact", False) else None}
    del tr, seq, dz, uz
    return out


def eye_like(F):
    return torch.eye(3, device=F.device).expand(F.shape[0], 3, 3)


def measure(win, F_ctrl, F_zl, F_zf):
    a = STATE["a"]
    t_start = a * win.T * win.prm.dt
    if not (STATE["t0"] <= t_start < STATE["t1"]):
        return
    t_clock = time.time()
    b, spec = 0, win.specs[0]
    Fp = spec.Fp.reshape(-1, 3, 3).float()
    Fc = F_ctrl.reshape(-1, 3, 3).float()
    prm_l = win.prm
    prm_f = dataclasses.replace(win.prm, f_ext=(0.0, 0.0, 0.0), floor_y=-1.0e9)
    row = {"window": a + 1, "t_start": t_start, "frozen": RN.stretch_record(Fc, Fp),
           "one": RN.stretch_record(LOAD_FREE_F(Fc, F_zl, F_zf), Fp), "chain": []}
    sl, sf = state_of(spec), state_of(spec)
    spl, spf = spec, dataclasses.replace(spec, prm=prm_f)
    Fl_prev, Ff_prev = sl["F"].reshape(-1, 3, 3), sf["F"].reshape(-1, 3, 3)
    for k in range(1, STATE["K"] + 1):
        if k > 1:
            spl, spf = respec(win, b, spec, sl, prm_l), respec(win, b, spec, sf, prm_f)
        sl, sf = roll(win, b, spl, True), roll(win, b, spf, False)
        if k == 1:
            row["first_vs_definition"] = {"loaded": float((sl["F"] - F_zl.reshape(-1, 3, 3)).abs().max()),
                                          "free": float((sf["F"] - F_zf.reshape(-1, 3, 3)).abs().max())}
        if k in KS or k == STATE["K"]:
            L = sl["F"] @ torch.linalg.inv(sf["F"])
            row["chain"].append({
                "k": k, "read": RN.stretch_record(LOAD_FREE_F(Fc, sl["F"], sf["F"]), Fp),
                "L": RN.stretch_record(L, eye_like(L)),
                "loaded_strain": RN.stretch_record(sl["F"], Fp), "free_strain": RN.stretch_record(sf["F"], Fp),
                "dF_loaded": float((sl["F"] - Fl_prev).abs().max()), "dF_free": float((sf["F"] - Ff_prev).abs().max()),
                "v_rms_loaded": float(sl["v"].square().sum(1).mean().sqrt()),
                "v_rms_free": float(sf["v"].square().sum(1).mean().sqrt())})
        Fl_prev, Ff_prev = sl["F"], sf["F"]
    settled = {"Fl": sl["F"], "Ff": sf["F"]}
    # step 2's candidate: the pair carried across windows (settled at the first measured window, then one window per
    # commit with the window's Fp), read against the freshly settled pair
    if STATE["carried"] is None:
        STATE["carried"] = {"l": sl, "f": sf}
        row["carried"] = {"read": row["chain"][-1]["read"], "note": "initialised from this window's settled pair"}
    else:
        cl = roll(win, b, respec(win, b, spec, STATE["carried"]["l"], prm_l, Fp=spec.Fp), True)
        cf = roll(win, b, respec(win, b, spec, STATE["carried"]["f"], prm_f, Fp=spec.Fp), False)
        STATE["carried"] = {"l": cl, "f": cf}
        Lc = cl["F"] @ torch.linalg.inv(cf["F"])
        Ls = settled["Fl"] @ torch.linalg.inv(settled["Ff"])
        row["carried"] = {"read": RN.stretch_record(LOAD_FREE_F(Fc, cl["F"], cf["F"]), Fp),
                          "L_vs_settled_max": float((Lc - Ls).abs().max()),
                          "L_vs_settled_rms": float((Lc - Ls).square().sum((1, 2)).mean().sqrt())}
    row["seconds"] = time.time() - t_clock
    STATE["rows"].append(row)
    e = row["chain"][-1]
    f99, o99, s99 = row["frozen"]["p99"], row["one"]["p99"], e["read"]["p99"]
    print(f"[settle] window {a + 1} t {t_start:.3f}: frozen p99 {f99:.4f}; one-window {o99:.4f} "
          f"({100 * (1 - o99 / f99):.0f}% removed); settled k={e['k']} {s99:.4f} ({100 * (1 - s99 / f99):.0f}% removed); "
          f"L p99 one {RN.stretch_record(F_zl.reshape(-1, 3, 3) @ torch.linalg.inv(F_zf.reshape(-1, 3, 3)), eye_like(Fc))['p99']:.4f} "
          f"settled {e['L']['p99']:.4f}; last-window dF {e['dF_loaded']:.2e} / {e['dF_free']:.2e}; carried "
          f"{row['carried']['read']['p99']:.4f}; first window vs definition {row['first_vs_definition']}; "
          f"{row['seconds']:.0f} s", flush=True)
    (STATE["out"] / "settle_partial.json").write_text(json.dumps({"rows": STATE["rows"]}, indent=1))   # kept per window
    torch.cuda.empty_cache()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    STATE["out"] = out
    if len(sys.argv) > 3:
        STATE["t0"], STATE["t1"] = float(sys.argv[2]), float(sys.argv[3])
    if len(sys.argv) > 4:
        STATE["K"] = int(sys.argv[4])
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
    import math
    from scripts import scene_run
    windows = int(math.ceil(STATE["t1"] * 6.0 - 1e-9))         # the bunny's windows are 1/6 s (2 T dt, T 20, dt 1/240)
    sys.argv = ["scene_run.py", "--scene", "scripts/probes/b9b/specs/b9d_bunny.json", "--tgt", "assets/bunny.obj",
                "--n", "40000", "--seed", "97", "--save_stride", "12", "--windows", str(windows), "--out", str(out / "run")]
    t0 = time.time()
    scene_run.main()
    res = {"t0": STATE["t0"], "t1": STATE["t1"], "K": STATE["K"], "windows_run": windows, "minutes": (time.time() - t0) / 60,
           "rows": STATE["rows"]}
    (out / "settle.json").write_text(json.dumps(res, indent=1))
    print(f"SETTLE done: {len(STATE['rows'])} windows measured; saved {out / 'settle.json'}")


if __name__ == "__main__":
    main()
