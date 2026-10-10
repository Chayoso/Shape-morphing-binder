"""The frozen recipe (tag freeze-2026-10-09b, D139; it supersedes freeze-2026-10-09) is scripts/pipeline_run.py's
defaults, and every switch of it still reaches the code as it was."""
import sys

from scripts.pipeline_run import parse_args


def _args(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["pipeline_run.py", *argv])
    return parse_args()


def test_the_defaults_are_the_frozen_recipe(monkeypatch):
    a = _args(monkeypatch)
    assert a.render_exterior and a.min_spacing == 0.9 and a.exterior_radius == 3.0          # D62, D70/D72, D123
    assert a.layer_relief and a.lambda_ema == 1.0                                           # D105, D120
    assert a.volume_exact == "carried" and a.match_density and a.assim_volume               # D131, D132, D135
    assert not a.cell_follows_n                                                             # the shape's cell (D139)
    assert not a.u_off and a.render_weight_scale == 1.0 and not a.baseline                  # D124 not adopted


def test_every_switch_reaches_the_old_path(monkeypatch):
    a = _args(monkeypatch, "--no-layer_relief", "--lambda_ema", "0.3", "--volume_exact", "off", "--no-match_density",
              "--no-assim_volume", "--no-render_exterior", "--min_spacing", "0")
    assert (a.layer_relief, a.lambda_ema, a.volume_exact, a.match_density, a.assim_volume, a.render_exterior,
            a.min_spacing) == (False, 0.3, "off", False, False, False, 0.0)
    assert _args(monkeypatch, "--cell_follows_n").cell_follows_n                            # D137 as an option


# every argument of D135's configuration as D139's bunny re-run recorded it (output/gpu/d139/bunny300k_SB.json,
# provenance; D135's stage-2 runs, output/gpu/d135s2/*300k_LT.json, record the same values and no cell_follows_n,
# which did not exist then: the shape's cell). Its term_dump (a read-only per-window dump) is the default '' here. The
# frozen defaults must be exactly this configuration, less the switches the cleanup (D141) removed with their paths: ls_probe,
# render_body_only, spray_gate, support_form, support_target_ref, support_weight, surface_density and w_dt, each recorded
# there at the value that ran the frozen path
D135_RUN_ARGS = {
    "animations": 300, "assim": None, "assim_volume": True, "baseline": "", "cell_diag": 26.0, "cell_follows_n": False,
    "drag": None, "exterior_radius": 3.0, "f_ext": None, "floor": False, "floor_friction": 0.0, "grad_dump": "",
    "lambda_ema": 1.0, "layer_relief": True, "live_dir": "", "live_port": 0, "loss_follows_n": True,
    "match_density": True, "min_spacing": 0.9, "ot_iters": 1600, "patience": 5, "poisson": None, "profile": False,
    "reject_stop": 3, "render_exterior": True, "render_res_hi": None,
    "render_target_draws": 8, "render_weight_scale": 1.0, "save_F_stride": 0,
    "telemetry": False, "term_dump": "", "u_off": False, "volume_exact": "carried", "xu_form": "oracle",
    "young": None,
}


def test_the_defaults_are_d135s_configuration(monkeypatch):
    a = vars(_args(monkeypatch))
    for k in ("src", "tgt", "n", "seed", "out"):
        a.pop(k)
    assert a == D135_RUN_ARGS
