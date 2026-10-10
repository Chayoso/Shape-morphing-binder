"""The frozen recipe (gates D136 and D137) is scripts/pipeline_run.py's defaults, and every switch of it
still reaches the code as it was."""
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
    assert a.cell_follows_n                                                                 # D137
    assert a.spray_gate == "knn" and not a.render_body_only and not a.u_off                 # D126/D127/D124 not adopted
    assert a.surface_density == 1.0 and a.render_weight_scale == 1.0 and not a.baseline


def test_every_switch_reaches_the_old_path(monkeypatch):
    a = _args(monkeypatch, "--no-layer_relief", "--lambda_ema", "0.3", "--volume_exact", "off", "--no-match_density",
              "--no-assim_volume", "--no-render_exterior", "--min_spacing", "0", "--no-cell_follows_n")
    assert (a.layer_relief, a.lambda_ema, a.volume_exact, a.match_density, a.assim_volume, a.render_exterior,
            a.min_spacing, a.cell_follows_n) == (False, 0.3, "off", False, False, False, 0.0, False)


def test_the_bare_volume_flag_is_still_history(monkeypatch):
    assert _args(monkeypatch, "--volume_exact", "--no-assim_volume").volume_exact == "history"


# every argument of the D137 gate's runs (output/gpu/d137s/*300k_LM.json, provenance; the twins differ only in
# render_weight_scale 0): the frozen defaults must be exactly the configuration the freeze was judged on
D137_RUN_ARGS = {
    "animations": 300, "assim": None, "assim_volume": True, "baseline": "", "cell_diag": 26.0, "cell_follows_n": True,
    "drag": None, "exterior_radius": 3.0, "f_ext": None, "floor": False, "floor_friction": 0.0, "grad_dump": "",
    "lambda_ema": 1.0, "layer_relief": True, "live_dir": "", "live_port": 0, "loss_follows_n": True, "ls_probe": False,
    "match_density": True, "min_spacing": 0.9, "ot_iters": 1600, "patience": 5, "poisson": None, "profile": False,
    "reject_stop": 3, "render_body_only": False, "render_exterior": True, "render_res_hi": None,
    "render_target_draws": 8, "render_weight_scale": 1.0, "save_F_stride": 0, "spray_gate": "knn",
    "support_form": "proximity", "support_target_ref": False, "support_weight": 8.0, "surface_density": 1.0,
    "telemetry": False, "term_dump": "", "u_off": False, "volume_exact": "carried", "w_dt": None, "xu_form": "oracle",
    "young": None,
}


def test_the_defaults_are_the_d137_runs_configuration(monkeypatch):
    a = vars(_args(monkeypatch))
    for k in ("src", "tgt", "n", "seed", "out"):
        a.pop(k)
    assert a == D137_RUN_ARGS
