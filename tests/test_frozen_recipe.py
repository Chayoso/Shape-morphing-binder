"""The frozen recipe is scripts/pipeline_run.py's defaults, and every switch of it still reaches the code as it was.

The re-freeze candidate (branch refreeze-cand; the user's decision of 2026-10-15: D135 + D146b + D149, the tag to come)
is the configuration of tag freeze-2026-10-09b (D139, which superseded freeze-2026-10-09) with three switches on:
--relax_driven (D146), --u_adam_carry (D146b) and --spacing_symmetric (D149). Each keeps its --no- form, the path of
freeze-2026-10-09b."""
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
    assert a.exterior_field == "zb"                                                         # D140 not adopted
    assert not a.ot_warm_trials and not a.cleanup_in_phys                                   # D142, D143 not adopted
    assert a.relax_driven and a.u_adam_carry                                                # D146, D146b: the re-freeze
    assert a.gap_split == "off"                                                              # D148 not adopted
    assert a.spacing_symmetric                                                              # D149: the re-freeze


def test_every_switch_reaches_the_old_path(monkeypatch):
    a = _args(monkeypatch, "--no-layer_relief", "--lambda_ema", "0.3", "--volume_exact", "off", "--no-match_density",
              "--no-assim_volume", "--no-render_exterior", "--min_spacing", "0", "--no-relax_driven",
              "--no-u_adam_carry", "--no-spacing_symmetric")
    assert (a.layer_relief, a.lambda_ema, a.volume_exact, a.match_density, a.assim_volume, a.render_exterior,
            a.min_spacing, a.relax_driven, a.u_adam_carry, a.spacing_symmetric) == (
        False, 0.3, "off", False, False, False, 0.0, False, False, False)
    assert _args(monkeypatch, "--cell_follows_n").cell_follows_n                            # D137 as an option


# every argument of D135's configuration as D139's bunny re-run recorded it (output/gpu/d139/bunny300k_SB.json,
# provenance; D135's stage-2 runs, output/gpu/d135s2/*300k_LT.json, record the same values and no cell_follows_n,
# which did not exist then: the shape's cell). Its term_dump (a read-only per-window dump) is the default '' here. The
# freeze-2026-10-09b defaults were exactly this configuration, less the switches the cleanup (D141) removed with their
# paths: ls_probe, render_body_only, spray_gate, support_form, support_target_ref, support_weight, surface_density and
# w_dt, each recorded there at the value that ran the frozen path; plus D140's exterior_field (added after; "zb" is the
# field D135 ran with) and the A/B switches of D142, D143, D146, D146b and D149 (ot_warm_trials, cleanup_in_phys,
# relax_driven, u_adam_carry, spacing_symmetric), off: the path D135 ran; and B13's force_scale, read only by the force
# baselines (--baseline force, force_mf)
D135_RUN_ARGS = {
    "animations": 300, "assim": None, "assim_volume": True, "baseline": "", "cell_diag": 26.0, "cell_follows_n": False,
    "drag": None, "exterior_field": "zb", "exterior_radius": 3.0, "f_ext": None, "floor": False, "floor_friction": 0.0, "grad_dump": "",
    "force_scale": 1.0, "lambda_ema": 1.0, "layer_relief": True, "live_dir": "", "live_port": 0, "loss_follows_n": True,
    "cleanup_in_phys": False, "match_density": True, "min_spacing": 0.9, "ot_iters": 1600, "ot_warm_trials": False,
    "gap_split": "off", "patience": 5, "poisson": None, "profile": False, "relax_driven": False, "u_adam_carry": False,
    "reject_stop": 3, "render_exterior": True, "render_res_hi": None, "spacing_symmetric": False,
    "render_target_draws": 8, "render_weight_scale": 1.0, "save_F_stride": 0,
    "morph_ramp": "off", "morph_ref_n": 0, "morph_u_gate": "window",   # D155 / D155b A/B switches (added after), off
    "telemetry": False, "term_dump": "", "u_off": False, "volume_exact": "carried", "xu_form": "oracle",
    "young": None,
}
# the re-freeze: D135's configuration with D146's, D146b's and D149's switches on, nothing else changed
REFREEZE_RUN_ARGS = {**D135_RUN_ARGS, "relax_driven": True, "u_adam_carry": True, "spacing_symmetric": True}


def _plain(a) -> dict:
    a = vars(a)
    for k in ("src", "tgt", "n", "seed", "out"):
        a.pop(k)
    return a


def test_the_defaults_are_the_refreeze_configuration(monkeypatch):
    assert _plain(_args(monkeypatch)) == REFREEZE_RUN_ARGS


def test_the_09b_configuration_is_three_switches_off(monkeypatch):
    """freeze-2026-10-09b's defaults (D135's configuration) stay reachable: the re-freeze's three switches off and
    nothing else changed."""
    a = _plain(_args(monkeypatch, "--no-relax_driven", "--no-u_adam_carry", "--no-spacing_symmetric"))
    assert a == D135_RUN_ARGS
