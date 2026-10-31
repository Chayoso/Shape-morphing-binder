"""Morph a source mesh into a target mesh with settled transport (README.md). CUDA only.

Run on hyde06 (scripts/ops/hyde06_env.sh sets REPO, OUT and PY):
  $PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 300000 --seed 97 --out $OUT/bunny

Outputs: <out>_render_full_dt_iso_nn.npz (the archive the renderers read: frames, the
delivered slice, F samples) and <out>.json (provenance, config, metrics, gates and the
per-window history). The prepare stage (mesh sampling, cached) is the only CPU work.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physmorph import gpu, metrics  # noqa: E402
from physmorph.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from physmorph.surface import surface_roughness  # noqa: E402
from physmorph.thin import thin_metrics, thin_set_cached  # noqa: E402
from physmorph.prepare import prepare  # noqa: E402
from physmorph.sampling.orientation import orient_name  # noqa: E402

# the archive name of the adopted recipe, kept so new archives read like the earlier ones
ARM = "render_full_dt_iso_nn"
TRACKED = ("physmorph/gpu.py", "physmorph/prepare.py", "physmorph/pipeline/config.py",
           "physmorph/pipeline/target.py", "physmorph/pipeline/render_loss.py",
           "physmorph/pipeline/window", "physmorph/pipeline/run", "physmorph/losses",
           "physmorph/mpm/kernels.py", "physmorph/mpm/traj.py", "physmorph/mpm/function.py")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", default="output/run")
    ap.add_argument("--animations", type=int, default=300, help="window budget")
    ap.add_argument("--patience", type=int, default=5,
                    help="windows without a merit improvement that stop the run")
    ap.add_argument("--reject_stop", type=int, default=3,
                    help="consecutive rejected windows that stop the run at the best window")
    ap.add_argument("--render_weight_scale", type=float, default=1.0,
                    help="multiplies the render weight; 0 = the render-off twin")
    ap.add_argument("--render_weight_ref", default="residual", choices=["residual", "floor", "shape", "floor_pitch"],
                    help="D162 (1) A/B: residual (the default, the frozen path) = the rule lambda |g_r| = 0.5 |g_p| at every "
                         "window; floor = lambda_0 max(0, 1 - L_floor / L_r): the render weight referenced to the render's "
                         "own excess over the render term of the pictures' own first sample; shape (D162b) = the same weight "
                         "with the excess read on the body's shape, 1 - b_floor / b (physmorph/pipeline/shape_read.py); floor_pitch (D162d) = floor with L_r and "
                         "L_floor read on pictures low-passed at one pitch")
    ap.add_argument("--cleanup_off", action="store_true",
                    help="29b (render-centric recipe): the cleanup off, w_dt = w_nn = 0 (the spray W1 pull and the near band)")
    ap.add_argument("--settle_windows", type=int, default=30,
                    help="D162h / D162i: windows of released physics with zero controls after the run stops (delivered); "
                         "30 = freeze-2026-11 (0 = the earlier tag)")
    ap.add_argument("--settle_u", default="zero", choices=["zero", "last"],
                    help="D166 A/B: u in the settle windows; zero (the default, D162h) or last = the last committed u held")
    ap.add_argument("--settle_relax", action=argparse.BooleanOptionalAction, default=False,
                    help="D166: the outer-layer relaxation in the settle windows (freeze-2026-11: off; --settle_relax = D162h)")
    ap.add_argument("--layer_relax", action=argparse.BooleanOptionalAction, default=True,
                    help="D170 A/B: the outer-layer relaxation in every window (default on, the frozen path); "
                         "--no-layer_relax = its fraction 0 throughout (u and everything else unchanged)")
    ap.add_argument("--stop_tol", type=float, default=0.12,
                    help="D174 A/B: the stop's own relative-improvement tolerance (the plateau and the patience stop); the "
                         "step anneal and the latch keep the config's tol (0.003); unset = tol, the frozen path")
    ap.add_argument("--render_every", type=int, default=1,
                    help="D174 A/B (runtime): the render gradient's adjoint sweep at every k-th iteration of a window, the "
                         "window's last one reused between (1 = every iteration, the frozen path)")
    ap.add_argument("--ls_max", type=int, default=10,
                    help="D174 A/B (runtime): the line search's trials per iteration (10 = the frozen path)")
    ap.add_argument("--render_split", default="appearance", choices=["", "u", "dFc", "u_rule", "masked", "content", "appearance"],
                    help="D172 A/B: the render gradient's weight by control block; \"\" = one weight on both (the frozen "
                         "path); u = the run's weight (held under --lambda_ema 0) on u and the rule's per-window value on dFc; "
                         "dFc = the reverse; the merit keeps the run's weight; u_rule = the run on the rule (its merit and "
                         "stops too) with u at the first calibration's value (what --lambda_ema 0 holds); masked (D178) = u "
                         "as u, dFc on the rule plus the held weight on the render residual inside the content mask "
                         "(the pixels where the pictures differ from the 3D target's own beyond the draw-noise floor); content "
                         "(D178, M2) = masked, and with an appearance target the held term is the appearance part of the "
                         "render residual everywhere plus its geometry part inside the geometry content mask; appearance "
                         "(D178, M3) = u, and with an appearance target dFc takes the held weight on the appearance part "
                         "of the residual (the 3D target fixes no colour); geometry stays on the rule for dFc (through u)")
    ap.add_argument("--keyframe", nargs="+", default=[],
                    help="D175 (off by default): picture keyframes, one .npz per --keyframe_time, each with 'samples' "
                         "(K, N_k, 3): the keyframe's sample and its draws in the target's frame, read through their "
                         "pictures only (physmorph/pipeline/run/storyboard.py); an optional 'A' (3, 3) is the linear map "
                         "that made it from the source, checked to carry no rotation")
    ap.add_argument("--keyframe_time", nargs="+", type=float, default=[],
                    help="D175: the keyframes' times in s (strictly increasing; needs --lambda_ema 0)")
    ap.add_argument("--keyframe_soundness", default="every", choices=["every", "final"],
                    help="D175c: every (D175) = the released motion and the end drift in every window; final = only with the "
                         "final target: a keyframe segment's merit is its comparison alone")
    ap.add_argument("--render_compare", default="exterior", choices=["exterior", "column"],
                    help="D178 (the J5 attempt): the final target's comparison; exterior (the default) = the exterior "
                         "silhouette + shading; column = the projected count picture against the render samples' (d_count)")
    ap.add_argument("--keyframe_compare", default="silhouette", choices=["silhouette", "column"],
                    help="D175b: the keyframe comparison; silhouette (D175) = the exterior silhouette + shading; column = "
                         "the particles' projected count picture against the keyframe samples' (the final target's render "
                         "term is unchanged)")
    ap.add_argument("--keyframe_geometry", default="own", choices=["own", "final"],
                    help="D175 ablation: own (the method) = a segment reads only the keyframe's pictures and the "
                         "soundness terms; final = the pictures swapped, the final target's geometry and cleanup kept")
    ap.add_argument("--placement_at_cell", action=argparse.BooleanOptionalAction, default=False,
                    help="D162 (2) A/B (off by default): the placement terms read the body at the loss cell (the "
                         "proximity covers within one loss cell, the near band's band is empty, the spray gate in loss cells)")
    ap.add_argument("--proximity_berth", action=argparse.BooleanOptionalAction, default=False,
                    help="D161b A/B (off by default): the surface proximity counts a target point as covered when a body "
                         "particle lies within the near band's berth (its floor K(berth) in place of K(sp) / 2)")
    ap.add_argument("--render_tgt", default=None,
                    help="D156 (off by default): the mesh the render's target pictures are drawn from (and the frame, the "
                         "cameras'), in place of --tgt's; --tgt is then the transport's and the body's target, sampled in "
                         "that frame (prepare.prepare)")
    # the defaults below are the frozen recipe (tag freeze-2026-10-09b, D139; it supersedes freeze-2026-10-09): D105's,
    # whose render arm is ahead of its physics-only twin on every display measure read against an independent sample
    # (D91, D92; the minimum spacing D70/D72, the exterior D62, the layer relief D105), with the render weight at the
    # rule's value (D120) and the volume-conserving body (D131, D132, D135), on the MPM cell from the shape (D137's cell
    # following N is an option); each stays an A/B switch back to the code as it was
    ap.add_argument("--lambda_ema", type=float, default=0.0,
                    help="D120: the render weight's moving average per window; 1 (the default) = the calibration rule's "
                         "value at every window; 0.3 = the average before the freeze (config.lambda_ema)")
    ap.add_argument("--render_exterior", action=argparse.BooleanOptionalAction, default=True,
                    help="read the render terms on the exterior (surface discs) in place of the particle cloud")
    ap.add_argument("--min_spacing", type=float, default=0.9,
                    help="the position update keeps particles this far apart, in pitches of the rest volume (0: off)")
    ap.add_argument("--spacing_symmetric", action=argparse.BooleanOptionalAction, default=True,
                    help="D149 (the re-freeze's default): the minimum spacing's neighbours are the symmetric closure of "
                         "each particle's 16 nearest at the window start, so every pair's push is equal and opposite (the "
                         "push keeps the centre of mass); --no-spacing_symmetric = each particle's own 16 nearest, the "
                         "path of freeze-2026-10-09b")
    ap.add_argument("--exterior_radius", type=float, default=3.0,
                    help="D123 A/B: the exterior field's kernel radius in pitches, its offset in proportion (0.8 at 3); "
                         "3 = D59's field, the code as it was")
    ap.add_argument("--exterior_field", default="zb", choices=["zb", "aniso", "zbs"],
                    help="D140 / D144 A/B: the exterior's field; zb (the default) = Zhu and Bridson's, the code as it was; "
                         "aniso = the anisotropic density field (D140, not adopted); zbs = Zhu and Bridson's with "
                         "Solenthaler et al.'s correction (D144)")
    ap.add_argument("--relax_driven", action=argparse.BooleanOptionalAction, default=True,
                    help="D146 (the re-freeze's default): the layer relaxation acts in the T driven steps only, the steps "
                         "u acts in, so the released half has no position edit; --no-relax_driven = every step of the "
                         "window, the path of freeze-2026-10-09b")
    ap.add_argument("--gap_split", default="off", choices=["off", "on", "interface", "soft", "fine", "walls", "nearest",
                                                          "carry"],
                    help="D148 A/B: the two sides of a target gap on their own grid fields (mpm/gap_kernels.py); on = at "
                         "every gap node (variant A), interface = where the node's particle volume is below one cell "
                         "(variant B); off (the default) = the single field, the frozen path")
    ap.add_argument("--u_adam_carry", action=argparse.BooleanOptionalAction, default=True,
                    help="D146b (the re-freeze's default): u's Adam moments carry from one committed window to the next "
                         "(u still starts at zero; each coordinate's bias correction counts its own age; reset on an outer "
                         "reject); --no-u_adam_carry = fresh moments every window, the path of freeze-2026-10-09b")
    ap.add_argument("--render_normal_radius", type=float, default=0.0,
                    help="D183: the render loss's disc normals (run and target pictures) from the Zhu-Bridson gradient at this "
                         "kernel radius in pitches (1.5: the measured scale); 0 (the default) = the field's own 3-pitch gradient")
    ap.add_argument("--stop_physics", action=argparse.BooleanOptionalAction, default=False,
                    help="D181b: the stop rule (its best, last improving window, stale count, plateau) reads the physics part "
                         "of the merit (q - lambda x render); acceptance, the latch, the anneal and the delivery keep the full "
                         "merit; off (the default) = the stop of freeze-2026-11. The twin (lambda 0) is unchanged")
    ap.add_argument("--u_render_smooth", action=argparse.BooleanOptionalAction, default=False,
                    help="D181: the render's gradient on u is averaged over each layer particle's same-side layer "
                         "neighbours (Gaussian of one particle spacing, sigma = sp0, times the normal agreement), after "
                         "PCGrad and the lambda calibration, before Adam: u moves as a field over the outer layer; off "
                         "(the default) = the per-particle step of freeze-2026-11. No effect where the render's weight on u "
                         "is zero (the twin) and none on dFc's gradient")
    ap.add_argument("--u_off", action="store_true",
                    help="D124 A/B (ablation): the u channel never acts (its gate zero on every particle); all else as is")
    ap.add_argument("--volume_exact", default="carried", choices=["off", "carried"],
                    help="D131 (default carried): the motion's own volume J per particle, carried like F in the smoothed F "
                         "itself (det F = J after every step, the stress reading F + dFc); off = the old path bit for bit")
    ap.add_argument("--appearance", default="", choices=["", "white", "half", "band", "ears", "turn", "stripes", "texture"],
                    help="D160 (an option, off by default): each particle carries a fixed albedo from the source and the "
                         "render's shading term compares colour pictures of the target's paint (physmorph/pipeline/"
                         "appearance.py: the scenes); the physics never reads it; \"\" (the default) = the frozen path; "
                         "white = one channel of ones against a white target, the frozen path bit for bit")
    ap.add_argument("--appearance_placebo", action="store_true",
                    help="D178 (the placebo arm, off by default): the scene's paint is carried and archived (src_albedo, "
                         "tgt_albedo: the colour readers' input), but the render loss reads the 3D target's own grey "
                         "pictures (no appearance in the loss): content-free colour pictures")
    ap.add_argument("--colour_kernel", default="nearest", choices=["kernel", "nearest"],
                    help="D173 A/B: the render loss's disc colours (target and body alike): kernel = D160's 3-pitch weights (the "
                         "frozen path); nearest = the nearest particle's colour, a softmax of -d^2 / (pitch / 2)^2 (differentiable)")
    ap.add_argument("--texture", default="",
                    help="D173 (with --appearance texture): OBJ,PNG -- the textured OBJ (UVs; vertices in the --tgt mesh's raw "
                         "coordinates) and its image (map_Kd)")
    ap.add_argument("--render_target_draws", type=int, default=8,
                    help="the render's target pictures are the mean over this many independent samples of the target")
    ap.add_argument("--render_target", default="samples", choices=["samples", "mesh_shell"],
                    help="D159 A/B: the render's target pictures; samples (the default) = the target sample and its draws "
                         "through the exterior, the frozen path; mesh_shell = the target mesh grown by the run's own "
                         "measured exterior offset (render/mesh_shell.py)")
    ap.add_argument("--xu_form", default="oracle", choices=["oracle", "paper"],
                    help="the baseline loss: the C++ code's (min-mass penalty, out-of-target gradient x5) or the "
                         "published one (Xu et al., arXiv 2409.15746)")
    ap.add_argument("--baseline", default="", choices=["", "xu", "xu_spray", "force", "force_mf"],
                    help="xu: the comparison baseline, Xu et al.'s objective alone in this simulator (D112); "
                         "xu_spray: the same with our spray cleanup, the ejection guard; force (B13): everything ours "
                         "but an external force per particle in place of dFc (held at zero); force_mf: that force with "
                         "no net force and no net torque")
    ap.add_argument("--force_scale", type=float, default=1.0,
                    help="B13: the force leaf's unit, x (lam + 2 mu) V / (m dx); read only by --baseline force*")
    ap.add_argument("--layer_relief", action=argparse.BooleanOptionalAction, default=True,
                    help="the outer layer's relaxation keeps the target mesh's own relief (n / 4 points of its surface; "
                         "D105); --no-layer_relief = the plane of the neighbours, as before")
    ap.add_argument("--assim_volume", action=argparse.BooleanOptionalAction, default=True,
                    help="D135 (default on): the per-commit assimilation takes the elastic stretch's volume too (Fp loses its "
                         "isochoric restriction; the band assim_smin..assim_smax bounds every principal stretch), so the volume "
                         "the end keeps becomes the rest volume; --no-assim_volume = isochoric as before")
    ap.add_argument("--match_density", action=argparse.BooleanOptionalAction, default=True,
                    help="D132 (default on): the source sample rescaled so that it represents the volume of the target "
                         "sample's own fill (equal number densities; the target and every measure on it unchanged); "
                         "--no-match_density = the meshes' volumes matched at a 110^3 fill, as before")
    ap.add_argument("--cell_follows_n", action=argparse.BooleanOptionalAction, default=False,
                    help="D137 (an option; off by default since freeze-2026-10-09b, D139): above the reference N (40k) the "
                         "MPM cell follows the particle count, dx = (diag / cell_diag) x (40000 / N)^(1/3) (the particles "
                         "per cell stay 40k's); the loss grid exactly as before, the u gate and the thin set on the shape's "
                         "cell. It opens gaps narrower than two of the shape's cells but lets the surface form features "
                         "ahead of the bulk early (D139); off (the default, or N <= 40k) = the shape's cell")
    ap.add_argument("--render_res_hi", type=int, default=None,
                    help="the render resolution, used from the first window (default: the config's fine one, following N)")
    ap.add_argument("--replay_noise", default="window", choices=["window", "start"],
                    help="D164: the commit check's replay noise from the window's own replay pairs, floored at the "
                         "objective's float32 resolution, and the commit reading the accepted candidate's transport "
                         "potentials and proximity lists (default); 'start': the tag's check and commit (comparison only)")
    ap.add_argument("--ot_iters", type=int, default=1600, help="Sinkhorn sweep budget per solve")
    ap.add_argument("--ot_warm_trials", action=argparse.BooleanOptionalAction, default=False,
                    help="A/B (D142): line-search trials warm-start their transport solves from the window's latest "
                         "solved potentials; the commit reads the accepted candidate's (off = the frozen path)")
    ap.add_argument("--cleanup_in_phys", action=argparse.BooleanOptionalAction, default=False,
                    help="A/B (D143): the cleanup's gradient joins the physics channel for PCGrad and the render "
                         "weight, two adjoint sweeps an iteration (off = the frozen path)")
    ap.add_argument("--loss_follows_n", action=argparse.BooleanOptionalAction, default=True,
                    help="transport grid, blur and render pictures follow the particle spacing above mass_ref_n")
    ap.add_argument("--cell_diag", type=float, default=26.0,
                    help="the MPM cell from the shape: dx = source bbox diagonal / cell_diag")
    ap.add_argument("--save_F_stride", type=int, default=0,
                    help="archive every k-th frame's F (0 = every T frames)")
    ap.add_argument("--grad_dump", default="", help="directory of per-window gradient dumps")
    ap.add_argument("--profile", action="store_true",
                    help="diagnostic: record the wall-clock split of every window (slows the run)")
    ap.add_argument("--young", type=float, default=None, help="material: Young's modulus (default: the config's)")
    ap.add_argument("--poisson", type=float, default=None, help="material: Poisson's ratio")
    ap.add_argument("--assim", type=float, default=None,
                    help="material: the share of the elastic strain made permanent at every commit")
    ap.add_argument("--drag", type=float, default=None, help="material: momentum drag rate (1/s)")
    ap.add_argument("--f_ext", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="external force: a uniform acceleration (wu/s^2) on every grid node")
    ap.add_argument("--floor", action="store_true",
                    help="external force: a separating floor at the source's lowest point; the target stands on it")
    ap.add_argument("--floor_friction", type=float, default=0.0, help="Coulomb friction of the floor")
    ap.add_argument("--telemetry", action="store_true",
                    help="diagnostic: the per-iteration steering telemetry and the per-window records (support split, active sets, scale, control, thin, OT divergence)")
    ap.add_argument("--term_dump", default="",
                    help="diagnostic: directory of each term's per-particle position gradient at every committed window")
    ap.add_argument("--live_port", type=int, default=0, help=">0: stream to the live viewer")
    ap.add_argument("--live_dir", default="", help="file-backed viewer sink (scripts/viewer_serve.py)")
    return ap.parse_args(argv)


def eval_gates(res, met, prm, T, rel_tol=0.003, hole_tol=0.02):
    """G2 guards all zero; G3 rest (tail jitter and the drift of the delivered terminal
    velocity); G4 holes (absolute, or the target's own level) and ejection."""
    dn = res["deliver_n"]
    recs = [h for h in res["history"] if "v_mean" in h and h.get("frame_end") is not None
            and not h.get("null_commit") and h["frame_end"] <= dn]
    v_mean = recs[-1]["v_mean"] if recs else 0.0
    drift_rel = v_mean * prm.dt * T / max(met["bbox_diag"], 1e-9)
    gates = {"G2_guards": all(v == 0 for v in res["guards"].values()),
             "G3_rest": met["jitter_rel"] < rel_tol and drift_rel < rel_tol,
             "G4_holes_abs": met["hole_frac"] <= max(hole_tol, met.get("hole_frac_tgt", 0.0) + 0.005),
             "G4_ejection": met["outside_max"] == 0.0 and met["stray_max"] < 2e-3}
    print("[gates] " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items())
          + f"   (guards={res['guards']}, jitter_rel={met['jitter_rel']:.5f}, drift_rel={drift_rel:.5f}, "
          f"hole={met['hole_frac'] * 100:.2f}% tgt={met['hole_frac_tgt'] * 100:.2f}%, "
          f"outside_max={met['outside_max'] * 100:.3f}%, stray_max={met['stray_max'] * 100:.3f}%)", flush=True)
    gates["drift_rel"] = drift_rel
    return gates


def provenance(args, prm) -> dict:
    root = Path(__file__).resolve().parent.parent
    files = sorted(p for t in TRACKED for p in ((root / t).rglob("*.py") if (root / t).is_dir() else [root / t]))
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:16]
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root,
                                          stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        vf = root / "VERSION"                  # tarball deploys carry the sha in VERSION
        git_sha = vf.read_text().strip() if vf.exists() else None
    return {**vars(args), "mpm": dataclasses.asdict(prm), "git_sha": git_sha, "code_hash": code_hash}


def live_hooks(args, src, tgt, prm, cfg):
    if not (args.live_port or args.live_dir):
        return None, None
    from physmorph.render.covariance import sigma0_from_nn
    from physmorph.viewer.server import LiveServer
    sink = (LiveServer.to_dir(Path(args.live_dir) / f"{Path(args.out).name}_{ARM}") if args.live_dir
            else LiveServer(args.live_port))
    return sink.begin_run(ARM, src, tgt, prm, cfg, sigma0_from_nn(tgt, 0.9))


def prepared(args, seed: int):
    """(the material config, the prepare stage's result) of a run (the multi-body scene, scripts/scene_run.py, prepares
    each body with it)."""
    material = {k: getattr(args, k) for k in ("young", "poisson", "assim") if getattr(args, k) is not None}
    cfg0 = PipelineConfig(loss_follows_n=args.loss_follows_n, **material)
    prep = prepare(args.src, args.tgt, args.n, seed, args.cell_diag, cfg0.young, cfg0.poisson,
                   log=lambda s: print(s, flush=True),
                   loss_ref_n=cfg0.mass_ref_n if cfg0.loss_follows_n else 0, floor=args.floor,
                   surface=args.n // 4 if args.layer_relief and not args.baseline.startswith("xu") else 0,
                   draws=args.render_target_draws,
                   match_density=args.match_density,
                   cell_ref_n=cfg0.mass_ref_n if args.cell_follows_n else 0,
                   mesh=args.render_target == "mesh_shell" or getattr(args, "render_weight_ref", "residual") == "shape",
                   render_tgt=getattr(args, "render_tgt", None))
    return cfg0, prep


def main():
    args = parse_args()
    gpu.require_cuda()
    cfg0, prep = prepared(args, args.seed)
    src, tgt, prm = prep.src, prep.tgt, prep.prm
    if args.drag is not None:
        prm = dataclasses.replace(prm, drag=args.drag)
    if args.f_ext is not None:
        prm = dataclasses.replace(prm, f_ext=tuple(args.f_ext))
    if args.floor:
        prm = dataclasses.replace(prm, floor_friction=args.floor_friction)
    cfg = run_config(args, cfg0, prep)
    run_main(args, prep, src, tgt, prm, cfg)


def run_config(args, cfg0, prep):
    """The run's PipelineConfig from its flags (the frozen recipe at the defaults) and the prepare stage."""
    # the render pictures follow the particle spacing as the transport grid does (prepare.py): above mass_ref_n
    # particles a pixel keeps its size in pitches (D74: at 300k a 64-px pixel was 3.8 pitches, 1.9 at 40k).
    # One resolution, the fine one, from the first window (D89): the coarse stage fitted a picture whose pixel
    # was wider than the detail and left the fine stage a few windows at the end (D73, D81); started fine, the
    # 300k render arm is 13-43 % ahead of its physics-only twin and the 40k gallery is unchanged (D83, D85)
    per_dx = max(1.0, (args.n / cfg0.mass_ref_n) ** (1.0 / 3.0)) if cfg0.loss_follows_n else 1.0
    res = args.render_res_hi or int(np.ceil(cfg0.render_res * per_dx))
    cfg = dataclasses.replace(cfg0, animations=args.animations, patience=args.patience,
                              reject_stop=args.reject_stop, render_weight_scale=args.render_weight_scale,
                              lambda_ema=args.lambda_ema,
                              render_exterior=args.render_exterior, render_res=res,
                              min_spacing=args.min_spacing, spacing_symmetric=args.spacing_symmetric,
                              exterior_radius=args.exterior_radius,
                              exterior_field=args.exterior_field, render_target=args.render_target,
                              u_off=args.u_off, relax_driven=args.relax_driven,
                              gap_split=args.gap_split,
                              gap_vp=((prep.density["target_fill"] if prep.density else prep.v_src) / args.n
                                      if args.gap_split in ("interface", "walls", "nearest", "carry") else 0.0),
                              u_adam_carry=args.u_adam_carry, u_render_smooth=args.u_render_smooth, stop_physics=args.stop_physics,
                              render_normal_radius=args.render_normal_radius,
                              volume_exact=args.volume_exact, assim_volume=args.assim_volume,
                              cell_shape=prep.cell_shape or 0.0,
                              ot_iters=args.ot_iters, ot_warm_trials=args.ot_warm_trials,
                              replay_noise=args.replay_noise,
                              cleanup_in_phys=args.cleanup_in_phys,
                              loss_res=prep.loss_res, unit_ref_res=prep.unit_ref_res,
                              nn_berth_k=prep.nn_berth_k, proximity_berth=args.proximity_berth,
                              render_weight_ref=args.render_weight_ref, placement_at_cell=args.placement_at_cell, settle_windows=args.settle_windows,
                              settle_u=args.settle_u, settle_relax=args.settle_relax, layer_relax=args.layer_relax,
                              render_split=args.render_split, colour_kernel=args.colour_kernel,
                              render_every=args.render_every, max_ls_iters=args.ls_max, stop_tol=args.stop_tol,
                              keyframe_time=tuple(getattr(args, "keyframe_time", ()) or ()),
                              keyframe_geometry=getattr(args, "keyframe_geometry", "own"),
                              keyframe_compare=getattr(args, "keyframe_compare", "silhouette"),
                              render_compare=getattr(args, "render_compare", "exterior"),
                              keyframe_soundness=getattr(args, "keyframe_soundness", "every"),
                              grad_dump=args.grad_dump,
                              profile=args.profile, term_dump=args.term_dump, work_telemetry=args.telemetry,
                              baseline=args.baseline, xu_mass=1.0 / prep.ppc, xu_form=args.xu_form,
                              force_scale=args.force_scale)
    if getattr(args, "cleanup_off", False):  # 29b: the cleanup off (the spray W1 pull, the near band)
        cfg = dataclasses.replace(cfg, w_dt=0.0, w_nn=0.0)
    if args.baseline.startswith("xu"):     # the comparison baseline: Xu et al.'s objective, nothing else of ours
        cfg = dataclasses.replace(cfg, render_weight_scale=0.0, min_spacing=0.0)
    return cfg


def load_keyframes(paths, times):
    """D175: per keyframe time, the keyframe's samples ((N_k, 3) float32 arrays) from its .npz ('samples'). A keyframe
    made from the source by a linear map (its 'A') must not ask for a rotation: the map's polar rotation within 0.5 deg."""
    if len(paths) != len(times):
        raise ValueError(f"--keyframe needs one file per --keyframe_time ({len(paths)} vs {len(times)})")
    out = []
    for p in paths:
        z = np.load(p)
        s = np.asarray(z["samples"], np.float32)
        if s.ndim != 3 or s.shape[2] != 3 or not np.isfinite(s).all():
            raise ValueError(f"{p}: 'samples' must be finite (K, N, 3)")
        if "A" in z.files:
            u, _, vt = np.linalg.svd(np.asarray(z["A"], np.float64))
            ang = float(np.degrees(np.arccos(np.clip((np.trace(u @ vt) - 1.0) / 2.0, -1.0, 1.0))))
            if np.linalg.det(u @ vt) < 0 or ang > 0.5:
                raise ValueError(f"{p}: the keyframe's map rotates the body by {ang:.3g} deg; a keyframe asks for no "
                                 "net rotation")
        out.append(list(s))
    return out


def content_samples(args, prep, tgt, cfg, app):
    """D178 (render_split "masked"): the 3D target's own samples for the content mask (target.content_masks): its sample
    and the draws the run would read without a render target (seeds seed + 1 + 100 k, k = 1 .. D - 1: the pictures T0),
    and D further independent draws (k = 8 .. 7 + D: T0'). None off the rule, and where the pictures are the 3D target's
    own (no render target, no paint: T = T0 and the mask is empty, the rule is Hu's)."""
    if cfg.render_split not in ("masked", "content") or not cfg.render_exterior or (prep.render_x is None and app is None):
        return None
    if args.floor:
        raise ValueError("render_split masked: the content mask's own draws are not lifted onto the floor")
    from physmorph.sampling.mesh import draws_in_frame
    n, D = len(tgt), max(int(args.render_target_draws), 1)
    if prep.render_x is None and prep.tgt_draws is not None:          # the run's own draws are the 3D target's
        own = [tgt] + list(prep.tgt_draws)
    else:
        own = [tgt] + (draws_in_frame(args.tgt, prep.frame, n, [args.seed + 1 + 100 * k for k in range(1, D)])
                       if D > 1 else [])
    indep = draws_in_frame(args.tgt, prep.frame, n, [args.seed + 1 + 100 * k for k in range(8, 8 + D)])
    return own, indep


def run_main(args, prep, src, tgt, prm, cfg):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[v2run] {args.src} -> {args.tgt}  N={args.n}  T={cfg.T}  iters={cfg.iters}  "
          f"anims={cfg.animations} | dx={prm.dx} dt={prm.dt:.5f} smoothing={prm.smoothing}", flush=True)
    print(f"[v2run] baseline chamfer (undeformed) = {metrics.chamfer(src, tgt):.4f}", flush=True)
    out = {"provenance": {**provenance(args, prm), "ppc": prep.ppc, "density": prep.density,
                          "cell_shape": prep.cell_shape},
           "arms": {}}
    cfg_dump = dataclasses.asdict(cfg)
    print(f"\n[v2run] ===== ARM {ARM} =====", flush=True)
    t_thin = time.time()
    # the thin part of the target (measurement), in the shape's cell (D137: also when the MPM cell follows N)
    ts = thin_set_cached(tgt, prep.cell_shape or prm.dx, cfg.mass_ref_n)   # D174: cached (exact)
    print(f"[v2run] thin set: {len(ts.points)} of {ts.n_outer} outer target points below two MPM cells "
          f"({time.time() - t_thin:.1f} s)", flush=True)
    on_commit, on_iter = live_hooks(args, src, tgt, prm, cfg)
    t0 = time.time()
    stride = args.save_F_stride if args.save_F_stride > 0 else cfg.T
    app = None
    if args.appearance:                                     # D160: the scene's albedos (source, target, its draws)
        from physmorph.pipeline.appearance import make_appearance
        tex = None
        if args.appearance == "texture":                    # D173: the textured OBJ (the --tgt mesh's raw coordinates) and its image
            from physmorph.pipeline.appearance import TextureMap
            obj, img = args.texture.split(",")
            tex = TextureMap(obj, img, prep.frame, args.tgt)
        app = make_appearance(args.appearance, src, tgt, prep.tgt_draws, texture=tex)
        out["provenance"]["appearance"] = app.params
    keyframes = load_keyframes(args.keyframe, cfg.keyframe_time) if cfg.keyframe_time else None
    res = run_pipeline(src, tgt, prm, cfg, log=lambda s: print(s, flush=True), on_commit=on_commit,
                       on_iter=on_iter, F_stride=stride, thin=ts, surface=prep.tgt_surface, draws=prep.tgt_draws,
                       mesh=prep.tgt_mesh, appearance=None if args.appearance_placebo else app, render_x=prep.render_x,
                       target_fill=prep.density["target_fill"] if prep.density else None, keyframes=keyframes,
                       content=content_samples(args, prep, tgt, cfg, None if args.appearance_placebo else app))
    seconds = time.time() - t0
    frames, dn = res["frames"], res["deliver_n"]
    delivered = [h for h in res["history"] if h.get("frame_end") and not h.get("null_commit")
                 and h["frame_end"] <= dn]
    detF_min = min([1.0] + [h["Jmin_traj"] for h in delivered])
    met = metrics.summarize(frames.x[:dn], tgt, n_held=res["n_held"], detF_min=detF_min)
    met.update(thin_metrics(frames.x[dn - 1], ts))
    try:                                                    # a measurement; its failure must not cost the archive
        met.update(surface_roughness(frames.x[dn - 1], tgt))
    except Exception as e:                                  # noqa: BLE001  (D122: the first surface-dense 300k run died here)
        print(f"[v2run] surface_roughness failed: {e!r}", flush=True)
        met.update(surf_rough=None, surf_rough_p90=None)
    mv = [h["move"] for h in res["history"] if "move" in h]
    met["move_cv"] = float(np.std(mv) / max(np.mean(mv), 1e-9)) if len(mv) > 2 else float("inf")
    met["move_first_frac"] = float(sum(mv[:3]) / max(sum(mv), 1e-9)) if mv else 1.0
    gates = eval_gates(res, met, prm, cfg.T)
    idx, F_samples = frames.archive_F()
    J_samples = frames.archive_J()                  # D129: the tracked volume at F's frames (empty when not tracked)
    keep = sorted(set(list(range(0, int(dn), 12)) + [int(dn) - 1]))   # D145: the kept frames only, no full archive
    np.savez(f"{args.out}_frames12.npz", frames=np.stack([frames.x[k] for k in keep]), raws=np.asarray(keep),
             tgt=tgt, deliver_n=np.int64(dn),
             **({} if app is None else dict(src_albedo=app.src, tgt_albedo=app.tgt,
                                            appearance=np.str_(json.dumps(app.params)))))
    out["arms"][ARM] = {"config": cfg_dump, "metrics": met,
                        "gates": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in gates.items()},
                        "guards": res["guards"], "converged": res["converged"], "balancer": res["balancer"],
                        "deliver_n": int(dn), "truncation": res["truncation"], "n_held": res["n_held"],
                        "seconds": seconds, "history": res["history"], "content_mask": res.get("content_mask")}
    print(f"[v2run] ARM {ARM}: chamfer={met['chamfer']:.4f}  silIoU={met['sil_iou']:.4f}  "
          f"hole={met['hole_frac'] * 100:.2f}%  jitter_rel={met['jitter_rel']:.5f}  "
          f"detFmin={met['detF_min']:.4f}  move_cv={met['move_cv']:.2f}  "
          f"first3={met['move_first_frac'] * 100:.0f}%  ({seconds / 60:.1f} min)", flush=True)
    print(f"[v2run] thin: uncovered {met.get('thin_uncovered', float('nan')) * 100:.1f}% (world "
          f"{met.get('thin_uncovered_world', float('nan')) * 100:.1f}%) of {met.get('thin_n', 0)} thin outer points",
          flush=True)
    Path(f"{args.out}.json").write_text(json.dumps(out))
    print(f"\nsaved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
