# Experiment log — settled-base

Append-only. Stamps are the hyde06 server clock (CDT). The earlier line's log is `docs/experiments.md` on `v3-grid-gs`;
the comparison that led to this branch is there as entries B1 (paired 300k runs) and B2 (gradients, render influence,
momentum).

**2026-09-29 11:40 CDT — S1: the branch.** `settled-base` = `michael/settled-transport` (835af64, on the release
680622e) + from `v3-grid-gs` (at fd8c865): the 4K PBR splat renderer `scripts/render_splat_photoreal.py` and the quick
splat renderer `scripts/render_splat_gpu.py`, their modules (`physmorph/render/` `covariance_torch`, `studio`,
`settled`, `surface_gaussians`, `footprint_policy`, `footprint_diagnostics`; `photoreal` and `support` replaced by the
`v3-grid-gs` versions, which contain this branch's versions unchanged plus `render_3dgs_torch`, `live_support`,
`MaterialShadingNormals` and the filters; `knn_gpu` kept as this branch's, which already has `knn_self_torch` and
Michael's early-return fix), their tests, `scripts/ops/hyde06_env.sh` (pointed at `repo_settled` and `output/settled`,
with `$SETTLED`), and the measurement probes under `scripts/probes/settled/`. Code changes to this branch's own
files: both renderers accept archives without pins (no appearance latch engages); `--render_weight_scale` (config
`render_weight_scale`, default 1) multiplies the render weight wherever it is set, so the render-off twin is
`--render_weight_scale 0` (settled mode refuses `--lambda_auto 0`); `--reject_stop` exposes the existing config field.
With the defaults, the pipeline's behaviour is unchanged.

**2026-09-29 11:45 CDT — S2 pre-registered: the adoption gates.** Seed 97 everywhere, the same recipe, each arm from
its own snapshot on hyde06 (`repo_settled` = this branch; `repo_v3snap` = `v3-grid-gs` at fd8c865). (a) The 40k
gallery, 19 targets (A, armadilo, beast, bimba, bob, bunny, cheburashka, C, cow, dragon, fandisk, heart, homer,
maxplanck, nefertiti, ogre, spot, teapot, V): settled (`$RECIPE $SETTLED`) against the adopted 40k form of
`v3-grid-gs` (g41pw: `$RECIPE --ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_arrived --ctrl_rprop_hold_onset --u_rprop
--u_rprop_floor 0 --settle_pin --settle_pin_assim --settle_pin_slip`). (b) The 300k dragon: settled against the bo300
form of `v3-grid-gs`. Per target: silIoU, chamfer, det F min, stray particles, wall and windows, the target surface
beyond 1.5 spacings, window-to-window reversals (second-half share and longest streak), centre-of-mass drift. Gates for
adopting settled as the production form: on every target silIoU ≥ the `v3-grid-gs` form − 0.005, no stall (the run ends
by convergence or plateau, not by a stuck arrival), second-half reversal share ≤ 25 %, centre-of-mass drift ≤ 0.1
spacing; the dragon's silIoU ≥ the paired `v3-grid-gs` dragon − 0.003. Refuted target by target; a failing target is
reported with its mechanism, not averaged away.
