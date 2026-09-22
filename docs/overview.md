# PhysMorph v3 — overview

*(rewritten from scratch 2026-09-01 on branch `v3-grid-gs`; the v1/v2 doc set lives in git
history ≤ `f0b31e9`.)*

**Thesis.** A source shape is carried to a target by *real* elastodynamics, and
**differentiable render feedback makes the morph qualitatively and measurably better than
3-D supervision alone** — while touching only physical quantities (control stress, material,
plastic rest state, terminal velocity). No displacement injection, no geometric post-ops.

**Two families, one measurement harness:**

| family | motion | render coupling | code |
|---|---|---|---|
| **dynamic** (v2) | T-step MLS-MPM rollouts, control sequence dFc[t] | terminal loss → tape adjoint → dFc (+ per-particle λ,μ) | `pipeline/optimizer.py`, `pipeline/runner.py` |
| **quasi-static VBD-MPM** (v3) | per-commit grid equilibrium (colored block descent) | render is an ENERGY term of the equilibrium | `vbd/solver.py`, `pipeline/runner_vbd.py` |

Both share: the MPM engine (`mpm/`), target pack + λ balancing (`pipeline/render_loss.py`),
exact elastic-stretch plastic assimilation (`plasticity/assimilation.py`), guard counters,
metrics (`metrics.py`) and gates (`docs/experiments.md`).

**Arms** (`scripts/pipeline_run.py --arms …`): `phys` (mass-only baseline, same code path),
`render` (headline), `render_mat` (+ material field), `render_ws` (+ safeguarded warm
start), `render_gs` (+ Sobolev render direction), `render_pbr` (+ Lambertian shading
channel), `render_pc` (+ PCGrad conflict projection), `render_c2f` (+ coarse-to-fine
targets), `render_pace` (+ paced trajectory), `render_full` (pbr+pc+c2f+pace+clip).
The quasi-static VBD family is retired to `deprecated/`.

**Docs**: [method.md](method.md) — equations + formulations (the file code cites as
`docs/SPEC.md`); [pipeline.md](pipeline.md) — the production path stage by stage, flags,
server layout under `/data`, ops scripts; [experiments.md](experiments.md) — gates, metrics,
result log; [related_work.md](related_work.md) — the papers each design choice leans on.

**Layout**

```
physmorph/
  mpm/        MLS-MPM engine (warp kernels, tape trajectory, torch bridge, F repair,
              geometric F_g, discretisation contract)
  pipeline/   config / render_loss / grid_smooth (Chebyshev) / control_basis /
              grad_combine / optimizer / runner
  vbd/        quasi-static grid block-descent solver (torch)
  plasticity/ assimilate_elastic (exact stretch relaxation)
  losses/     d_vol (eq 13), soft silhouette primitives (eq 14)
  metrics.py  loss-independent gate metrics
  render/ sampling/ viewer/   3DGS raster + covariance, mesh sampling, PLY export,
              file-backed live sink (filehub) + live/quad/compare pages
scripts/      pipeline_run (arms+gates), viewer_serve / viewer_tunnel (persistent
              monitor), probes/oscillation_triage, grad_analysis,
              probe_gs_differentiability, quicklook, make_gif
tests/        42 CPU/warp-CPU tests incl. end-to-end smokes of BOTH families
legacy/       the C++ oracle (Xu et al. DiffMPMLib3D) — untouched reference
```

**Workflow**: all simulation runs on hyde06 (`ssh -J chayo@hyde01.dabh.io
chayo@hyde06.dabh.io`, repo `~/physmorph_v2`, python
`/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python`); local machine = code + `pytest`
only (AGENTS.md rule 1). Adversarial verification (Codex gpt-5.6-sol xhigh + Claude Opus,
REFUTE mode) before anything ships; every reported number carries its discretisation.


Problem dossiers (2026-09-02): `docs/floaters.md` (floating gaussians —
four populations, mechanism history, papers, v1-vs-v2), `docs/oscillation.md`
(near-optimum unrest — four drivers closed, v1 global damping vs v2 cause removal).

**2026-09-14 — the render-controls-physics contract** ([render_controls_physics.md](render_controls_physics.md)):
the answer to the ten-question pass (gradient gap, holes, grid dependence, conflict,
viewer, VBD/Chebyshev, PhysGaussian/PhysDreamer, C++ parity, oscillation, 2022–2026
literature). New, all opt-in: control on a coarse node basis (`pipeline/control_basis.py`),
render covariance on the geometric F_g (`mpm/kernels.k_geom_update`, `function.warp_mpm_ext`),
running kinetic term, Chebyshev-accelerated grid propagation of BOTH render covectors,
gradient-combination modes (`pipeline/grad_combine.py`), density loss units, the
discretisation contract (`mpm/discretisation.py`, `--ppc`), arms `render_ctrl*`.
Companions: [viewer.md](viewer.md) (persistent multi-run monitor + tunnel keeper),
[oscillation_triage.md](oscillation_triage.md) (driver classification probe). The hyde06
ladder (§10 there) has not run yet.

## State (2026-09-22; branch `v3-grid-gs`)

**Recipe (frozen; `scripts/ops/hyde06_env.sh`):** `render_full_dt_iso_nn` at 40k particles with
`--cell_diag 26` (MPM cell = shape diagonal / 26 = 3.6 spacings), density-unit cell-sum loss,
kinetic recipe, `--bonds`, `--domain auto`, plus the outer-layer relaxation projection, the
position-mode control channel u, the denoised shading reference and stratified sampling
([method.md §10.9–10.14](method.md)). 150k is excluded from the experiments until re-opened.

**Deliverable surface:** the outer particle layer → same-side plane pulling → screened Poisson →
the mass rule, with the exterior test that drops surfels sitting on interior density steps
(§10.12–10.13). Every frame of every gallery video is counted (pieces, bridges, cavities).

**What is established** ([surface_gradient.md](surface_gradient.md) §6–§14, [experiments.md](experiments.md)
2026-09-19 … 22): the render channel is a third of every accepted control update (g_share
0.36–0.40, deterministic per window) and moves the outcome 2–8 run-to-run spreads (silhouette
+0.6 … +2.0 points; on the fixed bunny target +1.9, normal error −1.4°, detail correlation
+0.08); through the stress control it carries the outline at the cell scale, below the cell
the kinematic u channel acts; coupling u to the physics (through F), gating it or driving it
by the render channel alone are all worse (the candidate round). The stratified cloud has no
shot noise (§9, proved and measured). The volume fill of non-watertight meshes is fixed
(§14, `fill_check.py`). Every report carries a "rendering influence" block by standing rule.

**Server:** hyde06 under `/data/relcfd/chayo/physmorph_v2` (`repo/`, `output/`; `scripts/ops/
hyde06_env.sh`); GPUs 1 and 3 belong to another user. The final 40k gallery is `g40`
(`output/report_g40`, `scripts/ops/gallery40.sh` on the server; builder `build_report150.py`).
