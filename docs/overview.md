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
`docs/SPEC.md`); [experiments.md](experiments.md) — gates, metrics, result log;
[related_work.md](related_work.md) — the papers each design choice leans on.

**Layout**

```
physmorph/
  mpm/        MLS-MPM engine (warp kernels, tape trajectory, torch bridge, F repair)
  pipeline/   config / render_loss / grid_smooth / optimizer / runner / runner_vbd
  vbd/        quasi-static grid block-descent solver (torch)
  plasticity/ assimilate_elastic (exact stretch relaxation)
  losses/     d_vol (eq 13), soft silhouette primitives (eq 14)
  metrics.py  loss-independent gate metrics
  render/ sampling/ viewer/   3DGS raster + covariance, mesh sampling, PLY export
scripts/      pipeline_run (arms+gates), grad_analysis, probe_gs_differentiability,
              quicklook, make_gif
tests/        42 CPU/warp-CPU tests incl. end-to-end smokes of BOTH families
legacy/       the C++ oracle (Xu et al. DiffMPMLib3D) — untouched reference
```

**Workflow**: all simulation runs on hyde06 (`ssh -J chayo@hyde01.dabh.io
chayo@hyde06.dabh.io`, repo `~/physmorph_v2`, python
`/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python`); local machine = code + `pytest`
only (AGENTS.md rule 1). Adversarial verification (Codex gpt-5.6-sol xhigh + Claude Opus,
REFUTE mode) before anything ships; every reported number carries its discretisation.
