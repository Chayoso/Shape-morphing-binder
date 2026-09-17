# PhysMorph pipeline — what runs, in what order, where (2026-09-16)

One page for the current production path (branch `v3-grid-gs`). Equations are in
[method.md](method.md); the result log and every gate in [experiments.md](experiments.md);
the viewer in [viewer.md](viewer.md).

## 1. Stages of one run (`scripts/pipeline_run.py`)

```
 mesh (assets/*.obj) ─► sampling ─► discretisation ─► target pack ─► calibration
        │                 │              │                │               │
   load_normalized   sample_volume    derive()       build_target    calibrate_units
   (bbox diag 8 wu)  (axis fills,     dx from N,ppc  loss grid (dx), (density units:
                     jittered voxel   dt, domain     W1 distance      legacy/density
                     centres)         box, loss_res  transform, nn    ratio on a 0.5 wu
                                                     band, render     reference cell)
                                                     targets (18 views)
                                                          │
                           ┌──────────────────────────────┘
                           ▼
   window loop (runner.run_pipeline → optimizer.optimize_window), 300 windows × T=20 steps
     start state x, v, C, F, Fp, Fg  ─► ControlBasis leaf dFc (per particle or node grid)
     ├─ terms():   tape rollout (PersistentAdjoint: forward + adjoint as CUDA graphs)
     │             losses: D_vol (density units) + kinetic (end + running + variance)
     │                     + W1/dt + nn band + render (silhouette + PBR, λ-balanced)
     │             gradients gp / gdt / gr → PCGrad/λ combine (→ optional Sobolev H1)
     ├─ line search over the Adam step: eval_terms() = persistent no-grad trajectory
     │             (CUDA graph), Armijo + noise floor + state sanity (det F, finite);
     │             an exhausted search ends the window
     └─ commit rollout (same no-grad trajectory) → frames, F_seq, end state
   runner commit:  F repair (condition_F) → plastic assimilation (torch, isochoric band)
                   → guards (clamp / nan / F flips) → outer-merit trust gate (brake,
                   reversal, patience) → archive frames (stride) + live packets → history
                                                          │
                                                          ▼
   deliverable:  <out>_<arm>.npz (frames, F, meta), <out>.json (history, metrics, gates),
                 live/<out>_<arm>/commits/*.bin (3-D viewer), <out>.log
```

Key flags of the production recipe (`render_full_dt_iso_nn` arm):

```
--n 150000 --cell_diag 26 --loss_units density --warm_start --w_kin 5 --w_kin_var 200
--animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000
--bonds --domain auto --archive_stride 8 --live_dir <OUT>/live
```

| flag | meaning | where |
|---|---|---|
| `--cell_diag 26` | the MPM cell follows the SHAPE: dx = source bbox diagonal / 26 (0.31 wu on the 8 wu normalisation), ppc = N dx³ / V (25 at 40k, 91 at 150k). The mass-ejection ladders (2026-09-17, docs/method.md §10.9) showed the ejection variable is the cell size relative to the shape: dx 0.20 fractures at ppc 8 and 27 alike, 0.31 holds at ppc 27 and 91 alike, 0.41 holds but loses detail. `--ppc` keeps the old contract (dx follows N). Loss grid = MPM cell in density units | `scripts/pipeline_run.py`, `mpm/discretisation.py` |
| `--loss_units density` | resolution-invariant D_vol; legacy weights converted on a 0.5 wu reference cell (`cfg.unit_ref_res`) | `pipeline/runner.py::calibrate_units` |
| `--domain auto` | MPM grid = leash box (1.25 × max|target|) + 2 dx; the reference cell stays 0.5 wu (fix 6cf1a23) | `scripts/pipeline_run.py` |
| `--bonds` | material re-coupling v5: frozen source kNN, rest lengths as state, fragment mask on the dilated occupancy | `mpm/kernels.py`, `pipeline/runner.py::fragment_mask` |
| `--warm_start` | safeguarded warm start of dFc from the previous window | `pipeline/optimizer.py` |
| `--w_kin / --w_kin_var` | end kinetic + per-particle kinetic variance (coherent motion) | `optimizer.phys_core` |
| `--archive_stride 8` | every 8th step archived (150k memory) | `pipeline/runner.py` |
| `--no_outer_merit` | disables the trust gate (diagnostics only — the gate stopped C's runaway) | `pipeline/config.py` |
| `--grad_h1`, `--assim_consensus`, `--control_grid`, `--eject_veto`, `--w_esc`, `--v_max`, `--continuity` | mechanism trials of the ejection ladder (all falsified; kept for A/B) | `docs/experiments.md` |

Speed switches: `PHYSMORPH_TIMING=1` prints a per-window breakdown (eval rollouts / losses /
det check / tape forward / the three adjoint passes / commit rollout);
`PHYSMORPH_NO_ADJ_GRAPH=1`, `PHYSMORPH_NO_LS_BREAK=1` bisect the two speed passes.

## 2. Where things live on hyde06 (everything under `/data`)

```
/data/relcfd/chayo/physmorph_v2/
  repo/        the deployed code (tar of the local repo minus assets/legacy/output; VERSION = commit)
  output/      <run>.log, <run>.json, <run>_<arm>.npz, live/<run>_<arm>/ (viewer packets),
               report150/<T>/, report_new40/<T>/ (post-processed media), rcp_ladder_status.log (markers)
```

Nothing is run from or written to `$HOME` (user rule 2026-09-16). Python:
`/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python`. GPUs 0 and 2 only, thread caps 8,
launches staggered ≥ 45 s, `setsid nohup … < /dev/null &`, one 150k run per GPU.

## 3. Ops scripts (`scripts/ops/`, copied to the server by `deploy.sh`)

| script | runs on | does |
|---|---|---|
| `deploy.sh` | local | tar the tracked repo (minus assets/legacy/output) → scp → extract into `repo/`; prints VERSION |
| `hyde06_env.sh` | server | `REPO`, `OUT`, `PY`, thread caps — sourced by every server script |
| `run_batch.sh <gpu> <n> <prefix> <targets…>` | server | the production recipe per target (`<prefix>_<T>`), markers `<PREFIX> <T> DONE` |
| `post_run.sh <prefix> <target> <gpu>` | server | surface video, particle GIF, PBR stills (delivered + target, az 35/215), scatter probe, stray census, loss curves, ARM/gates, fragment count → `report_<prefix>/<T>/` |
| `watch_post.sh <prefix> <targets…>` | server | post-processes each target once its DONE marker appears |
| `run_trial.sh <gpu> <name> <target> <flags…>` | server | one mechanism trial (40k recipe + extra flags), marker `RT <name>_<T> DONE` |
| `timed150.sh <tag> <gpu> [flags]` / `profile150.sh <tag>` | server | 6-window 150k timing (`PHYSMORPH_TIMING=1`) / cProfile |
| `fetch_report.sh <prefix> <targets…>` | local | scp `report_<prefix>/<T>/` into `output/report_<prefix>/` |
| `build_report150.py` | local | `output/report150/` → `index.html` (GIF→MP4 via ffmpeg, JPEG stills) + `docs/highres150_report.md` |
| `gpu_pick.sh` | local | GPU placement rule (free GPUs on hyde06) |

Probes (`scripts/probes/`): `stray_census.py` (far particles at the end), `fragment_trace.py`
(neighbour-distance history of the end fragments), `scatter_probe2.py` (thin-target sparsity),
`loss_curves.py`, `grad_field.py`, `gate_probe.py`, `web_probe.py`, `ear_views.py`, `cover_diff.py`.

## 4. Post-processing and the report

`post_run.sh` renders the **object, not particles**: `scripts/render_surface_video.py` (GPU
z-buffer disk splats sized by the local spacing → smoothed depth → normals → GGX, two
azimuths, target outline) and `scripts/render_pbr.py` (CPU still, `--target` for the
reference). Metrics never consume the renderer (chamfer, silIoU, hole, fragments, census are
raw-state numbers). `build_report150.py` assembles the page; the artifact is republished with
the same file path (`output/report150/index.html`).

## 5. Viewer

Every run with `--live_dir` writes commit packets; `scripts/viewer_serve.py --root
<OUT>/live --port 8765` serves them (tunnel: `ssh -J chayo@hyde01.dabh.io -L
8765:127.0.0.1:8765 chayo@hyde06.dabh.io`, then `http://127.0.0.1:8765/`).

## 6. Rules that shaped this pipeline

- Rendering controls physics: the image only ever changes physical quantities (control
  stress, material, plastic rest state); no geometric post-ops.
- No parameter-only fixes: a defect is removed by a mechanism whose constants come from the
  discretisation or the physics, never tuned per shape.
- Refute before shipping; state the discretisation with every number; per-frame visual QA of
  rendered deliverables; never gate a commit on `pytest | tail`.
