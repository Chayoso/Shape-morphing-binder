# AGENTS.md — settled-base

Shared guide for agents (Codex, Claude Code) working on this branch.

## What this branch is

`settled-base` starts from `michael/settled-transport` (835af64, Michael Jin, 2026-09-29), which is the public
release (`origin/main` at 680622e) plus the settled-transport method. On top of it, the renderers and measurement
tools were brought from `v3-grid-gs` (2026-09-29, see `docs/experiments.md` S1). `v3-grid-gs` keeps the full history
of the earlier line: its `docs/experiments.md` holds the diagnoses and the comparison that led here (entries B1, B2).
The two branches have no common ancestor; move code between them by file, never by merge.

The settled-transport method, in one paragraph: each window drives the body for `T` MPM steps with the controls
(`dFc` stress increments, `u` outer-layer normal offsets), then releases them for another `T` steps with the physics
still running, and evaluates every loss at the released end. The physics objective is a debiased Sinkhorn divergence
to the FIXED target on the loss grid plus the residual drift `(T dt v)^2`; a local particle-support term is bounded by
the transport energy (`E + E B / (E + B)`); the render weight is calibrated once and then fixed; one merit drives
progress, acceptance and delivery. See `README.md` ("Settled transport with local particle support").

## Rules

- Run Python only on hyde06, never locally. Local work is editing, git and reading.
- Pre-register each experiment's prediction and gates in `docs/experiments.md` before reading its result; record
  refutations there. The log is append-only. Stamp entries with the server's clock (CDT).
- Fixes are algorithmic. Constants come from the discretisation or the physics, never tuned per shape.
- A change is adopted only when it holds on the 19-target 40k gallery and at 300k, with per-target tables.
- Every result report states the render influence: the render share of the control update, the render-off twin
  (`--render_weight_scale 0`) against the seed-to-seed spread, and what the render does not change.
- When server results exceed 100 GB, clean up superseded runs; archive their logs first and list what was deleted.
- Commit messages end with the co-author line the session specifies; push after each commit.

## Server workflow (hyde06)

- Connect with `ssh hyde06j` off campus. All four GPUs may be used when free; never kill other users' processes.
  Kill your own processes from a server-side script with bracketed patterns (`pkill -f "tag_chai[n]"`): the ssh
  command line itself contains the tag and a plain pattern kills the session.
- `source scripts/ops/hyde06_env.sh` sets `REPO=/data/relcfd/chayo/physmorph_v2/repo_settled`,
  `OUT=/data/relcfd/chayo/physmorph_v2/output/settled`, `PY`, `RECIPE` and `SETTLED`. `repo/` and `output/` at the same
  level belong to `v3-grid-gs`; leave them alone.
- Deploy with `git archive settled-base` into `repo_settled` (it is not a git checkout).
- A run: `CUDA_VISIBLE_DEVICES=g $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/bunny.obj
  --n 300000 --seed 97 $RECIPE $SETTLED --out $OUT/<tag>_bunny > $OUT/<tag>_bunny.log 2>&1`.
- The 4K deliverable render: `PYTHONPATH=$REPO $PY scripts/render_splat_photoreal.py <npz> <out.mp4> --width 3840
  --height 2160 --stride 12 --fps 20 --azimuth 35 --elevation 18 --frames-dir <dir>`. Quick splat video:
  `scripts/render_splat_gpu.py`. Both accept settled archives, which carry no pins.
- Measurement probes and the gradient check: `scripts/probes/settled/README.md`.

## State (2026-09-29, 300k bunny, seed 97; `v3-grid-gs` docs/experiments.md B1, B2)

| | settled (this branch) | v3-grid-gs form | release legacy |
|---|---|---|---|
| silIoU | 0.9851 (0.9850 with reject_stop 20) | 0.9769 | 0.9646 |
| wall / windows | 7 min / 31 | 12 min / 47 | 28 min / 183 |
| det F min / stray particles | 0.934 / 0 | 0.608 / 30 | 0.581 / 40 |
| late surface motion (spacings per frame) | 0.007 / 0.009, no pins | 0.128 / 0.113 unpinned | 0.049 / 0.080 |
| window-to-window reversals | 12 % | 0 % (pinned) | 70 %, all of the second half |
| ear tip (reference particles, gate 13) | 11.1 (12.1 with reject_stop 20) | 8.8 | 4.3 |
| target surface farther than 1.5 spacings | 11.7 % (10.4 %) | 7.8 % | 23.5 % |
| centre-of-mass drift over the morph | <= 0.02 spacings | 0.6-0.8 spacings | 0.03 spacings |

Gradients on the settled path are verified (the branch's tests pass; finite differences match autograd within 3 %
at windows 1 and 20). The render share of the control update grows from 0.33 to 0.90 over the morph; the render-off
twin loses 0.018 silIoU and doubles the surface error, 8x the seed spread.

## Open

- Head and body relief: the v3-grid-gs form still sits closer to the target surface there (7.8 % against 10.4 %).
- The ear-tip gate (13 reference particles) is not met.
- Only the bunny is tested: the 40k gallery and the 300k dragon are the adoption gates.
- `--support_weight 8` and `--ot_iters 1600` are validation values, not derived constants.
- The outer layer's relaxation and `u` move positions without velocity (11-22 % of that layer's motion); `u` is
  driven almost entirely by the render gradient.
- After convergence a rejected candidate can repeat identically each window until the patience runs out.
- `--mass_ref_n` defaults to 40000 here; Michael's 100k validation used unit masses (`--mass_ref_n 0`).
