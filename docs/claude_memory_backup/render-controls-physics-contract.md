---
name: render-controls-physics-contract
description: "2026-09-14 session — the render-controls-physics contract (control basis, geometric F_g, running kinetic, Chebyshev covector smoothing, grad modes, density units, discretisation contract), where it lives, and that the hyde06 ladder has NOT run (ssh key rejected)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-15T01:08:35.747Z
---

On 2026-09-14 the user asked for a full pass on ten problems (render/physics gradient
gap, holes at low N, grid dependence, gradient conflict, persistent live viewer,
VBD/Chebyshev propagation, PhysGaussian/PhysDreamer gradient flow, C++ parity,
oscillation, 2022-2026 literature; PhysMorph-GS excluded) under the fixed premise
"rendering controls physics". The previous session's work (response_control trust-region,
geometric.py, gradient_flow_audit) had been REVERTED by the user in commit f7cde6a; the
audit's findings (F-route changes the image at zero motion; smoothing branch dropped the
F covector; C++ adjoint has sG / missing (1-s) terms; 12k per-entry FD of the image
gradient fails while directional passes) were re-used, the code was not.

What now exists (all opt-in, legacy path bit-identical when off):
- `docs/render_controls_physics.md` — master design + literature + pre-registered ladder.
- `physmorph/pipeline/control_basis.py` (`control_grid`, `control_tknots`): dFc on a
  coarse trilinear node grid x time knots — the literature's low-dim field.
- `mpm/kernels.k_geom_update`, `Trajectory(track_geom)`, `function.warp_mpm_ext` (5
  outputs incl. Fg_T and all v_t); `render_F_geom` renders Sigma from F_g.
- `w_kin_running`, `grad_project_mode` (render/phys/cagrad/blend in `grad_combine.py`),
  `render_gs_cheb` (Chebyshev, both covectors), `gauss_robust_eps`, `loss_units=density`
  (unit ratio converts eps/target_norm/noise floor AND every fixed weight; relative cap).
- `mpm/discretisation.py` + `pipeline_run.py --ppc`; arms `render_ctrl`,
  `render_ctrl_gauss`, `render_ctrl_first`; `--live_dir`.
- Viewer: `viewer/filehub.py`, `scripts/viewer_serve.py`, `scripts/viewer_tunnel.py`,
  `docs/viewer.md`. Triage: `scripts/probes/oscillation_triage.py`, `docs/oscillation_triage.md`.

REFUTE round done 2026-09-15 (Opus, 12 findings: measured density unit ratios, LS basis projection, fixed-beta blend, --ppc keeps loss units, c2f keeps gauss/kde scales, Chebyshev guard) — `docs/reviews/refute_rcp_opus_20260915.md`; committed on v3-grid-gs. hyde01 still rejects the ed25519 key (2026-09-15 08:40): the user must add `~/.ssh/id_ed25519.pub` to authorized_keys on hyde01 AND hyde06 (they were logged in interactively).

**hyde06 ladder results 2026-09-15 (20k, T=20, dx=0.5, loss_res 64, pace 0, w_kin 0.5):**
baseline 0.1599/0.9655; render_ctrl 0.1617/0.9533 (tie, detFmin 0.75->0.88); gauss 0.1621/0.9566;
first 0.1626/0.9569; density units FALSIFIED (silIoU 0.897); --ppc 8 + density 0.1552/0.9625
(best chamfer); grid sweep 6<12<24 monotone (24: 0.1597); dfc_clip 0 collapses BOTH families.
Oscillation = window-locked CONTROL limit cycle (speed 0.47->0.10->0.45 inside every window,
95% power at period T; not stiffness, not volume) - triage rule extended (modulation>2);
remedy batch d (w_kin_var 10/50, w_kin 5, w_kin_running 10) launched 09:2x CDT.
Results table: docs/experiments.md "2026-09-15" section. Deploy = tarball to ~/physmorph_v2
(backup pre_rcp_20260915.tgz); viewer_serve on 8765 + local viewer_tunnel.py running.

Batches d-f verdict (2026-09-15): oscillation cause = per-window terminal-only objective (push-and-
return is free), NOT cold start (warm start leaves power 0.90); w_kin_var 200 kills the window lock
(power 0.95->0.06) at zero shape cost; warm_start+w_kin5+kv50 -> visible 0.3%. Basis arms need a
higher w_kin_var (grid24+kv50 still power 0.81). Candidate recipe: flagship + --w_kin 5 --w_kin_var 200
(+ --warm_start); batch g (ws+w_kin5+kv200): power 0.04, visible 1.9%, chamfer 0.1596 - lock gone. Ladder COMPLETE (14 arms); recipe recommended, NOT made default (needs 40k replicate + REFUTE).

40k replicate (batch h, 2026-09-15): base 0.1302/0.9626; ws+w_kin5+kv50 0.1296/0.9676 power 0.29;
kv200 0.1304/0.9621 power 0.07; --ppc 8 (dx 0.271, density units) 0.1164 (-10.6%) but oscillation 2x
(needs w_kin_var 200-500 in density units); basis 24^3+kv200 0.1301/0.9570. F_g total deformation
rendered NEEDLE splats -> relax_stretch at commits (502b543). REFUTE #2 (adoption) launched.

REFUTE-2 (2026-09-15 evening, docs/reviews/refute_rcp2_opus_20260915.md, 19 findings) answered in
e234248: commit-time F_g relaxation WITHDRAWN (premise violation) -> render-model stretch saturation
gauss_cov_sat; density-unit conversion now vs a fixed reference grid (unit_ref_res 64; the ppc8 arms of
batches h/i ran weights 0.31x); probe driver renamed C_window + tortuosity + 6% lock band; recipe shape
claims downgraded to "tie within single-seed noise". Batch j (attribution T=10/40, --assim 0, seeds 2/3,
code-matched 40k pair, ppc8 legacy vs density-ref) launched ~11:30 CDT; results -> experiments.md.

Batch j DONE (2026-09-15 ~12:00 CDT): period follows T (10/20/40), survives --assim 0 -> control
re-optimisation confirmed; seeds 1-3: recipe = tie within noise (chamfer std 0.45%), oscillation removal
seed-invariant; code-matched 40k pair tie; --ppc 8 decoupled: fine MPM dx -> silIoU 0.9757 (+1.1pt),
fine LOSS grid (149^3, density ref-64) -> chamfer 0.1142 (-12%). Recommended next flagship candidate:
--ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 at 40k (needs a REFUTE pass on the
batch-j section). All 43 archives re-triaged. Viewer server + tunnel left running.

Thin-feature "scatter then return" (2026-09-15 afternoon, docs/thin_feature_transport.md): NOT volume, NOT
the nn/W1 pulls (ablation) - the core mass descent flings surface particles into empty ear cells (Xu 2025's
own "ejection = quickest descent"), then numerical fracture keeps them apart; thin targets end 15-25% under-
filled. Levers measured: control basis (best: 0.180 delivered / sparse 0.28) and --ppc 8. Implemented from
the literature (Luiten 2023 rigidity, Yue 2015, Yao-Zhao 2026, Hsieh-Charon): w_coh (Laplacian), w_bond
(one-sided per-window bond stretch), vol_frontier (frontier-masked D_vol). Batch l (w_coh) running, batch m
(bond/frontier) staged; pre-registration in the dossier §4. Probe: scratchpad scatter_probe2.py (server copy /tmp).

**Why:** the user's standing rules (papers before design, structural fixes not knob
tuning, adversarial gates) and the audit made a basis-level control + kinematic F the
defensible answer; nothing was validated at scale because ssh to hyde01 (jump) rejected
the ed25519 key all day (JumpCloud sync, see [[hyde06-ops-etiquette]]).

**How to apply:** next session must (1) get the key re-registered by the user, (2) run the
§10 ladder in docs/render_controls_physics.md on hyde06 (commands in docs/experiments.md),
(3) run the triage probe on a real archive before claiming anything about the
oscillation, (4) REFUTE review (Codex + Opus) before adoption. Do not re-open the
trust-region/finite-difference surface control the user reverted.
