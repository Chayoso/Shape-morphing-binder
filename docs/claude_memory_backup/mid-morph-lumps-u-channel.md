---
name: mid-morph-lumps-u-channel
description: "2026-09-22 evening — the lumpy mid-morph surface was the u channel; fixed by the transport gate (--layer_gate_ot, in RECIPE); gallery re-run as g41; refuted levers listed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-23T23:38:12.018Z
---

The g40 videos were lumpy mid-morph (outer-layer roughness 1.8–2.4× the end value at frames
15–135). Cause: the u channel (position-mode, per-particle) acting on material still in transit —
proven by the render × u factorial twins' roughness traces (u on = +30–50 %; the render channel
without u within the twin spread). Fix adopted 2026-09-22 20:30: `--layer_gate_ot` — u acts only
on layer particles whose remaining OT transport is within one MPM cell (recomputed per window in
the `ot_pace` block). Roughness −16…−21 %, peaks −14…−31 %, end silIoU −0.04…−0.2 (u off would be
−22…−30 % at −0.3…−0.6; the end surface vs the true mesh is the same for all, like for like). RECIPE in hyde06_env.sh carries the flag; the
gallery is re-run as g41 (`$OUT/gallery41.sh`).

**Resolved (2026-09-22 22:00):** g41 gallery = roughness −6…−29 % on 18/19, silIoU ±0.3. nefertiti
stopped early under the gate (anim 16, 0.935: the arriving front spills outside the outline, d_dt
+65 % in one window, and the full-merit catastrophe brake rejected it 3× — g40's u-in-transit had
been masking the spill). Fix (§15f, b8cd7db): the brake reads the PRIMARY objective (ot_div) alone
→ nefertiti 0.9676, 90 windows, roughness −25 %. Refuted on the way: brake on all physics
components (398d216, d_dt still trips it), the normal-component gate (`--layer_gate_ot_normal`,
§15e: no gate, bunny back at g40's roughness) — do not retry either. C (hole topology, `ot`
regime) is ungated. g41 videos/post run via `$OUT/gallery41.sh`
(G41_DONE). Interim page: artifact "40k 갤러리 g41" (output/g41_page/index.html).

**REVERTED 2026-09-23 (user):** after watching g41 the user judged the g40 videos better ("a frame
jumps and a connecting part is wiped at once") and asked to go back to the GitHub version: RECIPE
without --layer_gate_ot, full-merit brake, per-frame videos (TRACK=0). The gate/brake/tracking
stay as flags. The wipe was the tracked surface's re-mesh (measured on 38 videos with
`scripts/probes/video_jumps.py`: re-mesh frames change 3–9× more than others while particles
move as usual), plus stretched-triangle webs between growing ears ("messy triangles"). Fixed
in render_photoreal (`--track_keep --track_stretch 2 --track_every 0`, commits 2ec467d/e89a180/
a35c2e4): re-mesh frames now change no more than ordinary ones. Speed: the merit divergence
warm-starts (verified, −0.8 s/window), `--prefetch` 2.6× per video. 2026-09-23 09:40: the user chose
g41 ("일단 g41로 가자") after the comparison page — RECIPE has the gate again, the brake reads the
primary objective, TRACK=1 (fixed policy) is the video default; the 19 g41 videos are being
re-rendered and the g41 page republished. The user will judge from the videos next.

**Speed / 300k (2026-09-23 evening):** result-identical speed passes (GPU k-NN via Warp hash grid
`render/knn_gpu.py`, device-side layer/gate/census/det/SVD, sample cache, uncompressed archives
≥100k; 3-window 300k test 102→67 s). Found and fixed a real discretisation bug: unit particle
masses made the 300k body 7.5× sluggish (`mass_ref_n=40000`, config; 40k bit-identical). 300k
bunny now: silIoU 0.9677 at 176 windows / 44 min under host load; at 70 windows (16 min) it
already equals the 40k end silhouette — the tail is real sub-cell gains, so use a window
budget (`--animations 70-90`) for a 20-min 300k. Control basis at 300k REFUTED (det F 0.005).
The host (dabh's jobs, load 70-90) slows all host-side work 5-10×; hyde01 worse; hyde04 idle
but A4000 16 GB (no 300k) and no env.

**Why:** the user's complaint was the mid-morph surface; end frames were fine. Refuted before
this (all pre-registered, docs/surface_gradient.md §15): coarse control basis (cg17/cg9), u-step
projection (sm), geometric gate on the nearest target surface (ug/ugh — wrong residual).

**How to apply:** do not re-test those levers; the lump measurement is `scripts/probes/lump_trace.py`
(2 sp / cell / 2 cells + transport speed). If mid-morph texture is still an issue on some target,
the remaining excess is u on arrived material (+0.05 sp), the same as the end frames carry.
Related: [[always-report-render-influence]], [[no-parameter-only-fixes]].

**300k isolated pieces (2026-09-23 night, method.md §10.17a):** the plain-mesh stills showed beads
at the ear tips. They are stretched tip material inside half an MPM cell (18 particles, 0.12-0.15
wu off the body, ON the target; 8-NN 1.7-2.4 native spacings) — connected for the grid, a bead
for the native-spacing renderer; no mechanism owns it (decoupling counts 1 particle, fragment =
grid connectivity, kNN gate ratio 1 in a clump, DT pull 0 on the target). Root cause: the
constants. At 300k on the 40k grid every spacing-derived length was half the 40k length and every
neighbour count a seventh of the mass (the tip holds 5.9 reference particles vs 13 at 40k).
Rule (38): lengths x (N/mass_ref_n)^(1/3), counts x N/mass_ref_n. Render side ADOPTED
(`render_photoreal --ref_n`, default 40000; 40k bit-identical): no beads, octree at the 40k level,
bump 1.9 -> 1.5 (prediction <= 1.3 refuted; the tip knobs are physics). Physics side `--disc_ref`
implemented (config.disc_ref_factor; layer/asymmetry/isolation/bond counts, decoupling count,
nn_sp, sp0, sigma0, sp_t), twin `d300_bunny` pre-registered (P5 tip mass >= 10 ref particles,
P6 8-NN <= 1.3, P7 silIoU >= 0.964, windows <= 210) — read `$STATUS` "D300 DONE", then
`$OUT/scratch/iso_tip.py d300 g41`. Plain-mesh videos (stride 12) for the user: `$OUT/r300/plain/`.

**Twin d300 RESULT (2026-09-23 20:05):** tip mass 18.3 ref particles (5.9 / 40k 13), tip 8-NN 0.58
(1.67), silIoU 0.9699 in 52 windows / 16 min (0.9677 in 176 / 44), det F 0.745 — BUT native bump
2.3° (1.9), mid-morph roughness 2x, 95 particles in small pieces (37): P5/P7 hold, P6b/P8/P9
refuted. `--disc_ref` stays opt-in (NOT in RECIPE). The pair proves the root defect: no term orders
the quadrature below the cell (c300 chases sub-cell residuals with window-to-window reversal corr
−0.5…−0.7 for 120 windows; d300 ignores them). Next: a loss-neutral, mass-preserving sub-cell
ordering step (particle shifting / redistribution), after the papers (agent digest requested:
Eisenberger divergence-free / Hamiltonian / Smooth Shells / NeuroMorph, Lind 2012 shifting, Gao
2017 MPM). Probes: `$OUT/scratch/osc_probe.py NPZ 19`, `iso_tip.py`, `lump_trace.py`.

**Digest verdicts (2026-09-23 night, docs/related_work.md):** (a) only a band-limited control field
removes the sub-cell null space (Eisenberger line) → our `--control_grid 17`; b300's refutation was
confounded by unit masses → re-run as `e300_bunny` (pre-registered P10–P15, running). (b) the
derived, mass-preserving repair is Fickian shifting Δx = −½h²∇C (Lind 2012; λ = ½ the stability
limit), positions only, tangential on the layer → E2 to implement as an opt-in commit step.
(c) no paper optimises window by window; the reversal is the symptom of chasing sub-cell residuals.

**E1/E2 results (2026-09-23 21:00):** e300 (basis + mass contract): silIoU 0.9795 (best), det F 0.74,
206 windows, tip 8.7 — the b300 refutation overturned; the tail reversal PERSISTS (−0.84) → the
oscillation is the outer loop's, not the control DOF's. f300 (shifting): chamfer 0.0554 (best),
roughness 0.18–0.21 (best), silIoU 0.9748, BUT a 295-particle chunk detached at the right ear tip
(the stretched thread that held the thin tip is evened out) → not adoptable alone. Running: g300
(disc_ref + shift) and h300 (+ basis), pre-registered P22–P28. Next: E3 outer-loop damping.

**The visible 300k "oscillation" DIAGNOSED (2026-09-23 night, docs/oscillation.md Addendum 9):** the
OUTER LAYER breathes along its normal 0.1–0.2 native sp/window (0.007–0.027 wu), 60–80 % sign flips
per window, net drift 0.04 sp for 2.5 sp summed — a render-driven u step undone the next window; the
40k tail has the same at 0.003–0.009 wu but its video is HELD after convergence at 57 windows. Not
the basis, not shifting, not subsample noise, not the render (all measured). Candidates: (a) stop the
delivered trajectory on net/summed displacement (hold like 40k), (b) per-particle Rprop damping of
u or a noise-floor stop. v4/v6 reversal brakes are retired (froze descent) — do not retry them.

**COMBINATIONS (2026-09-23 22:40) — the first runs where all three defects are gone together:**
g300 (`--disc_ref --shift_sub`): silIoU 0.975 / chamfer 0.0556 / det F 0.72, 59 windows 17 min, tip
23.7 ref particles (8-NN 0.79), off-body 5 singletons, NO breathing (net/summed 0.7–0.9, layer corr
+0.1…+0.3), g_share 0.37; roughness median 0.27 (marginal). h300 (+ `--control_grid 17`): 0.977 /
0.0554 / det F 0.835, 79 windows 32 min, tip 41.7 (over-packed 1.9×), off-body 4, roughness 0.22.
Under the reference constants the loop converges instead of breathing (d300/g300/h300). Candidate
recipe for N > 40k: `--disc_ref --shift_sub` (+ basis); 40k untouched. Stills / videos pending;
adoption = the user's call. i300 (stop_on_cycle) and j300 (u_rprop) are native-constant experiments,
now secondary.

**Independent audit (docs/diagnosis_300k_20260923.md, untracked, written outside this session at
dc6d1b8) — its corrections ACCEPTED and folded into the docs:** (1) f300's 300-particle ear-tip
piece has nonzero cubic-stencil overlap with the body (coupling weak, not absent; "shares no node"
was wrong); (2) sub-cell arrangements are UNDERCONSTRAINED, not an exact loss null space; shifting
is not "loss-neutral by construction" (record the objective change across the shift); (3) the
breathing's attribution to render-driven u is unproved (u is fed by both channels + the relaxation;
the render-only-u twin is the test); (4) the derive() CFL prints unit mass while the run rescales —
fix the diagnostic; (5) i300 stop_on_cycle fired at animation 106 (silIoU 0.9625). Audit's next
diagnostics: separate the u / relaxation / stress increments on a late checkpoint; roughness and
connectivity at fixed world scales; spatial refinement at fixed N and mass; shifting's state transfer.
**i300 (stop_on_cycle):** fired at 106 (pred 60–100 ✗), silIoU 0.9625 ✓, 19 min; the pre-freeze tail
still breathes (|dI| 0.0026) — a safeguard, not a fix; secondary to g300/h300. Opt-in.
**j300 (u_rprop) REFUTED (23:45):** u bound at the floor for 74 % of the layer, yet sign flips 92–97 %
and layer corr −0.9 in the tail; silIoU 0.9641, 184 windows. The breathing is NOT u's — it is the
stress control + relaxation chasing sub-cell residuals at the native constants; only the reference
constants (d300/g300/h300) remove it. Do not retry u damping / render-only u for the breathing.

**State at 2026-09-23 23:50 (against the acceptance criteria of [[until-zero-oscillation]]):**
g300/h300 meet: breathing gone (net/summed 0.5–0.9, layer corr ≥ −0.3), end bump at the reference
render 1.3°/1.2° (= 40k), pieces ≤ 5 singletons, silIoU 0.975/0.977, det F 0.72/0.84, cell-scale
roughness in wu = 40k's (0.03–0.04 wu). Remaining: the sub-cell texture (2-native-spacing rms
0.014–0.022 wu — invisible to the 40k probe), the video's sub-cell drops in 17–21 % of frames (40k
9 %), h300's over-packed tip (1.9×). Running: k300 (cell_diag 36 = dx 0.22, grid 49³, + disc_ref +
shift; 30 s/window), the external baselines (ISD no-corr done: end Chamfer/diag 0.0070 but the
trajectory collapses to 1 % volume at t=0.5, genus 24 at t=0.1; 4Deform, torus pair, CORR=nearest
queued), two literature surveys (MPM particle–grid–surface 2013–2026; SIGGRAPH geometry). New
metric `metrics.layer_breathing` in every run's metrics (flips / net ratio / step).
**CORRECTION (23:55):** by `layer_breathing` (layer normal component, last 10 windows) g300 flips
0.56 / net 0.25 (a random walk, not c300's cycle 0.78 / 0.12), h300 0.66 / 0.13; the 40k reference
itself flips 0.74 at 0.003 wu. The visible quantity is the AMPLITUDE: 40k 0.0029 wu/window, g300
0.0047, h300 0.0078, c300 0.0050; video delivered-tail |dI|: 40k 0.0012, g300 0.0019, h300 0.0017,
c300 0.0025. "Breathing gone" was overstated: reduced to a random walk at 1.6–2.7× the 40k
amplitude. Acceptance restated: ≤ 0.003 wu/window and ≤ 0.0012 per frame, then hold.

**2026-09-24 00:45 — two surveys in (docs/related_work.md) and the first mechanism from them:**
`--commit_pic` (mpm/gridfilter.py, method.md 10.20) = XPIC(5) null-space projection of each window's
displacement at the commit (Gritton & Berzins 2017 / Hammerquist & Nairn 2017); tests pass; the 20k
smoke removed 28 % of the first window's displacement. Run l300 = g300 recipe + commit_pic queued on
GPU 2 after k300 (pre-registered P48–P52: layer step ≤ 0.003 wu, flips ≤ 0.5, roughness ≤ 0.15,
silIoU ≥ 0.972, tip ≥ 18; refutation silIoU < 0.965 or tip < 12). Other shortlist items not yet
tried: F-domain surfels (CPDI2 corners for the objective/render), grid-origin shift diagnostic,
envelope-constrained PSR (Kazhdan 2020), stochastic PSR variance test, point-cloud Laplacian
spectral criterion, Yu&Turk PCA kernel render (stills pending: g300_end_pca240/32).

**l300 (2026-09-24 00:20) = g300 + `--commit_pic` (XPIC(5) null-space projection at the commit): the
BEST run — silIoU 0.9797, chamfer 0.0551, det F 0.66, 85 windows / 22 min, 6 singletons off-body,
tail layer step 0.0017 wu/window (< 40k's 0.0029), 78 % of the tail's window motion removed as
grid-invisible; the low-band (>= 5-cell) alternation persists (flips 0.74) as in EVERY run incl.
40k. Candidate recipe for N > 40k: `--disc_ref --shift_sub --commit_pic`. Pending: its end bump,
video tail |dI|, tip mass; the rebound diagnostic (elastic spring-back hypothesis for the low band).

**The low-band alternation's MECHANISM (2026-09-24 00:35–01:00, rebound_probe):** the carried
momentum (v0 = the commit's velocity, C) travels on +0.3…+0.9 of a window's displacement with zero
control (the elastic part is a tenth) and the next window cancels it → the coherent two-window
alternation in every run. Windows from rest (`--rest_commit`, method.md 10.21) kill the alternation
(20k: flips 0.62→0.33, net 0.20→0.46) but unconditionally cost the transport (silIoU 0.919 vs 0.967
at 30 windows; m300 at 300k 3× slower → killed) → the ARRIVAL-GATED form (rest_commit_gate = u
transport gate 100 %, latched) is running as m300b (P53–P57). Do not retry the unconditional form.
**l300 full (01:05):** end bump 1.19° at the reference render (= the 40k bar), native 1.33°, tip 18.5,
video drops 3/124 frames, tail |dI| 0.0016 (40k 0.0012). Acceptance: surface ✓ pieces ✓ fit ✓
amplitude ✓; flips (universal momentum alternation) and the video tail (0.0016 > 0.0012) remain →
m300b (arrival-gated windows from rest) is the test for those.
