---
name: until-zero-oscillation
description: "User directive 2026-09-23 night: keep experimenting and reading until the oscillation is zero and the surface smooth; search SIGGRAPH geometry + MPM grid/resolution/mesh papers since 2013 and apply the ones that address the current problems"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-24T04:29:25.967Z
---

The user (2026-09-23, after the 300k defects and the audit): "진동이 0, 그리고 surface가 매끈 할 때
까지 실험 및 논문 찾기 계속 해. 특히 geometry 쪽 siggraph랑 mpm 해상도와 격자, 그리고 mesh간의
관계와 관련된 2013년부터의 모든 paper를 찾아서 가장 현재 문제를 다루고 있는 paper들을 찾아서
적용해 봐."

**Why:** the deliverable is judged by eye on the plain-mesh video; the three defects (rough
surface, isolated pieces, the tail's breathing) are the blocker, and the user wants the
literature (SIGGRAPH geometry; MPM particles ↔ grid resolution ↔ mesh) mined systematically rather
than one lever at a time.

**How to apply:** treat "oscillation 0 and a smooth surface" as the acceptance state (docs/
experiments.md 2026-09-23 night "acceptance criteria": layer breathing flips ≤ 0.5 and net/summed
≥ 0.5 over the last 10 windows, end bump at the reference render ≤ 1.2°, mid-morph roughness within
the 40k band, no piece ≥ 6 particles, the plain video's tail per-frame change ≤ the 40k value). Every
candidate mechanism comes from a paper (the two survey digests in docs/related_work.md), is
pre-registered, and is refuted or adopted on the 300k bunny before anything else. Keep the 40k
gallery untouched. Related: [[papers-before-ambiguous-design]], [[mid-morph-lumps-u-channel]],
[[competitor-landscape]].

**Extended (2026-09-24 01:10, the user):** "최적화 진동 + 여전히 귀쪽으로 갈 때 어색해 (너무 가늘고,
물방울이 서로 만나서 귀를 이루는 것 같아) 이거 전부 수정할 때까지 계속 paper 찾으면서 수정해 줘
(300m 후에)" — after m300b, keep going on BOTH the optimisation oscillation and the EAR FORMATION
(the ears grow too thin, as droplets that meet and merge). Diagnostic first: the ear region's
particle count, connected components (droplets) and thinnest extent per frame during the growth
(scratch `ear_probe.py`); candidates from the surveys: local refinement (k300 video), CPDI
F-domain surfels (render), Ando 2012 anisotropic split/merge, mesh-carried thin features.
**The user's hypothesis (01:20):** "귀쪽으로 초반에 가는 입자 수 때문" — too few particles reach the ears
early. Test = the ear region's mass fraction per frame against the body's arrival curve
(`ear_probe.py`, y-up frame, split height auto-detected); then the OT plan's assignment to the ear
cells per window if the curve lags. Fix candidates must come from papers (SIGGRAPH rule) and be
pre-registered, as before.

**Ear mechanism found (2026-09-24 01:45, ear_slab.py):** the 300k ear grows as a base bulge (the
lowest ear slab at 1.5–1.7x its target count) feeding a filament at 0.5–0.7 of the target thickness
that breaks into pieces at the native spacing; the 40k tongue grows at 0.8–1.0 thickness. Cause: the
paced target is the straight-ray displacement interpolant of the plan, not volume preserving in
transit for an anisotropic map (det(I + t(J − I)) ≠ 1) — over-dense at the feature's root,
under-dense in the stream; at 300k a sub-cell filament can realise the sparse stream. Fix built
(method.md §10.22, `--pace_project`): the paced step projected onto divergence-free fields on the
body, p = 0 on the free surface (Maury 2010, Perthame 2014, Eisenberger 2019). Pre-registered n300
(P58–P61). Lesson: the bulk node mass must be the histogram MODE — a median above half the maximum
followed a compressed pocket and shrank the body to 200 nodes (first n300 launch, restarted).
**Oscillation latch (same night):** m300b's u-gate-at-100 % latch fired at window 62 of 66 (useless);
the reversal cosine reads arrival cleanly (+0.9 in transport, then negative every window: m300b
from 44, l300 from 49, 40k from 20). Second form `--rest_commit_reversal` (latch at the second
consecutive negative), pre-registered o300 (P62–P65). Both runs launched ~01:50 server time.
**o300 refutes momentum (2026-09-24 02:50):** windows started from rest (latched at 43) still
reverse their predecessors (cos −0.63…−0.31, kinetic energy down 7x); the step size is at its
annealed floor in l300/40k and they reverse too; λ/g_share do not alternate. What alternates is the
shape merit itself (40k: the outer gain flips sign every window, ±1–2 %; silhouette term in step;
cell sum flat) — the commits alternate between a better and a worse state, and the outer gate
ACCEPTS slightly negative gains. Next: p300 (render off from window 45) decides whether the render
channel's per-window action carries it; the structural deliverable fix under consideration is a
monotone merit acceptance (never commit a worse state) which ends the run at the alternation's
onset with the deliverable at the best commit.
**n300 verdict (03:30):** the projected paced target does NOT change the growth (base 1.50×, thickness
0.63) — the cell sum is blind to incompressible interior flow, the compression pocket deepens
(40 %); but the end state improves (ear fill 0.987, det F 0.72, 2 strays, tail step halved) → kept
opt-in. The ear mechanism restated: front pulled + material stretched, supply piles at the base
because the LOG cell sum's gradient collapses at high occupancy. Next: t300 = linear cell sum
(`--dvol_form linear`, P74–P77); if refuted → incompressible MPM step (Stomakhin 2014) is the
remaining structural route. Oscillation: p300 (render off from 45) and r300 (gate armed at the
alternation onset, `--outer_latch_reversal`) running.
**Bulge's constant (04:05):** c300 (native) has NO base bulge (but an unfilled ear); the bulge came
with --disc_ref because the OT plan blur is sample-derived (0.119 wu at 40k, 0.116 at c300,
N-independent) and disc_ref scaled its input spacing too → 0.227 wu (double count, rule 38).
Fix `--plan_native` (u300, P78–P81); fold into disc_ref if it holds. Never put a literal pkill
pattern in the same ssh command as text that contains it (killed my own shell twice).
**Verdicts so far (05:40 local doc time; server 02:30 CDT):** t300 (linear cell sum) refuted for the
ear (base 1.34×, thin 0.6; fit 0.9707, det F 0.53, 24 strays) — only the compression pockets vanish.
p300 (render off from 45) refuted: reversal persists (−0.29), fit −0.003. r300 (gate armed at the
onset) stopped at 43 on near-zero cosines (silIoU 0.9776, tail 0.0030 — the per-frame tail measure
conflates transport with flicker); onset now = cos < −0.2 twice (r300b running). Two phenomena:
the merit reversal (starts at arrival) vs the layer breathing (flips 0.7 in every phase) — w300
(no --layer_relax) tests the relaxation as the breathing's carrier; v300 (--ot_handoff) tests the
moving paced target as the reversal's carrier. u300 (plan blur native) done: silIoU 0.9788, 53
windows; ear slabs pending. Torus head-to-head: ours 0.039 end, genus 1, weak case recorded.
**07:05:** r300b (gate armed at cos < −0.2 twice) stops at 48, deliverable 45, fit 0.9782 — the stop,
not the cure (breathing flips 0.69 unchanged). v300 (--ot_handoff) weakens the merit alternation
(5 negative windows then ≈ 0) at no fit cost (0.9795): the moving paced target carries part of it.
The deliverable recipe candidate: --ot_handoff --outer_latch_reversal. Breathing carrier test =
w300 (no --layer_relax) pending; ear = y300 (KDE) pending.
**Breathing conclusion (07:30):** w300 (no relaxation) flips 0.94 → the relaxation DAMPS it. Every
carrier tested is out (momentum, render, relaxation, step, balancer, u bound, XPIC/shift, moving
target partly). What all runs share: the optimiser's floor step (anneal floor = literal 0.05 in
runner.py, α 0.001) → the layer's normal step sits at 0.002–0.003 wu/window at both N — the
amplitude the acceptance criterion already admits. Structural cure left open: a line search on the
merit at the surface / a discretisation-derived floor; deliverable cure: --outer_latch_reversal
(+ --ot_handoff). Never lower the 0.05 by hand.
**Ear mechanism found (07:55): y300 = l300 + --plan_native --w_kde 1** (the particle-scale KDE density
term at equal gradient norm): base pile gone (1.11×), the ear grows as a tongue from the base at
0.9–1.05 target thickness where filled, tip 16.1 ref particles, 4 strays, fit 0.9782 — price: the
ear fills later (0.31 at t=0.2), det F 0.588, the leading edge still fragments. Combined
deliverable candidate z300 = + --ot_handoff --outer_latch_reversal launched 08:00 (P93–P96).
**08:30:** z300 (KDE + hand-off + onset gate) refuted: the gate stops at the bulk's onset (58) while
the KDE-slowed ear is at 0.84 → silIoU 0.9705. Pieces don't compose with the gate. z300b = KDE +
plan_native + ot_handoff (no gate) launched (P97–P100). Deliverable candidates so far: y300 (ear)
and v300 (calm tail) — z300b is their union.
**09:05 FINAL of the night:** z300b = l300 + --plan_native --w_kde 1 --ot_handoff = the best
deliverable (ear 0.936, silIoU 0.9802, 7 strays, tongue at target thickness, bump 1.24°); the
post-arrival alternation persists (35 negative windows) — the onset gate can't be combined with
the KDE-slowed ear until an ear-aware onset exists (open). Recipe for N>40k stays opt-in; 40k
gallery untouched. Open items: ear-aware onset; thin-feature carrier for the tongue's leading
edge; a discretisation-derived anneal floor (the breathing's amplitude).
**10:30 — the structural fix built:** per-particle Rprop on the control step (`--ctrl_rprop`,
§10.24; u floor → `--u_rprop_floor 0`): reversing particles halve their step without floor,
transporting ones keep it (no conflict with the slow ear). ab300 = z300b recipe + this, P103–P106.
BLOCKED: hyde01 rejects the ssh key since ~09:00 (server time 03:xx CDT); retry loop running;
code unverified on the server (syntax + smoke first). Also ready: M1–M3 (kind of the tail motion),
ear_census.py, R-1 `--keep_attached` in render_photoreal.
**2026-09-24 15:00 (the morning's measurements):** M2 — the reconstructed surface moves along its
normal 1:1 with the layer (v_n/d·n 0.91–0.98): the breathing is GENUINE (a), not refit re-sampling;
M1 — normal-only twin = full change, tangential-only 70 % (a refit floor coexists); M3 (leaf =
cell) changes nothing → band-limiting is not the fix. Rprop: g41r (40k) alternation gone (corr
+0.2, flips 0.47, fit +0.006) but amplitude stays as drift; ac300 (300k) flips 0.41, step 0.00095
wu ✓ but det F 0.39 (per-particle scales = sub-cell control noise) → ad300 = neighbourhood-smoothed
Rprop (`--ctrl_rprop_smooth`, P116–P119) running. Ear tip: census shows particles monotone, mesh
one component, fit captures 3–27 tip surfels (varying) — unrepresentable at the node size
(PoissonRecon probe flat); `--thin_fallback spheres` draws the bead chain faithfully; the fix is
physics (P-1 sheet-aware splitting at the commit), next.
**15:50:** ac300 (l300 + per-particle Rprop) reaches the ACCEPTANCE tail value at 300k (video tail
0.0013 = 40k; stride-19 ALT 0.0007 vs 0.0014) with flips 0.41, corr +0.04 — but det F 0.39 and
the ear tip thinner (10.7 vs 18.5). Smoothed forms: k≈60 (ad300) keeps det F 0.64 but dilutes
(corr −0.31); k=8 (ag300, creg's neighbourhood) running; af300 = z300b combo + smoothed(60)
running; the 19-target 40k sweep (g41s_*) running for the generalisation rule.
**16:35 gallery (g41s, 19 targets, 40k + smoothed Rprop):** fit up on 17/19, flips down on 15/19,
but C −0.011 silIoU (curved transport round the hole: legitimate direction changes halved) and
homer det F −0.06 → P124/P125 refuted → third form `--ctrl_rprop_arrived` (halve only ARRIVED
particles, the paced target's own mask) — g41t sweep running (P135–P137). Lesson: any per-particle
step rule must not read transport direction changes as overshoot.
**19:15 — the residual motion is the rollout's, not the optimiser's:** g41f (freeze at arrival:
control 0, v 0, stretch assimilated for 80 % of particles) still moves the body 0.0016 wu/window
(g41 0.002–0.004) and costs 0.005 silIoU → NOT adopted. g41a (assim 0) worse → assimilation not the
carrier. g41h (held step) removes the alternation (corr −0.18, 1 negative window) with no cost.
So: optimiser side = per-particle Rprop, smoothed k=8, arrival-gated, held step (adopt if ai300 +
the g41u gallery hold: C/beast det F collapsed at the arrived/in-transit boundary in g41t);
deliverable side = the tracked surface with a half-spacing band (+ moving average at the
window's time resolution) so a sub-resolution jitter is not re-fitted every frame (P151–P153).
**20:30 — the every-method campaign at 40k is closed:** on top of H (held step + arrival gate +
smoothing k=8) the five remaining levers — windows from rest (g41k), onset gate (g41o), fixed
target (g41m), settled-body viscosity (g41v), settle-at-commit (g41w) — all give the same tail
(flips 0.47, step 0.02 sp, ALT 0.0014) with fit ≥ g41. The residual jitter is the floor of
delivered simulated frames; only the deliverable surface (tracked mesh, half-spacing band,
window-time average: P151–P153) remains for the visible flicker. At 300k ai300 (l300 + H):
video tail 0.0011 < 40k's 0.0013, det F 0.70, low-band +0.11; al300 (KDE ear + H) = candidate.
Viscosity/settle forms are DIAGNOSTIC only (they change the material during the morph; the user
requires physical-parameter response to external forces).
**22:30 — GENERALISATION ALARM (D2 recipe):** the dragon at 300k with z300b's recipe (KDE ear +
hand-off + plan_native) gives silIoU 0.977 but det F 0.14 and 0.2 % strays BEFORE any Rprop —
the KDE term (suspected) compresses spiky targets. Attribution twins dl300 (l300 recipe) / dk300
(no KDE) running. Do not adopt --w_kde for N>40k until the dragon passes det F ≥ 0.55.
Oscillation (D1): H generalises with the arrival gate on the halving AND the onset read on
arrived particles (g41y sweep running); the global onset misfired on nefertiti/C (g41x).

**Extended (2026-09-24 night → 09-25, the user):** "아예 진동이 0이어야 해" + "plasticity처럼 최적화되면 고정시켜버리면 안 되나?" → the PIN (`--settle_pin`, method.md §10.27): an arrived, twice-reversed particle is a kinematic constraint inside the rollout (k_g2p/k_update/k_layer_project skip it; mass stays on the grid). g41p (40k bunny): 0 reversal windows, the settled body exactly still (0 of 28k un-pinned, read from the frames by `scratch/pin_probe.py`), stride-19 ALT 0.0005 (H 0.0014), silIoU 0.9679 (+0.007), det F 0.74 — P181–P184 ✓. M4 (advect-only render of ai300: 0.0015 vs re-fit 0.0011) settled "particles vs re-mesh": the motion is the particles'. Viscosity/settle forms stay diagnostic (the user: the morphed object must respond to external forces via λ, μ). Adoption gates pending: g41z (19 targets), g41q (pin without the global hold; C's stall), ap300 (300k). The per-particle Rprop's global hold stalls slow transports (C, nefertiti); the arrival-read onset fixed nefertiti but not C.
