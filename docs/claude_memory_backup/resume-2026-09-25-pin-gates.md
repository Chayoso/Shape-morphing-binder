---
name: resume-2026-09-25-pin-gates
description: "Resume point 2026-09-25 ~02:40 KST — the pin (--settle_pin) adoption gates running on hyde06 (g41z 19 targets, g41q, ap300 300k, am300 chain, dl300/dk300 dragon); where the verdict files are and what to read/decide next"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-26T03:06:44.146Z
---

State when the user closed the local session (2026-09-25 ~02:40 KST; the user: "일단 돌리고 있어 줘"):
runs are detached (nohup) on hyde06 and keep going. Repo HEAD 23438db (branch v3-grid-gs); the page
https://claude.ai/code/artifact/4a8f8c34-9abd-4f4f-8209-909bd5454920 carries g41p + M4.

**Pending readings (all under `$OUT=/data/relcfd/chayo/physmorph_v2/output`, scratch `$OUT/scratch`):**
- g41z (the pin = g41y form + `--settle_pin`, 19 targets, GPUs 2 & 1): `python scratch/gallery_table_z.py`.
  Gate P185: fit within −0.003 of g41y, pinned ≥ 0.5 on ≥ 15, det F within −0.05. At 4/19: C +0.061,
  bunny +0.006, fit holds; det F cow 0.775 (−0.078), nefertiti 0.681 (−0.083) → boundary compression
  (last arrivals against the pinned body). If systematic: variant = pin only particles whose coherence-kNN
  neighbourhood is arrived too (no constant); pre-register before running.
- g41q (pin WITHOUT the global hold = the 40k ADOPTION CANDIDATE; GPU 0): `gallery_table_q.py`. P186 ✓ — bunny 0.9689,
  C 0.9621 (≥ the hold form); launched 02:45 on the remaining 17 targets. Gate: P185 bounds applied to g41q.
- ap300 (300k bunny, ai300 form + pin, GPU 1): verdict `scratch/verdict_ap300.txt` (chain: verdict.sh + pin_probe +
  video_flicker; "AP300 CHAIN DONE"). Gates P181–P184 (pinned ≥ 0.6, tail ≤ 0.0008, silIoU ≥ 0.9660, det F ≥ 0.6).
- am300 (KDE ear + onset hold, 300k): run done (silIoU 0.9799, det F 0.41 ✗, hold at 90, alternation continued);
  chain still rendering → `verdict_am300.txt` ("AM300 CHAIN DONE") for the video tail / ear fill.
- dl300 / dk300 (dragon 300k attribution of det F 0.14: l300 recipe vs z300b without KDE): logs
  `$OUT/dl300_dragon.log`, `$OUT/dk300_dragon.log`; P179/P180 in experiments.md 22:30 entry.

**Then:** record verdicts in docs/experiments.md (append-only), dossier_oscillation_thin.md (§13+), update the
page (output/r300_page/index.html, republish with `url`), decide the 40k recipe (pin; hold likely dropped) and
the 300k recipe (ap300 form; KDE ear only if the dragon passes), commit. Tools: `scratch/pin_probe.py NPZ`
(exact stillness check; needs `pinned`/`pinned_at` in the npz — g41z/ap300 have `pinned`, later runs `pinned_at`).
See [[until-zero-oscillation]], [[eval-trajectory-is-the-commit-rollout]], [[generalise-beyond-bunny]].

**Update 2026-09-25 09:55 (server 19:50 CDT):** g41z/g41q/ap300/am300/dl300/dk300 all read and recorded (experiments.md
09:50 entry, dossier §13, page). Verdict: the pin removes the oscillation on all 19 but the pinned body is an obstacle
(det F −0.05…−0.11 on 6 targets, minima in the unpinned set; ap300 ear tip starved 3.9 vs 13.6). New rule
`--settle_pin_clear` (eq. 50b: no unarrived particle within the pace radius). NOW RUNNING: g41n (19 targets, no-hold
pin + clear; GPUs 1/3; `gallery_table_n.py`), an300 (300k; `verdict_an300.txt`), dp300 (dragon without disc_ref, P187;
`verdict_dp300.txt`), g41q tail (GPU 0). Gates P188–P191. If g41n passes: 40k recipe = arrival-gated smoothed Rprop +
pin + clear (no hold); if an300's ear recovers: the 300k recipe likewise; else the pin's onset must be read per feature.

**Update 11:05 (server 20:10 CDT):** corrections — (1) the onset hold STAYS (nefertiti early-stops at 38–42 windows
without it under the pin; g41n stopped at 6/19, archives kept); (2) the commit-time isochoric assimilation undid the
pin-time assimilation's volumetric part → pinned particles are now excluded (`_Fp_new[settled_p] = Fp[settled_p]`).
NOW RUNNING: **g41pz** = z form + `--settle_pin --settle_pin_clear --settle_pin_assim` on all 19 (GPUs 1/3;
`gallery_table_pz.py`; gates P188'/P189'/P190/P192), an300 (300k ear reading, `verdict_an300.txt`), dp300 (dragon
without disc_ref, `verdict_dp300.txt`). Probes: `scratch/detf_time.py NPZ` (det F over time by pin state + END
elastic det F_e from the archived Fp), `scratch/detf_dist.py`. If g41pz passes → 40k recipe = z form + pin/clear/assim;
then the 300k run with the same and the D2 phase ([[after-oscillation-droplets-smooth]]).

**Update 13:05 (server 20:25 CDT):** the pin's remaining defect = a pinned body is a no-slip wall over the grid
kernel's support (2 dx): streams that must pass along/through settled material stall (nefertiti stops at 38–42 of
90 under every variant with clearance < 2 dx; the 300k ear starves). Rules now: `--settle_pin_clear` (point) and
`--settle_pin_ray` (plan rays), both with clearance max(r_pace, 2·dx); `--settle_pin_assim` (stress-free pinned
body, F_p final). RUNNING: g41ps (nefertiti beast ogre bunny C cow; GPU 0; `gallery_table_ps.py`; P200–P203),
g41pz remaining (point rule at r_pace; GPUs 1/3), as300 (300k, ray + assim + stencil clearance; GPU 2;
`verdict_as300.txt`; P204–P206), dp300 (dragon w/o disc_ref; `verdict_dp300.txt`; P187). an300 read: det F 0.657 ✓,
ear tip 6.8 ✗, tail 0.0011. If g41ps nefertiti ≥ 0.965 → run the g41ps form on all 19 (the 40k adoption gate).

**Update 17:30 (server 21:25 CDT):** the pin's drag is the sticky/Dirichlet wall of shared-node MPM (digests in
related_work.md). Variants: clear (50b), ray (50d), yield (50e) = geometric substitutes; the principled fix is
`--settle_pin_slip` (addendum 5, eq. 50f: pinned mass out of the momentum average, k_pin_mass once per window,
k_grid_op removes the approaching normal component). RUNNING: g41py (yield form; 14/19; `gallery_table_py.py`),
g41pw (slip form; nefertiti beast bunny cow; GPU 1; `gallery_table_pw.py`; P213–P215), ay300 (300k + yield;
`verdict_ay300.txt`; P211). as300 read: tail 0.0006 (best 300k), body still at 90 % pinned, ear 7.7 (✗). Health
gate for pin runs = end-state det F p1 ≥ 0.85 and ≤ 0.1 % below 0.5 (`detf_dist.py`); the run minimum is the
transit's dip. Next after g41pw: the slip form on the 19 and at 300k; then the D2 phase (dossier §14, item 0 =
capacity-aware arrival P212; nested fine box by penalty and the quadratic/compact kernel are the resolution levers).

**Update 18:00:** g41pw nefertiti (slip collider) = 0.9732, 57 windows, 95 % pinned, det F run-min 0.752 (= g41's),
no squeeze → the slip form (`--settle_pin --settle_pin_assim --settle_pin_slip` + onset hold + arrival-gated smoothed
Rprop) is the 40k adoption candidate; sweep on all 19 running (GPUs 0/1; `gallery_table_pw.py`; gate P218). 300k:
aw300 (slip) on GPU 3 (`verdict_aw300.txt`; P216–P217), ay300 (yield) on GPU 2 as the comparison.

**Update 20:20:** g41pw (slip form) 11/19: fit up/within −0.003 on 10 (homer −0.004), end p1 0.87–0.93 — the slip
collider is the 40k recipe's pin form (dossier §15). 300k: aw300 (slip) 0.9754, p1 0.856, ear 8.5 at 62 windows
(merit plateau at 99.8 % arrival — the ear's sub-cell deficit); running ak300 (slip + KDE ear, GPU 1; P220), ar300 (ray
clearance + slip, GPU 3; P222), dw300 (dragon slip, GPU 2; P219). Naming fix: the 300k slip run's outputs were renamed
as300→aw300; the original as300 archive is lost (numbers in experiments.md 17:20; video kept). Next: the 40k adoption
decision on the full g41pw table, the 300k decision on ak300/ar300, then D2 (§14: capacity-aware arrival P212, R-2
reconstruction, Ando split; the nested fine box and quadratic kernel from the digests).

**Update 21:00 — DECISION:** 40k pin form ADOPTED = RECIPE + `--ctrl_rprop --ctrl_rprop_smooth --ctrl_rprop_arrived
--ctrl_rprop_hold_onset --u_rprop --u_rprop_floor 0 --settle_pin --settle_pin_assim --settle_pin_slip` (g41pw 19/19:
17 within −0.003, armadilo −0.0032, homer −0.0043; flips 0.17–0.43; p1 ≥ 0.87; pinned median 92 %). D1 closed at 40k.
300k pending: ar300 (0.9773 run; ear/tail verdict), ak300 (+KDE), dw300 (dragon). Then D2 (dossier §14).

**Update 21:15:** ar300 (ray clearance + slip, 300k) = silIoU 0.9773 (best ever), no reversal, p1 0.864, pinned 60 %,
ear tip 9.2 (middle slabs at ai300's fill; the last two short), tail 0.0012 — the pin runs plateau at 61–62 windows:
the tip's sub-cell deficit is not seen by the cell-sum merit. Pending: ak300 (slip + KDE; `verdict_ak300.txt`) decides
the 300k form; dw300 (dragon slip; `verdict_dw300.txt`). If ak300 fails: the early-stop rule under the pin is the next
lever (gains of −0.003 at 99.8 % arrival end the run), then D2 §14.0 (capacity-aware arrival).

**Update 23:05:** 300k under the pin: every run stops at 61–104 windows with the ear tip at 0.5–0.6 (aw300 8.5, ar300
9.2 [best fit 0.9773], ak300 8.7 with KDE) — the pinned body no longer SUPPLIES the ear (ai300 fed it from the whole body
over windows 62–112). New rule `--settle_pin_follow` (addendum 6, eq. 50g: "arrived" read every window; a pinned particle
whose plan image moved beyond r_pace is released, re-pinned when arrived): az300 = ar300 form + follow (GPU 1;
`verdict_az300.txt`; P224). Capacity rule (`--arrive_cap`, §10.28): fixes the dragon's spike over-fill at 40k (none
below 0.5) but costs nefertiti −0.005 → opt-in for spiky targets; dc300 (300k dragon + cap) vs dw300 (0.9646) pending.

**Update 2026-09-26 01:35 (Opus 5.5 session):** branch v3-grid-gs PUSHED to origin (was never pushed; auto-push after
commits is fine now that the user asked). Server CLEANUP done: 472 → 98 GB, logs archived to
`$OUT/logs_archive_20260925.tgz` + `.deleted_list.txt`; kept g41, g41pw, g41pc, g41pf, ai300, l300, z300b, ar300, aw300,
ak300, az300, dl300, dw300, dc300, reports. az300 (follow) inert: plan 100 % arrived. New lever `--settle_pin_kkt`
(addendum 7, eq. 50h: release pinned particles whose fixed-target cell-sum gradient > free median): aq300 (300k; P226)
+ g41pk (40k bunny nefertiti dragon; P227). A research agent is gathering "grow thin target features from a bulk source"
papers + cookbooks (the user: 귀 안 되면 계속 paper/cookbook 찾아줘) → append to related_work.md when it returns.

**Update 2026-09-26 03:35 — REFRAME:** the 300k ear under the pin is COMPLETE at the end (ar300 end still = ai300's;
coverage at the spacing equal; the "tip ≥ 11 reference particles" bound measured density and is withdrawn). The ear's
remaining defect is the GROWTH: a knob on a neck (ar300 left ear t=0.25: top slabs 1.20×/1.46× target thickness over a
0.66–0.76× neck) = D2. Knob index K (top-two-slab max thk / min thk below) pre-registered: ai300 1.6, ar300 2.2, aim ≤ 1.2.
KKT release abandoned (a pinned particle's gradient is the pin's multiplier). Probes: scratch/cover_probe.py,
grad_probe.py, gx_probe.py; montage tool = ffmpeg crops of the stride-12 videos. Page now at
https://claude.ai/artifact/AD1cFFC22zz2EJgSE1Ew8B (same artifact, v45). Research agent (thin features from a bulk source)
still running → related_work.md, then pick the D2 growth mechanism (candidates: WFR/unbalanced growth from the base,
tongue growth ordering, capacity-exact matching).

**Update 2026-09-26 03:55:** the growth knob is answered by the KDE term UNDER THE PIN: ak300 (slip form + `--w_kde 1
--plan_native --ot_handoff`) grows the ear as a tongue (knob index 1.1–1.3 vs 2.3 without KDE), end ears complete with
tapered tips, end p1 0.859; cost: body bump 1.3° (smoothness = next item). At 40k the knob is PERMANENT (g41p top slab 2.8×
= bulb tip). Running: ae300 (ar300 form + KDE; P228), g41pe (40k adopted form + KDE on bunny armadilo dragon nefertiti cow
C; P229). Supply digest appended to related_work.md (Ḣ⁻¹ held: same splat, reach not resolution).
**CORRECTION 04:10:** there is NO permanent 40k bulb — the 40k top-slab thickness (2.8×) is an ear_slab mask artifact;
g41p's end still shows tapered tips. Read the knob on the 300k tables and by eye (montage/stills). g41pe bunny −0.011
(KDE at 40k costs the fit, does not change the early tip detachment at t=0.15–0.25).

**Update 05:20:** H⁻¹ (`--w_h1 1`, the existing term from 2026-09-04) under the pin is the strongest lever of the night:
g41ph bunny 0.9724 / armadilo 0.9681 / dragon 0.9695 (+0.008..+0.011 vs g41), end p1 0.89–0.92, none below 0.5, pinned
0.97–0.99, no reversal; au300 (300k) 0.9770, det F 0.76. Growth by eye: earlier, no knob, but curled flap + forked tips.
g41ph running on all 19 (`gallery_table_ph.py`) → if it passes, adopted 40k form = g41pw form + `--w_h1 1`. KDE: refuted at
40k (3/3); ae300 (300k KDE) 0.9737. Chains of ae300/au300 rendering (verdict_*.txt).

**Update 06:40 — 300k VISIBLE OSCILLATION = mostly RE-MESH (the user asked to keep watching it):** tail heatmaps (ffmpeg
tblend difference + tmix over the last 24 frames) light the ears, the whole silhouette and body speckle although 60–90 %
of the particles are pinned; ar300 rendered with `--surface mc` (local marching cubes, fixed grid) → tail 0.0003 vs Poisson
0.0012 (same particles), body smoother, ears slightly fatter + faint voxel banding. Two owners: the delivered surface
(Poisson's global solve) and the last free particles. KDE (ae300) = tongue growth; H⁻¹ (au300) = 0.977, det F 0.76, tip
filled, knobby growth → av300 = both (GPU 2). g41ph (40k H⁻¹) at 7/19. Cleanup rule over 100 GB saved; done 117 → 95 GB.

**Update 2026-09-26 08:10:** g41ph (H⁻¹ under the pin) 19/19: 17 up/within −0.003 (bunny 0.9724 best), end p1 0.89–0.94,
none < 0.5; FAILS bimba (brake stop at 15: H⁻¹ inside the core inflates the balancer's physics norm → render out-pulls the
transport) and beast (fixed-target pull vs the pace on a long transport). Running: g41pho (`--h1_outside`), g41phn
(`--h1_onset_pin`), g41phb (both) on bimba beast bunny (P233) → the passing placement goes to the 19 → 40k form = g41pw +
w_h1. 300k trio by eye: ae300 KDE (hook tip), au300 H⁻¹ (knob; 0.977, det F 0.76), av300 both (detached bead at t=0.2):
the spike-stage tip shape is the open D2 defect. Cleanup 2 done (85 GB). Page fragment `h1_section.html` staged for the
page (insert before "<h2>D2 ·"), videos ar300_plain/mc + growth_300k_trio.png copied to output/r300_page.

**Update 2026-09-26 10:40:** H⁻¹ placement decided = BOTH (`--h1_outside --h1_onset_pin`): bimba 0.9778, beast 0.9544
(outside-alone ejected on beast; onset-alone brake on bimba); g41phb running on all 19 (`gallery_table_g41phb.py`) — if it
passes, the 40k form = g41pw + `--w_h1 1 --h1_outside --h1_onset_pin`. The user's c2f trick as a loss schedule:
`--c2f_onset_pin` (+ `--render_res 32 --render_res_hi 96|64`): fit-neutral on bunny; growth-by-eye pending (videos
rendering: g41cA/cB/cC_plain_s12). The early dent (user's frame) is a transport drain in the particles (dent_probe band:
−15 %, 18–21 bins at window 5); H⁻¹-from-start halves it, the coarse render start does not. mc ears' fatness = the
iso-level, not the kernel radius (ref_n 40k/100k/300k alike). Server 92 GB.

**Update 11:35:** the c2f render schedule (user's trick as a loss) = fit-neutral, NO change at the spike stage by eye
(P234 ✗). The growth order is the TRANSPORT's → new lever `--pace_front` (method.md 10.29: the paced image clamped along
its ray at the revealed target region = cells filled to ½ m_t or within r_pace of one; the ear fills from the base).
g41fr (adopted form + pace_front) on bunny dragon nefertiti (P236 by eye tongue, P237 fit within −0.003). g41phb (H⁻¹
both) at 8/19: nefertiti −0.0036 vs g41pw marginal, others fine. Next after g41fr: the front at 300k (ar300 form) if it
gives the tongue; then adopt H⁻¹ placement if the 19 pass.

**Update 12:45:** grid-scale front (`--pace_front`) refuted by eye (the spike is thinner than the loss cell; head rounder only).
Particle-scale front `--pace_front_pts` (filled/revealed on the target's own points) → g41fp on bunny dragon nefertiti
(P238) with an auto video chain (`frontpts_video.txt`). g41phb 16/19: neutral at 40k except cow −0.007, nefertiti −0.004
(no catastrophes) → not adopted at 40k; the 300k supply test = ao300 (ar300 + `--w_h1 1 --h1_outside --h1_onset_pin`,
P239). Cleanup 3 done → 72 GB.

**Update 14:00:** g41phb (H⁻¹ outside+onset) 19/19 = neutral at 40k (cow −0.007, nefertiti −0.004; no catastrophes) → NOT
adopted at 40k; 40k form stays g41pw. Beast: D1 residual (10/87 reversal windows, transport stalls at 76 %). Running:
g41fp (particle-scale front: bunny 0.9678; dragon nefertiti beast cow pending; bunny video for the by-eye read), ao300
(300k H⁻¹ placed; chain rendering). Server 72 GB.

**Update 16:50 (server 07:45 CDT) — the FRONT (D2 growth) and its costs:** particle-scale front `--pace_front_pts`
(g41fp): ear grows as a short thick STUB (montage `frontpts_growth_40k.png`, page v48), beast's transport finishes
(+0.006, 0 reversal windows) — but bulk targets are serialised (nefertiti 115 windows vs 57, −0.008; cow −0.006) and the
dragon's spike fronts over-fill (44 < det F 0.5). Cap `--pace_cap` (g41fq): bunny 0.9704 (= g41pw, p1 0.93, 0 reversals),
dragon health fixed (0 < 0.5) but fit 0.9508 (−0.013): the cell-level cap and the point-level fill DEADLOCK on sub-cell
spikes (front cell full, unfilled points unreached → 35 % pinned at 86 windows, stop). Geodesic front `--pace_front_geo`
(eq. 52b, method.md 10.29 addendum; g41fg nefertiti only, loop stopped) = diagnostic; likely vacuous for the ear (wave
speed = pace speed). NEXT LEVER being implemented: `--pace_front_fill` (eq. 52c) — held images ASSIGNED to revealed
vacancies within one pace, one particle per target point (capacity as arrive_cap) → no pile, no deadlock, air-side
material seeds the surface. Running: g41fq nefertiti+cow (GPU 3), g41fq beast (GPU 2), af300 (300k bunny, ar300 form +
front + cap; P242 = the D2 test), dh300 (300k dragon + H⁻¹ from the start; P241), ao300 chain video. ao300 read: 0.977,
det F 0.72, tip 7.2 ✗ (P239 ✗). 40k form still g41pw. Server 86 GB (cleanup at 100: candidates g41fp/g41fr/g41phb npz,
ae300/l300/ai300 npz — archive logs first). Tables: `gallery_table_g41fq.py`, `_g41fg.py`, (`_g41fv.py` for the fill).

**Update 19:00 (server 09:55 CDT):** the front's standing hold (0.26–0.32) was the point cloud's COVERAGE GAP: 24–27 % of a
target-filling cloud lies > 1 spacing from every target point, 0–1 % beyond the 8-NN shell (1.98 sp) → the front's radius is
now the per-point shell radius (method.md 10.29 correction 17:40). 52e added: held particles with no vacancy in reach
approach the front's nearest open vacancy (rounds, one per point). Refuted/withdrawn: g41fg (geodesic), g41fv, af300, g41fq
nefertiti/beast (starve under the cap; cow ✓ 0.9648). g41fw dragon (shell radius alone) 0.9588 healthy. **dh300 P241 ✓**
(0.9685, p1 0.839) → 300k form = ar300 form + `--w_h1 1`. RUNNING: g41fx (52c+52e; nefertiti dragon bunny beast; GPU 3;
`gallery_table_g41fx.py`; P245), ag300 (300k bunny, front+cap+fill; `verdict_ag300.txt`; P242 = the D2 test). Cleanup
script ready: `scratch/cleanup_20260926d.sh` (46 npz, 40 GB) — run when ≥ 100 GB (95 now). Page v49 pending publish.

**Update 21:10 (server 12:05 CDT) — the ear's growth MEASURED (dossier 17.2, `scratch/ear_order_probe.py`):** the ear's
material is one column of the head translated 1.34 wu (tip/neck/base-bound groups ordered from window 0, equal travel);
the physics STRETCHES the column (rear slow in the head, front free): 68 % of tip-bound above the neck at t=0.4 for 27 %
of neck-bound arrived → the spike = the stretched column, the knob = the tip group stopping first. Fronts with "filled =
one particle" are chains (held ≈ 0 once the coverage-gap and air-side-hold bugs were fixed); the g41fp "stub" was the
pile's pull; the cap costs −0.006 (surface-halo nodes) → withdrawn; the thin restriction is vacuous at 300k (0.1 %).
NOW: `--pace_front_dense` (52f: filled at half the shell count, reveal one shell) + `--pace_front_fill` (52c/52e), no cap.
RUNNING: g41fd (bunny+dragon GPU 0, nefertiti GPU 2; `gallery_table_g41fd.py`; P247), ba300 (300k bunny, ar300 form +
w_h1 + dense front; GPU 3; `verdict_ba300.txt`; P248). ag300 (morning's front + cap, 300k) = 0.9803 best 300k fit, knob
intact, 5-window reversal streak at the end (video chain rendering: `r300/plain/ag300_plain_s12.mp4`). dh300 P241 ✓ →
300k form = ar300 + `--w_h1 1`. Cleanup 4 done (101 → 62 GB; `.deleted_list_20260926d.txt`). Page v49 published; the
growth-strip tools: `scratch/growth_strip.sh TAG`, `growth_montage.sh OUT TAG...` (crop 420x380 at 120,140 of the left view).

**Update 22:30 (server 13:25 CDT):** the density front is FIT-NEUTRAL (7+ targets ±0.002; dragon 68 windows for 103) but
does NOT change the ear's growth (by eye + `ear_order_probe.py`: 75 % vs 72 %); at 300k ba300 = tip 16.3 (best), no
reversal, knob 2.9 (worse). `knob_probe.py`: the early knob is the ear's OWN column — tip-bound first + neck-bound
OVERSHOOTING to the top (31–46 %) → suspects: (a) the render channel (g_share 0.4–0.5 in the growth windows) → twins
bb300 (300k, `--lambda_auto 0`, GPU 1) and g41pl (40k bunny, GPU 3), P250/P251; (b) the plan's kNN blur (k=64
neighbours, 0.227 wu under disc_ref; `--plan_native` 0.116 — only ever run inside the KDE bundle ak300) → bc300 (ar300 +
plan_native, GPU 2), P252. Read each with ear_order_probe + knob_probe + the verdict slab table; strips via
growth_montage.sh (ar300 ba300 bb300 bc300). g41fd sweep (16 targets) still running on GPUs 0/2 (record only). Page v51.

**Update 2026-09-27 00:15 (server 15:10 CDT) — the knob has TWO makers, both measured:** (1) the tip-bound column's lead
(the transport's stretch): the coherent pace (method.md §10.30, eq. 53, `--pace_coherent`; bd300) cuts the overtaking
68 → 19 % at fit 0.9771 (tip starved 7.3); the native plan blur (`--plan_native`; bc300) 10 % with tip 14.4 (det F transit
0.59, knob 1.6). (2) NECK-bound material overshooting into the top region — present under both, absent only with the
render channel OFF (bb300: no top material before t=0.4, tongue, but starved 6.1 and −0.008): the silhouette term pulls
whatever is nearest to the ear's outline. Lever for (2): `--render_paced` (§10.31, eq. 54: the render channel targets the
paced cloud's own silhouettes/shading, re-rendered per window) — be300 (300k) + g41rp (40k ×3) running (P255); the PAIR
bf300 = `--render_paced --pace_coherent` running on GPU 3 (P256: top region empty before t=0.3, overtaking ≤ 40 %, knob
≤ 1.3, tip ≥ 9, ≥ 0.974). 40k coherent pace (g41fc) fit-neutral (nefertiti −0.0030 at the bound), healthy. Videos for the
300k strip (ar300 bb300 bc300 bd300 be300) rendering (`bdbc_video.sh`, `bb_video.sh`). Disk 90 GB → cleanup 5 at 100 (candidates:
g41fq/fw/fx/fg/fy/fv/fc/rp npz, ag300/af300 npz). Probes: `ear_order_probe.py`, `knob_probe.py` (top region by END class).

**Update 2026-09-27 01:45 (server 16:40 CDT):** be300 (render paced, 300k) = fit 0.9754 (−0.002 ✓), order 19 %, taper, knob
1.37 borderline, tip starved 6.9; at 40k render_paced costs −0.005…−0.008 (g41rp) → a 300k-only flag. **bf300 (render paced +
coherent pace) = the best growth profile: taper at t=0.3, knob ≤ 1.2 throughout, tip 14.8, p1 0.895, no reversal; fit 0.9732
(−0.004).** The 300k strip of the single levers (ar300/bb300/bc300/bd300) shows no tongue by eye → the STREAM pace (method.md
§10.32, eq. 55, `--pace_stream`: the step scales with the fill one pace behind on the ray; binds 1–2 % from window 0).
RUNNING: bg300 (300k candidate form ar300+w_h1 + render_paced + pace_coherent; GPU 1; P257), bh300 (ar300 + pace_stream;
GPU 2; P258), bi300 (ar300 + w_h1 + render_paced + pace_stream; GPU 0; P259); videos be300/bf300 rendering for the strip
(`growth_montage.sh OUT ar300 be300 bf300 bg300 bh300 bi300`). Cleanup 5 done (98 → 84 GB). Page v52. Next: pick the 300k
form from bg/bh/bi (fit ≥ 0.974 + tongue by eye + tip ≥ 9), then the 300k dragon with it and the 40k gate for any 40k-applicable
flag (pace_coherent/pace_stream at 40k; render_paced is 300k-only).

**Update 02:45 (server 17:40 CDT):** read: bh300 (stream) fit ✓ tip 10 but the neck-bound knob stays (the render's pull);
bi300 (H⁻¹ + paced render + stream) over-fills the tip (21.5), knob 2.9 — H⁻¹ from the start pulls the tip-bound group early;
bg300 (H⁻¹ + paced render + coherent) knob 2.4 AND a 16-window reversal streak at the end → **H⁻¹ from the start is OUT of
the 300k form when a pace rule is present** (dossier 17.3: the knob's three makers). bf300's strip by eye: the early ear is
still thin (taper of a thin ear). New flag `--render_paced_onset` (the paced render target until the pin's onset; runner
mirrors h1_onset_pin). RUNNING: bj300 (paced→onset + coherent; P260), bk300 (paced + stream; P261), bl300 (paced→onset +
coherent + stream; P262: base slabs ≥ 0.8× when the top slab appears), 40k twins g41fs (stream) / g41ro (paced→onset) on
bunny nefertiti dragon (GPU 0; tables `gallery_table_g41fs.py` / `_g41ro.py`; P263/P264 = within ±0.003 of g41pw). Page v53.
Disk 97 GB → cleanup 6 candidates when ≥ 100: bb300/bc300/bd300/be300 npz (videos exist), bg300/bi300 npz (read).

**Update 03:45 (server 18:40 CDT):** the onset placement (bj300/bl300) restores the 40k fit (g41ro ✓) but at 300k the pin's
onset precedes the ear's growth end → the neck-bound overshoot returns (knob 1.5–1.9; bl300 ends with 6 reversing windows).
bk300 (paced render + stream) = bf300's growth at 0.9743, tip 8.8. g41fd 19/19 fit-neutral (record only); g41fs (stream, 40k)
−0.003…−0.004 ✗. NEW placement `--render_paced_conv` (the switch when the paced cloud's silhouettes are closer to the
target's than the morph is to the paced cloud's; permanent; no constant): bm300 (+coherent, GPU 2), bn300 (+coherent+stream,
GPU 3), g41rc (40k ×3, GPU 0) running — P265–P267. Cleanup 6 done (112 → 64 GB). If bm300 passes (growth of bf300 + fit ≥
0.974 + tip ≥ 9): the 300k bunny form = ar300 + render_paced + render_paced_conv + pace_coherent (no H⁻¹ from the start);
then the 300k dragon with it, and the 40k gate for pace_coherent (g41fc neutral) + render_paced_conv (g41rc).

**CLOCK: the true time is the server's = the user's PC (CDT). All "09-26/09-27" stamps in this file's updates from
"16:50" on were 2026-09-25 CDT (07:00–11:30). DECISION 2026-09-25 11:35 CDT: the 300k bunny form = ar300's form +
`--render_paced --render_paced_conv --pace_coherent` (bm300: 0.9741, tip 13.3, no top material before t=0.4, no knob,
p1 0.893, no reversal; bf300 the same form 0.9732/14.8). bn300 (+stream) 0.9772 but tip 6.1. g41rc: paced render is
300k-only (−0.006…−0.008 at 40k in every placement but the onset's, which loses the 300k growth). The user's read of the
videos: bk300/be300 good by eye (bk300 a bit better); the remaining visible defect = the surface shake from the Poisson
re-mesh (Poisson 0.0011–0.0022 vs mc 0.0005 same particles; 41–50 % free at the end) — mc videos of bf300/bk300 on the
page (v57); bm300/bn300 videos rendering (`bm300_video.sh`/`bn300_video.sh`). RUNNING: bo300 = the 300k dragon under the
slip form + the three flags (GPU 2; `verdict_bo300.txt`; P268). Next: the surface decision (mc vs temporally coherent
Poisson) and the free half at the end (the pin's criterion vs the early stop) as the remaining D1 item.

**Update 2026-09-25 14:15 CDT — USER DECISIONS:** (1) bm300 ACCEPTED as the 300k bunny form ("flicker probably from the
optimisation, but fine"). (2) The delivered renderer: the user prefers the SPLAT render (scratch/render_splat_video*.py,
`render_3dgs` = PhysMorph-GS rasteriser) over every mesh: mc is stable but thick (the auto iso-level; iso 0.5 fixes the
thickness at tail 0.0007), Poisson shakes (0.0017), IMLS 0.0049 ✗, aniso mc 0.0024 ✗. The splat's defects to remove on
bm300's base: the dotted sparse regions mid-morph (frame 96: the stretched material's local spacing > the splat radius —
NOT the Gaussian count) → v2 opacity by support (n_i/(k/2)), v3 disc surfels, v4 adaptive radius (own shell radius /
target's); the surface roughness (density-gradient normals at 1.5 spacings). The user's caution: the sparse phase is the
physics made visible — judge the physics on the unhidden render. (3) The render loss is a particle splat (CIC, α=1−e^{−kw})
+ grid normals, NOT the reconstruction. (4) (b) sticky endpoints implemented (method.md §10.33, `--plan_sticky`): bp300
(bm300 + sticky, GPU 2; P269: pinned ≥ 0.75, mc tail ≤ 0.0005) and g41sk (40k ×3, GPU 0; P270) running; (a) waits. bo300
(300k dragon, chosen form): 0.9699 but end p1 0.77 (the free 38 % at the spikes). Cleanup 7 done (39 GB). Page v62.

**Update 15:15 CDT:** (b) sticky endpoints REFUTED (g41sk: no higher pin fraction, −0.001…−0.003; bp300: 7 % pinned at
window 40 — frozen competition for the same points) and (a) NOT pursued (the user's "engineering spiral" warning; the
free half at the end stays an open, stated item). Splat renders on bm300 (scratch/render_splat_video4.py = v2 script with
--adaptive/--disc/--blur): v1 0.0014, v2 (support opacity) 0.0014, v3 (discs) 0.0014, v4 (adaptive radius) 0.0016, **v5
(v4 + shading normals at 3 spacings) 0.0012 — the best splat, smooth**; the splat's stillness is bounded by the particles'
own drift (every splat sits on its particle), mc iso 0.5 averages it (0.0007). Frame 96's dots = splat radius vs local
spacing (not the Gaussian count) — gone in v4/v5. The delivered choice (splat v5 vs mc iso 0.5) is the USER's; page v65
has all videos + stills. Videos/stills in $OUT/r300/alt/. Next when the user decides: render the chosen form for bo300
(dragon) too; the open algorithm item = the free 40–50 % at the early stop.

**Update 17:40 CDT — the EMPTY TRANSIT measured (`scratch/hollow_probe.py`):** above the head at t=0.10/0.14 the material
is 5–8 % of the target's density (bm300, bq300), full by t=0.3 → the front's leads outrun the supply (continuity). The
stream pace (bq300: 0.9759, no reversal, tip 6.5) does NOT fix it (reads the dense-but-slow body behind the lead).
NEW: the SUPPORT pace (method.md §10.34, eq. 56, `--pace_support`: the step ∝ the density AT the particle over half the
target's own count at its nearest point) — bs300 = bm300 + pace_support running (GPU 2; P272: top-region density ≥ 0.5
at t ≥ 0.10, fit ≥ 0.971, tip ≥ 9). Also running: br300 (300k dragon, chosen form + stream pace, GPU 0), g41fc gallery
(16 targets, GPUs 0/1: the coherent pace's 40k adoption), splat v7 (render children ×4, GPU 3). Splat v6 (adaptive radius +
normals 3 sp + mesh lighting) = bright and smooth (page v66). The user's PhysGaussian split (surface Gaussians + interior
fill) does not transfer (the ears' surface is interior material); what transfers: render children (repo:
render/children.py) and adaptive shell-weighted sampling (later, gated). The user wants photorealism: typical 3DGS object
counts 100k–500k all on the surface vs our ~30k outer parents.

**Update 19:35 CDT — the delivered render is settled: `scripts/render_splat_gpu.py`** (discs + support opacity + adaptive
radius + normals at 3 sp smoothed over 32 NN ×2 + DEFERRED per-pixel hemispheric shading; 2 s/frame at 300k). Tails:
bm300 0.0008, g41pw 0.0002, bo300 0.0008. Page v71 = current results only (bm300, g41pw, bo300 + today's verdict table).
The vapour (first windows' sparse front, 5–25 % density): NOT the transport's to remove (bq/bs/bt300 all leave t=0.10–0.14
under half); the physics-side drag on unsupported material is PROPOSED, not run (the user's call). The surface flow =
the arrived-unpinned 47 % drifting 0.07–0.14 sp/frame → bp301 (sticky endpoints WITH capacity; P274) running on GPU 2.
g41fc gallery: cow −0.011 → the coherent pace is 300k-only; the 40k form stays g41pw. br300 (dragon + stream) worse →
the 300k dragon stays bo300. The user declined 100k screening; asked for faster turnaround → the GPU renderer.

**Update 20:50 CDT — sticky endpoints WITH capacity (bp301) + the pin at the stuck point (bp302, `--settle_pin_stuck`):**
bp301 0.9784 / bp302 **0.9798** (best 300k bunny), clean taper, no reversal, tip 8–11; but the pin stays 6–15 % (a stuck
particle converges monotonically = never "twice reversed"; and the reserved points are not reached — the material sits
2–4 sp off its point, creeping 0.05 sp/frame, half of bm300's 0.07). Proposed next (NOT run): a motion-based settle for
stuck particles (moves < one shell radius per window → pin). bp302's splat video on the page (v72) as a candidate; the
300k form is still bm300 unless the user takes bp302 (fit + growth better; the pin/creep open).

**Update 22:05 CDT — ROOT CAUSE measured (`scratch/strip_probe.py`, experiments 21:10):** the density objective's NULL
SPACE. (1) Early: the head top's own material (source depth 2.7 sp) makes no progress until t≈0.14 (even negative) while
the ear column's lead passes = the vapour; a filled region translating has equal cell sums, only the density-jump layer is
pulled. (2) Late: free surface particles move 0.18 normal / 0.13 tangential sp per frame (pinned 0.002/0.012); tangential
motion is free under density + silhouette. bp302 REJECTED by the user (looks worse: only 831 surface particles pinned).
Per-particle correspondence WITHDRAWN (leash falsified 3× in the log; detail dies). USER DECISIONS: (a) run the STILL pin
first (bp303 = bm300 + `--settle_pin_still`: arrived & moved < shell radius over 2 consecutive windows → pinned; GPU 2;
P276: pinned ≥ 0.90, surface move ≤ 0.02, fit ≥ 0.971), (b) then the NEIGHBOURHOOD-level correspondence (the plan's k-NN
set: its centroid pulled to its images' centroid; detail stays with the density/render terms), (c) the surface side
(PhysGaussian split) later: render children / adaptive shell sampling. The user asked about surface findings: none run.

## 2026-09-25 23:40 CDT update (after compaction)
- bp303 (still pin) REJECTED 22:50: pins 89 % by window 20 (before the transport passes), fit 0.9669, tip 23.1, 16-window reversal streak.
- bp304 v1 was inert (calibration inside the w_h1 block); v2 relaunched 22:57, DONE 23:09 (736 s, 44 windows). bp304 v2 REJECTED (P277 ✗): fit 0.9733 (= bm300), 0 reversals, but tip 8.4 (bm300 13.3), knob 95 vs 36 particles at t 0.5, vapour identical (top density 0.06/0.12 at t 0.10/0.14), late unpinned surface motion 0.224/0.144 (bm300 0.179/0.131). Reading: the 64-NN neighbourhood (0.227 wu) and the loss cell (0.306 wu) are the same resolution -> same null space as the paced cell sum. bm300 stays the 300k form. bp304 NOT on the page (refuted runs stay off it).
- Found: runner.py passes `log=lambda *_: None` to optimize_window -> h1/jdens/corr calibration lines never reach the run log; the corr line now uses print(flush=True).
- 40k vs 300k (hollow_probe/strip_probe on g41pw_bunny vs bm300/bp304): the vapour exists at 40k at ~3x the density (0.22/0.30 at t 0.10/0.14); the head-top-bound skin starts moving at t 0.05 at 40k vs ~0 until t 0.14 at 300k; same pace (0.306 wu/window), 45 windows x ~19 frames both, same u-gate opening; arrived share +19 pts by window 6 at 40k vs +9.5 at 300k => the 300k vapour excess = a DELAYED START, not stronger stripping.
- P278 pre-registered (experiments.md 23:35): plan-vs-delivery probe (`$SC/plan_probe.py`, dump hook `PHYSMORPH_DUMP_PLAN` in optimizer.py, chain `$SC/plandump_chain.sh`, output `$SC/plan_probe.txt`): 300k with disc_ref (plan300d), 300k without (plan300n), 40k (plan40), one window each on GPU 1. (i) plan blur doubled by disc_ref damps the skin's paced step, or (ii) the control delivers less of the same step at 300k.
- Probe tags take the FULL name (`hollow_probe.py bp304_bunny`, not `bp304`).
- Server 83 GB. Deletable refuted archives (logs to archive first): bq300, bs300, bt300, bp301, bp302, bp303, bp304, bp304v1 (~30 GB), br300_dragon (7.4 GB) — not deleted yet (under 100 GB); propose to the user.
- Surface side: still nothing run; queue = render children + appearance model, adaptive shell-weighted sampling.
- 23:45 CDT: P278 READ (`$SC/plan_probe.txt`): the plan asks the full pace (0.306 wu = one MPM/loss cell) of the head-top skin in all three (300k±disc_ref, 40k); plan blur irrelevant (hyp. i refuted). The window DELIVERS 0.064 wu (0.19 of the step along the plan) at 40k vs 0.037 wu (0.09) at 300k for the skin -> hyp. ii stands: the halved delivery at 300k is the open measurement (184 ppc vs 24.5 on the MPM grid; ctrl_rprop_k 8; u gate onset).
- P279 (bp305 = bm300 + --loss_res 128) WITHDRAWN: inert under --loss_units density (disc sets loss_res = MPM grid 36; loss cell = dx = 0.306). Stopped at window 1, files removed. Structural routes needing the user's decision: (a) decouple loss grid from MPM grid at 300k (72^3, 0.153 wu; pace = max(blur 0.227, cell)); (b) find/remove the cause of the halved delivery. No dx cut (user rule).
- pgrep/pkill LESSON (bit me twice tonight): the ssh command line itself contains file names like bp305_bunny.log / bp305_chain.sh, so `pkill -f "bp30[5]_chain"` kills the ssh session (exit 255). Put kill logic in a server-side script whose name doesn't match, then run `bash script.sh` in a separate ssh.
- GPUs all free at 23:45 CDT; server 83 GB.
- 23:35 CDT: read docs/gaussian_surface_physics_proposal.md (material-bound surface Gaussians + volumetric MPM). Measured its one testable separation (`$SC/normal_probe.py`, `$SC/normal_probe.txt`): at t 0.9-1.0 the settled 300k surface moves 0.004 sp/frame but the REFIT normal turns 2.6 deg/12 frames (30 % > 5 deg) vs 0.47 deg (2 %) transported by the fixed material neighbourhood's affine map -> the late shimmer at 300k is ~5x a renderer refit artefact. 40k: both 0. Verdict: implement material-bound normals/covariances (anchored at first exposure) in render_splat_gpu.py = renderer-only fix for the settled-surface shimmer; NOT a fix for the material flow (unpinned 47 %) or the vapour; the Gaussian-loss path (gauss_loss.py CUDA rasteriser, x4 children, constant colour) can't control sub-cell arrangement (184 ppc grid). Awaiting the user's go.
- 2026-09-26 00:30 CDT (after the user's review of the plan): P280 REFUTED (finer cell sum = noisier; 36^3 sees plan 0.5 sp; 72/144 fail) -> P281/P282 withdrawn. P283 partial: render_splat_gpu --material_normals --material_size -> bm300 tail 0.0008 -> 0.0006 (opt-in, page unchanged). P278b replay hook in optimizer.py (PHYSMORPH_REPLAY_SAVE/LOAD/MAP=direct|grid/U/OUT; iters skipped; commit rollout delivers); chain `$SC/replay_chain.sh` (+ `replay_chain2.sh` ingredient removal), probes replay_probe.py / ctrl_probe.py / band_probe.py (WORLD-unit depth bands 0-0.14/0.14-0.35/0.35-0.70 wu; native bands misled once). Round 1: R2=R1 exactly; u inert in window 1; grid projection -7 %; same grid control at 300k -> skin 0.0230 wu (36 % of 40k's 0.0632), along 0.045 vs 0.190; 300k own control 0.0245/0.012; |dFc| median 2.00e-2 = dfc_clip at both N, coherence 0.96-0.99 both -> FORWARD response difference, not cancellation/step scaling. Round 2 = remove layer_relax / disc_ref / bonds / layer_ctrl / bm300 flags one at a time under the same control.
- 2026-09-26 01:05 CDT: P278b rounds 2-3 DONE (58f0c86). Round 2: removing layer_relax/disc_ref/bonds/layer_ctrl/bm300 flags changes nothing (0.0230 to 4 digits); dt 1/240 + T 20 same at both N. Round 3 (PHYSMORPH_REPLAY_GRID=36, ppc swap): per PARTICLE LAYER (`$SC/layer_probe.py`) the outermost layer moves 0.057 (300k) vs 0.068 (40k) under the same control = 84 %; the excess zone = 2 particle layers at every N (0.07 wu at 300k vs 0.14 at 40k); the outer-layer response ~1/dx (dx 0.60: 0.023, 0.30: 0.06-0.07, 0.16: 0.103), independent of ppc/N at fixed dx; bulk (2-4 sp+) ~0.02 wu everywhere. 300k own control: outer 0.042 (62 % of the 40k control's effect on the same cloud), along 0.11 vs 0.17. CAUSAL CHAIN of the vapour complete: pace one cell ahead -> clip-saturated stress control -> MPM surface traction moves the outer 2 layers 2.5-3x the bulk -> those layers fill the lead = vapour of 2 layers' mass (half at 300k). Remedy must be on the CONTROL side (move the bulk by body forces, not the surface's stress jump) = formulation change -> user decides. GPUs free; server ~85 GB.
- 2026-09-26 02:45 CDT (reviewer's smaller-experiment order, all done except P284): item 1 (`accel_probe.py`, --T 1 replays): the first MPM step's acceleration sits in the OUTERMOST particle layer (x200 at 40k, x50 at 300k vs the deep bulk), the SAME under synthetic uniform 0.02 I / e_y e_y^T controls, independent of dx, scaling with the particle spacing (128-134e-4 wu at sp 0.070 vs 49e-4 at 0.035). Item 2 (`slab_probe.py`): slab fill per point alike (0.18 vs 0.16 head-top half), supply zone 0.15 wu (40k) vs 0.09 wu (300k), bulk along-plan progress 0.098 vs 0.029 (42-50 % backwards at 300k). Item 3 (`--pace_lead` = lead decoupled from arrival radius; `$SC/lead/`): 6-window runs, top density at window 6: 0.05 (baseline) -> 0.29 (1 sp) / 0.23 (2 sp) / 0.11 (4 sp), arrival 17 % slower. P284 pre-registered: bp306a (--pace_lead 0.035, GPU 1), bp306b (0.07, GPU 3) full runs; read with `hollow_frames.py TAG 76,114,152,190` (equal physical time), verdict_bp306a/b.txt. Corrections recorded (eec515e): same-cloud 74 %/65 %, stress-control over-generalisation withdrawn. P283 split into --material_size / --material_support (renders bm300_mat3s/mat3p on GPU 0; still_cov.py for the ear-base crop coverage).
- 2026-09-26 03:40 CDT: P284 HOLDS. bp306a (--pace_lead 0.035 = 1 native sp): silIoU 0.9792 (+0.006 vs bm300), vapour at frames 76/114 0.33/0.32 (bm300 0.17/0.09), 0 reversals, tip 11.3, 58 windows/13 min, splat tail 0.0006; bp306b (0.07): 0.9739, 0.30/0.28, tail 0.0005/0.0004, late surface 0.112/0.056. Rule flag `--pace_lead_sp 1` (lead = one native spacing of the source = 0.035 at 300k, 0.070 at 40k; arrival radius unchanged). ADOPTION GATES RUNNING: P285 bp306d (300k dragon bo300 form + rule, GPU 3, chain bp306d_chain.sh) and P286 g41ld (19-target 40k gallery, gallery_sweep_g41ld.sh on GPUs 0/1, table gallery_table_g41ld.py). If both pass -> adopt the rule (300k form = bm300 + --pace_lead_sp 1; 40k form g41ld), re-render, update the page (v73) with bp306a/b + dragon + gallery, cleanup (server ~95 GB now: bp301-305, bq/bs/bt, bp304v1, br300, scratch/replay + lead dirs deletable after archiving logs).
- 2026-09-26 02:45 CDT (server clock; my docs stamps 02:10-03:50 earlier were ahead, corrected in experiments.md): P285 ✗ (dragon + fixed 1-sp lead: stall 89.5 %, silIoU 0.9655 vs 0.9699, detF 0.137, 233 windows/64 min), P286 ✗ (40k gallery g41ld: 14/18 lower, bunny -0.0057, nefertiti -0.0064, detF much worse, windows x1.2-3.8; C crashed on latent arrive_idx_np bug -> FIXED in optimizer.py, synced). VERDICT: a FIXED lead holds only on the 300k bunny (P284 ✓: bp306a 0.9792, vapour 0.33/0.32); long transports starve. P287 PROPOSED (user decides): delivery-tied lead per window (median delivered progress along the plan of the moving set fixed at the window start, clamped [1 native sp, loss cell], fixed within the inner solve, hold on rejected/zero-delivery windows with a cap, arrival radius unchanged). Forms unchanged (bm300, g41pw); page unchanged. Cleanup: g41ld x19 + bp306d npz deleted (logs_archive_20260926f), server 54 GB. bp306a/b archives kept (evidence). All code synced (pace_lead / pace_lead_sp / replay hook / C fix). GPUs free except a bp306d chain leftover on GPU 3 (check pgrep -fa "bp306[d]").
- 2026-09-26 03:05 CDT: reviewer round 3 corrections applied (24b4ce4, +first-step redo). The T=1 "kick" was the layer projection at coefficient 1/T=1; at T=20 with a dFc=0 control (`step1_probe.py`, x_step1 saved by the replay hook) the projection alone = 6.5e-4 (40k)/2.5e-4 (300k) on the outer layer, the stress response 3.8/2.3e-4 (x6 / x2 the bulk). P287 as written has NO expansion (clip(d_w, s, h) with d_w <= l_w stays at s) -> needs an explicit expansion test; reviewer's order: (1) same-start-state lead comparisons at representative states (bunny initial, dragon mid-transport, a degraded 40k target) reading delivery/supply/local deformation, (2) expansion rule, (3) three gates together. stats["pace_lead_applied"] now records the applied lead (stats["pace_r"] = arrival radius). P285 = failed run but not an exact 1-sp test (lead bug after window 145); one fixed policy's failure != all fixed lengths. slab_probe = near-count share, not a flux.
- 2026-09-26 11:05 CDT (the user's go came ~10:15 CDT; the P288 docs stamp "03:30" is wrong, corrected in the log): P288 DONE (`$SC/p288/`, lead_response.py, fill_probe.py; --pace_lead_from switch works). Lead sets speed monotonically: dragon W=30 progress at W+12 cell 0.44 / 4sp 0.25 / 2sp 0.21 / 1sp 0.20, slowest decile 0.18 -> 0.02; nefertiti W=20 cell 0.79 / 1sp 0.60 / 2sp 0.48 and THINNER under small leads (0.16 vs 0.13); bunny initial cell 0.72 / 1sp 0.64. Per-window slab-fill signal = 0.05-0.19 everywhere (null). Next = P289 design (tested-candidate governor: candidate lead kept only if progress-vs-final and thinning don't worsen; the signal = accumulated transit density, a state) -> written for the reviewer, not launched. GPUs free; p288 npz kept (~8 GB), server ~62 GB.
- 2026-09-26 12:40 CDT: P290 (surface taper) ✗ at replay (outer/bulk 2.8->2.3-1.9 while the bulk collapses); P289 (tested-candidate lead governor, --pace_lead_gov) ✗ on bunny (tip 11.1, 63 windows; dithers <=0.14 wu through the growth) and dragon (0.9652, early stop 96.8 %, 3-window shrink cycle). USER'S EYE: bm300 looks better than bp306a (earlier, pointed ears) -> gates now include tip >= 13, windows <= 1.2x, ear onset not later (recorded 7b70c4c). Render-side stretched splats: visually null at frame 96. VERDICT: within the current formulation the lead and the taper are exhausted; bm300/bo300 stay the 300k forms; page unchanged. DECISION PENDING: body-force control channel (formulation change, days; keep render-off + dFc controls) vs accept the transit haze and move on (surface budget/photorealism, dragon end fit, late shimmer). Best video for the user: output/r300_page/video/bp306a_splat9_s12.mp4 (they prefer bm300's look). Server ~65 GB; bp308/bp308d npz kept (refuted, deletable). GPUs free.
