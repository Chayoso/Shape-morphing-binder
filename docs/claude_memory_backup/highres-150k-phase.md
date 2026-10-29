---
name: highres-150k-phase
description: "150k gallery phase — state as of 2026-09-19 09:00: v8 no-net gallery published; surface-smoothness round: bulk-density bug found, Poisson-on-outer-layer surface chosen, three Poisson videos rendering; what is running and what is next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-19T06:20:44.687Z
---

**Goal (user, 2026-09-16→17):** all targets at ≥150k particles, mass ejection removed algorithmically
(no per-shape constants, [[no-parameter-only-fixes]]), surfaces shown as objects, every run in the viewer.
All compute on hyde06 under /data ([[hyde06-ops-etiquette]], [[no-local-python]]).

**Cause (confirmed on 19 meshes, 2026-09-17):** the log cell-sum density loss rewards a lone surface
particle for reaching a far empty target cell; once it is one MPM cell from the body it shares no grid
node (numerical fracture) and plastic assimilation erases the restoring stretch. The variable that
decides it is the CELL SIZE RELATIVE TO THE SHAPE: dx 0.20 wu fractures at ppc 8 (40k) and ppc 27
(150k) alike; dx 0.31 holds at ppc 27 (40k) and ppc 91 (150k) alike; 0.41 holds but loses detail.

**Contract v2 (commit 51cfbf3, recipe in scripts/ops/hyde06_env.sh):** `--cell_diag 26` → dx = source
bbox diagonal / 26 (0.31 wu on the 8 wu normalisation), ppc = N dx³ / V (25 at 40k, 91 at 150k).
docs/method.md §10.9 has the definition; docs/experiments.md 2026-09-17 the ladders. NOT yet deployed
to /data repo (n150y batch runs on `--ppc 91`, the same dx) — deploy after it finishes.

**Evidence:** 40k, no re-attachment, 19 meshes: end fragments 307 (ppc 8) → 11 (cell 0.31); 15 meshes
0, max 4 (beast). 150k re-attachments (the pops) v3 8023 → v5 6143 → v6 1308; v6 per target: bunny 0
teapot 0 heart 0 spot 1 A 16 V 29 armadilo 21 dragon 755 bob 486 C 0 (C never morphs: gate stop).

**Pages:** main 150k gallery = v6 https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1
(local output/report_h150y_page, md docs/highres150_v6_report.md); v5 comparison (ppc 27, cell 0.20)
https://claude.ai/code/artifact/2ee53cd2-4c65-4b63-bea1-9026d13d8f78; v4 comparison (ot_pace)
https://claude.ai/code/artifact/6b144784-69e7-4703-ba73-ac34f6b45724; new meshes (40k v3 page)
https://claude.ai/code/artifact/4b18fc06-c010-45ff-89f2-256e77516a1c — to be replaced by the 150k v6
new-mesh page.

**Falsified today (docs/experiments.md):** OT loss alone, transport leash v1–v3, ot_pace variants
(snap / residual / shape / hand-off), loss_res 32, linear residual (not a drop-in), shell-biased
sampling (C++ scheme; C++ immunity = its 8-cell grid), one iteration per window, OT solver was also
fixed (row normalisation, ε-scaling + L1 stop; `--phys_loss ot*` kept as options).

**New meshes 150k v6 DONE (16:20):** re-attachments v3 10648 → v5 1982 → v6 710 (cow 28, homer 5,
maxplanck 0, nefertiti 152, fandisk 13, ogre 15, beast 493, cheburashka 3, bimba 1); silIoU −1…−2.4 on
smooth meshes, +4…+17 on the ejecting ones. Contract v2 DEPLOYED (3b8c644). Page: report_n150y built on
hyde06 (narrative scratchpad/global_n150y.json), local output/report_n150y_page, md
docs/newmesh150_v6_report.md; published to 4b18fc06 (16:40; a publish to an artifact not read this
session is refused — `action: read` with the url, Read the saved file fully, then publish). Open item:
dragon/bob/beast/nefertiti still shed ~0.3–0.5 % of particles at 150k even at cell 0.31 (rate follows
something other than the cell; the 40k runs shed none). Nothing is running on hyde06 now; the recipe
is `--cell_diag 26` (deployed). Server output/ holds only h150y/n150y/h150q/n150q + report_h150p.

**FIVE GOALS (user, 2026-09-17 evening; "keep going until all five are done"):** (1) zero ejection at 150k +
PROOF the render gradient changes the physics; (2) C morphs; (3) no floating particles / particle-looking blobs
in any video; (4) photoreal renders; (5) proof that material properties change the trajectory.
State 19:00: C = runaway expansion from the hole (not compression: poisson 0.0/0.45 freeze identically); ot_pace
gives direction; a MERIT BUG froze ot_pace runs (recorded d_vol was the paced loss) — fixed 0df3267, cd5_C re-run
pending. 150k v6 shedding = episodic chunks of thin features (~80 particles); h150z_dragon (cell 0.31 + ot_pace)
running as the candidate. Material sweep mat_{base,soft(young 3e4),stiff(6e5),nu45,el(assim 0.1)}_{bunny,dragon}
on GPU 0; render-proof twins rp_phys_<T> (lambda 0) and rp_cut_<T> (--render_until 40) for dragon/bunny/bob at
150k queued on GPU 2; photoreal batch (scripts/ops/photoreal_batch.sh 2 h150y <10>) running -> report_h150y/<T>/
<T>_photoreal.mp4 + components.txt (frame QA). Probes: scripts/probes/material_trajectories.py, render_effect.py.
Builder shows the photoreal video + QA line when present.
19:05: C: global hand-off never fires (19 far cells = arm tips) -> CELL-WISE hand-off (26457cb, cfg.ot_handoff /
--ot_handoff): cells within one cell of the body carry the fixed target mass, others the paced mass. Trials cd7_C,
oh3_{bunny,dragon,bob} (40k no net), gh_{bunny,dragon,bob} (--grad_h1, cell 0.31, no net), h150w_dragon (150k,
grad_h1 + net, queued after H150Z). h150z_dragon (150k ot_pace cell 0.31): 210 windows, only 19 merges so far (v6
755) but 4 % arrived — quality TBD. Photoreal QA baselines: bunny 54/369 frames with >1 component (ear-tip blob
282-366, early isolated 18-108); dragon 182/255 frames (whiskers/horns render as separate blobs; max 8). New-mesh
photoreal batch queued after the 10-target one (marker PHOTON150Y <T> DONE). Log buffering: count "OT plan solved"
lines (flushed) for ot_pace progress, not "anim N/".
19:15 C chain of findings: cd7 (all near cells fixed) 0.72 = runaway again; cd8 (deficit cells only) 0.81/hole 3.3;
cd9 (OT-divergence merit, 1800d88) 0.90 — the BODY stalls (ot_div flat, 5 % arrived): pace 0.18 < loss cell 0.31 ->
cell sum blind to sub-cell shifts. FIX 7e34ba2: pace = max(blur, loss cell). cd10_C running (marker RT cd10_C DONE).
h150z (150k dragon ot_pace) FALSIFIED as recipe (0.675, 4 % arrived, 19 merges) — same stall; re-test at 150k after
cd10 if C works. grad_h1 falsified for goal 3 (gh_dragon 4 frag, stray_max 2.3 %). Material bunny: trajectories
diverge 1.2–2.3 spacings (soft peaks 4.4 at frame 250), end chamfer equal; stills montage output/photoreal/
material_bunny_f250.png; mat_ctrl (identical config) running for the noise floor; dragon sweep running.
Photoreal QA so far (frames with >1 component / total): bunny 54/369, dragon 182/255, bob 119/290, teapot 2/263,
heart 1/188, spot 11/193.
19:50 GOAL 2 ANSWER: C morphs with the PER-PARTICLE transport loss (--phys_loss ot --ot_debias + net): cd11_C
silIoU 0.9625 (density 0.72), chamfer 0.140, hole 6.3 %, 165 windows/17 min; cd10 (pace = cell) overshoots
(0.839). Cause chain: cell sum pushes outward from the hole -> inertial overshoot -> brake; NOT compressibility.
150k C launched (c150_C, marker RT c150_C DONE); 40k C photoreal -> output/photoreal/cd11_C.mp4.
Material dragon: stiff 0 reatt/0.963, base 14/0.957, soft 343/0.846 (stiffness changes ejection). rp_phys_dragon
(render off): 259 reatt/0.840 vs render twin 755/0.939. grad_h1 closed at 150k too (h150w 733/0.927).
o2 trio (OT-only bunny/dragon/bob no-net) + oh2/oh3 dragon/bob + rp_cut + mat nu45/el dragon pending.
20:00: --phys_loss auto (5386fe4; RECIPE uses it): transport loss when >50 % of source particles sit in
target-empty cells (C 68 % -> ot; others density). o2_bunny OT-only 0.945 (density 0.960) -> OT is C's regime only.
GOAL 3 deliverable rule (cbae691): render_photoreal drops isosurface components < 1 MPM cell (dx^3, dx=diag/26);
sidecar has raw/drawn/dropped counts. Verified on dragon frame 210 (floating chunk gone, 3 dropped). Photoreal
re-render of all gallery videos queued after PHOTON150Y bimba (marker "PHOTO RERUN DONE"). auto_C/auto_bunny
running (marker RT auto_<T> DONE); c150_C (150k, OT) running.
20:10 KEY RESULT oh3_dragon (ot_pace + deficit hand-off + pace=cell + OT merit, 40k no net): 0 frag, 0.965,
stray_max 0.14 % (density 2.3 %) — expansion strays removed at the source; oh3_bunny 0.961; oh2_dragon 0.965.
auto rule updated in code e07e0b5 (overlap -> ot_pace + hand-off; hole -> ot) — COMMITTED, NOT DEPLOYED (deploy
after oh3_bob passes; then 150k v7 batch of all 19 with RECIPE (auto) + --reattach --archive_stride 8).
GOAL 1b render proof dragon: cut twin diverges 0.58 sp before window 40 (noise) -> 4.7 after; phys-only 6.0;
render share of step 35 %; outcomes 0.939 vs 0.840 vs 0.770. rp_ctrl_dragon (identical config) running for the
noise floor; bunny/bob twins queued. Plot output/photoreal/render_effect_dragon.png.
20:20: oh3_bob 0 frag/0.974/stray 0.29 % -> TRIO PASSED. DEPLOYED 354740d (auto recipe). 150k v7 LAUNCHED 20:00:
chains chain150v7_A..D (GPU0: h150v7 bunny armadilo dragon; n150v7 cow homer maxplanck | h150v7 spot bob; n150v7
nefertiti fandisk; GPU2: h150v7 teapot heart A; n150v7 ogre beast | h150v7 C V; n150v7 cheburashka bimba), markers
H150V7/N150V7 <T> DONE, media report_h150v7/<T>, report_n150v7/<T>. Then: photoreal_batch for both (sub-cell rule),
fetch, global.json v7 narrative (five goals), publish 2f348b78 + 4b18fc06, docs, Korean summary.
Still running: rp_ctrl_dragon, rp_phys/cut bunny & bob (goal 1b), c150_C (goal 2, OT at 150k — v7 C uses
auto=ot too), auto_C/auto_bunny, photoreal re-render (PHOTO RERUN DONE), v7 photoreal queued (PHOTO V7 DONE).
GOAL 5 DONE (docs): bunny + dragon, 5 materials + identical-config control; divergence 8-130x noise early, dragon
3-8 spacings at the end, soft sheds 25x more; stills output/photoreal/material_{bunny,dragon}_f250.png, plots
material_{bunny,dragon}.png. docs/method.md 10.10 written (paced cell sum + hand-off, loss regime, deliverable rule).
v7 narrative draft: scratchpad/global_v7_draft.json (placeholders __...__).
20:35 GOAL 1b CORRECTED: rp_ctrl_dragon (identical config) diverges from h150y_dragon exactly like the cut twin
(0.58/1.41 before 40 | 3.33 | 4.66 vs 0.58/1.41 | 3.38 | 4.72 spacings) -> trajectory divergence at 150k is CHAOS,
not a causal signature (claim withdrawn in docs/experiments.md). Proof rests on (i) per-window g_share 0.35 /
g_cos 0.02 (deterministic; render = 1/3 of every control update, orthogonal to the cell sum) and (ii) outcomes
vs the run-to-run spread 0.021: silIoU 0.939/0.918 (render) vs 0.840 (phys-only) / 0.770 (cut). Control outcome
773 reatt / 0.0876 / 0.918.

20:50 C at 150k (h150v7_C 0.946 / c150_C 0.954) MORPHS but sheds: 1914 / 2675 re-attachments in bursts of 100-440
(windows 38-50), stray peak 14.9 % (40k: 109-114, 6.5 %). CAUSE in code: the `ot` regime's per-particle target was
the raw map image (no material-kNN smoothing, no pacing) -> quadratic pull ∝ distance -> farthest-behind particles
(arm tips) lead and fracture. FIX efc7fa0 (deployed): ot target = material-smoothed displacement bounded to one
pace = max(plan blur, loss cell) per window. Tests op40_C (net) / op40n_C (no net) on GPU 0 (markers RT op40_C /
RT op40n_C DONE); compare to auto_C 114 reatt / 0.959. If it passes: re-run h150v7_C via run_batch (post_run),
then 150k. nn150 trio (150k, no net, bunny/dragon/bob; marker NN150 TRIO DONE) running on GPU 0 = goal-1a proof.
auto_bunny (40k, ot_pace+hand-off, net): 0 reatt / 0.967 / stray 0.05 %. Frame QA table + re-attachment column
in build_report150.py (4c30150; post_run writes <T>_reatt.txt; _figs/ folder for shared figures). v6 photoreal
frame check: isolated-particle peaks (A 33: 2160, dragon 48: 5286, bob 48: 4446) are the stretched early shell —
videos show ONE body, no floaters; the deliverable measure is "drawn components > 1".

21:00 PACED OT VERDICT 40k C: op40_C 7 reatt / 0.958 / 10.7 min (auto_C 114 / 0.959 / 34 min), G2 guards PASS;
op40n_C (no net) 9 fragments / 0.944. ADOPTED (docs 93d2fa1: method 10.10 + experiments). 150k C RE-RUN as h150v7_C
(run_batch on GPU 2, overwrites the 1914-reatt run; marker "H150V7 C DONE" appears twice); nn150_C queued after the
nn150 trio. v7 photoreal waiter RELAUNCHED to count unique targets + require the 2nd C marker (photo_v7_waiter.log).
v7 so far: teapot 0 reatt/0.970 (photoreal QA: drawn>1 in 0/130 frames), heart 0/0.979/stray 0.025 %; old C video:
drawn>1 in 116/199 frames (the shed chunks ARE visible -> the re-run must fix it). NOTE the name op40 collided with a
morning ladder (op40_<19 meshes>, markers 08:21-09:35 AM); op40_C.log/json were overwritten by the C test.

21:20 v7 so far: teapot 0/0.970, heart 0/0.979, spot 3/0.972, bunny 2/0.966 (v6 0.958) — chains slow (60-75 min per
150k mesh under 3-4 procs/GPU; gallery ETA ~01:00). rp_cut_bunny ran under auto=ot_pace -> twin of h150v7_bunny:
0.936 vs 0.966; divergence flat 0.14 sp before the cut, 0.77 after (plot render_effect_bunny.png, cleaner than
dragon). Queued on GPU 2 after rp_cut_bob: rp_phys7_bunny (λ=0, v7 recipe), rp_ctrl7_bob, then rp_ctrl7_bunny
(controls for the outcome spread / noise floor). Artifacts 2f348b78 and 4b18fc06 READ this session (publish OK).
Builder has render_html/material_html sections (8458e60); draft global_v7_draft.json validated on hyde06
(copy at output/global_v7_draft.json); _figs staged in report_h150v7/_figs (material_*, render_effect_*).

21:45 nn150_bunny (150k NO net): 0 fragments / 0.962 / G4 PASS = goal 1a proof for bunny; nn150_dragon running.
v7: V 1/0.969. C 150k: the PACE DID NOT TRANSFER (h150v7_C re-run 1393 reatt by window 111). Frames of the old
150k C video: a C-shaped SHELL with the top arm a separate slab + a floating ball at frame 162 — the map
discontinuity (material splitting between arms) tears the body at 150k (blur nbhd 0.10 wu = 1/3 cell). FIX: ot
displacement resolved on the LOSS GRID (CIC deposit/gather) before pacing; test c150s_C (150k, net, GPU 0; marker
RT c150s_C DONE). If it fails too: fall back to ot_pace+hand-off at 150k for C, or accept C with the net + note.

22:10 *** THE 150k SHEDDING MECHANISM = DOMAIN BOX TRAP *** (docs/method.md 10.11, commit 68f0a20 DEPLOYED, tests
26 passed): the MPM grid had no wall BCs; particles within 2 cells of the box edge (truncated stencil) freeze; the
auto domain (leash + 2 dx) puts that band 1 cell beyond the target; 150k transients reach it. "GUARD clamp" (n_out,
particles clipped at the box) tracks re-attachments in EVERY run (v6 dragon 10706/755, bob 10934/486, C 27-59k/
885-1914; 0/0 where never reached). Chunks were static at the box corners (probe output/chunk_origin.py). FIX:
separating walls (WALL_NODES=2 = stencil half-support). Grid-resolved ot displacement (f5c8849) FALSIFIED/reverted
(c150s_C 0.59). c150p_C (ot_pace on C) 0.56. RUNNING: c150w_C (150k C walls+paced, GPU 0, marker RT c150w_C DONE),
h150v7_bob re-run under walls (GPU 2, run_batch). Remaining v7 chains run under walls from here; finished targets
with 0 band hits stand. TODO after: copy the winning C into h150v7_C (+post_run), maybe re-run dragon v7 if it ran
pre-walls (chain A: armadilo started pre-walls, dragon after?) — check "domain" line timestamps vs deploy 22:05.

23:00 PACED OT TARGET WITHDRAWN (commit after d17f1bd, deployed): c150w_C (walls+paced) 0.550 — a target that walks
with the particle starves the line search (150k paced 0.88/0.55 vs unpaced 0.95). Walls keep the box trap away
(band hits 38983 -> 993). bob under walls 0.969/126 reatt (its own thin feature, sub-cell events). RUNNING:
c150u_C (150k walls unpaced smoothed ot, net; GPU 0) = the v7 C candidate; ou40_C (40k same). nn150 (no net,
pre-walls): bunny 0 frag/0.962, dragon 3 frag/0.952 (single particles, 0.002 %); nn150_bob running, then nn150_C.
Bob render-proof triple (pre-walls, v7 recipe): 0.970/0.952/0.956, cut-twin divergence 0.054 -> 0.57 spacings
(10x step; pre-wall render-on archive kept as h150v7pw_bob.json/npz). rp_phys7_bunny running, rp_ctrl7_bob,
rp_ctrl7_bunny queued. v7 done: teapot heart spot bunny V A armadilo(28 reatt) bob(walls) cheburashka(6);
remaining chains: dragon cow homer maxplanck (A), nefertiti fandisk (B), ogre beast (C), bimba (D).
FINAL C PROCEDURE: copy c150u_C.{log,json} + npz -> h150v7_C names, run post_run.sh h150v7 C <gpu>, then the
photoreal batch (waiter requires 2 "H150V7 C DONE" markers — already satisfied) renders it.

00:10 `ot` regime BACK TO THE RAW MAP IMAGE (e87b97d, deployed): kNN smoothing on a hole target 0.79 (ou40_C);
or40_C (40k walls raw) 0.959/101 = baseline restored. 150k C with walls+raw (c150r_C) still sheds ~100-particle
chunks (525 reatt by window 104; band hits 990 early only) — C at 150k remains the open item; the v7 C entry will be
the best netted 150k run (c150_C 0.954/2675 pre-walls or c150r_C) with the count stated. Goal 1b DONE on 3 shapes:
per-window share 0.35 / cos 0.02-0.08 + outcomes vs spread (dragon +10 @0.021, bunny +3.4, bob +1.4-1.8 @0.002);
trajectory signal on bunny/bob (cut 0.14->0.77, 0.054->0.57 vs bob floor 0.11->0.34), chaos on dragon. Probe has
--ctrl/--ctrl_ref (8588608); plots in report_h150v7/_figs. nn150 no net: bunny 0 / dragon 3 / bob 54 fragments.
h150v7_dragon (walls) 3 reatt @144 windows (v6 755). User rule saved: [[cleanup-after-v10]].

2026-09-18 07:40 STATE: v7 gallery COMPLETE (19 runs + photoreal 02:55); C entry = c150r_C (walls, raw ot: 0.940 /
530 reatt; copied to h150v7_C names, post_run + photoreal done 07:15). v7 totals: 18 w/o C 284 reatt (v6 2018), mean
silIoU 0.951 -> 0.962, dragon 755 -> 7. nn150 no-net: bunny 0 / dragon 3 / bob 54 / C 74 fragments. Controls done:
rp_ctrl7_bunny 0.969 (spread 0.003), rp_ctrl7_bob 0.968 (0.002). GOAL 3 FINDING: cow video 'floating ball' (62
frames) = CONNECTED material (one cluster at every cell scale down to 0.07 wu; probe output/blob_probe.py) whose
2-particle neck fell below iso 0.5 -> render_photoreal `--iso auto` = 2 s^2/(pi sigma^2) (0.283) (69d4457, method
10.10) — verified on cow frame 522 (1 component). RE-RENDER of all 19 videos with iso auto running on GPU 0 (two
photoreal_batch chains; markers after the "PHOTO V7B START" line; local waiter appends "PHOTO V7B DONE"). The other
user (nurshat) now holds ALL FOUR GPUs (0: 27 GB, 2: 44 GB) — only light renders fit. NEXT: fetch report_h150v7
(10) + report_n150v7 (9) + _figs + global.json -> build on hyde06 (build_report150.py <root> <title> <md>) -> fetch
index.html -> publish v7 to 2f348b78 (main) and n150v7 page to 4b18fc06 (both READ this session — re-read after a
restart!), md reports docs/highres150_v7_report.md + newmesh150_v7_report.md, final Korean summary. Draft narrative:
scratchpad/global_v7_draft.json (only __GOAL3_QA__ left; copy at output/global_v7_draft.json validated).

08:50 GOAL 3 RENDER RULES FINAL (c916743 deployed): iso auto (0.283) + `--bridge` filament rule (particles the
isosurface does not enclose but which link the body to a drawn component are drawn as a 1-spacing filament; union-find
over free particles + enclosed-component super-nodes). Verified: cow 372/345 stills show the teat bulb on a thread
(bridged 1). Cause: cow teat = 72-particle bulb ON the target tied by a 1-particle chain (one body at every cell scale
to 0.07 wu; blob_probe.py). grid_fragments.py (physical fragments >= 1 cell, net's criterion): 0 frames in 18/19
targets, C 6/202 frames (max 1.24 cells); sidecars <t>_gridfrag.txt in report folders; builder column added
(01a385a). The iso-auto re-render (PHOTO V7B, no bridge) is finishing; NEXT: launch the FINAL bridged re-render of
all 19 (photoreal_batch on GPU 0, two chains), then fill __GOAL3_QA__, copy global.json -> report roots, build, fetch
(remote tar excluding *.gif and *_pbr_az*.png), publish both artifacts, md reports, Korean summary.

09:40 CAVITY RULE (eaccdc0 deployed): bunny's extra 'drawn piece' (37 frames) = interior cavity in the ear (signed
volume opposite to the body) -> renderer removes/counts cavities apart; cavity_sweep.py writes <t>_cavity.txt for the
older sidecars (marker CAVITY V7 DONE; local waiter bxrtgohke); builder subtracts cavities (parse_qa cav_txt).
Bridged re-render (PHOTO V7C; waiter b4ef3q2dk) in progress on GPU 0. fetch_page.sh <prefix> fetches a built page
(remote tar, no gifs/pbr pngs) into output/report_<prefix>_page. Drafts: global_v7_draft.json / global_n150v7_draft.json
(scratchpad; copies at $OUT); only __GOAL3_QA__ left in both. Test build of both pages succeeded (QA table with
physical-fragment + bridged columns).

10:30 FINAL RENDER RULES (all in scripts/render_photoreal.py, deployed): iso auto (2-particle filament level 0.283)
+ mass rule (component drawn iff >= ppc particles inside, by voxel-label membership; plus >= dx^3) + interior
cavities removed/counted apart (sign of the signed volume) + filament bridges (particle connectivity). Cavity sweep
done (bunny 90 frames = ear hollow). Bridged QA: unbridged pieces 0 for 17 targets, bunny 3, C 53 (marble-sized
compressed chunks -> the mass rule); cow 22 bridged, beast 1. Re-render of C, bunny, cow, beast under the mass rule
running (marker PHOTO V7D DONE). Then: fill __GOAL3_QA__ in both drafts, cp to report roots, rebuild, fetch_page.sh,
publish 2f348b78 (main) + 4b18fc06 (new meshes), commit md reports, Korean summary.

2026-09-18 11:10 *** V7 PUBLISHED — FIVE GOALS DELIVERED *** main gallery v7 -> 2f348b78 (figs/ folder, not _figs:
a published path may not start with "_"), new meshes v7 -> 4b18fc06; md docs/highres150_v7_report.md +
docs/newmesh150_v7_report.md; local pages output/report_{h150v7,n150v7}_page (fetch_page.sh). Final deliverable
QA: detached drawn pieces 0 frames for 18/19, C 29/203; physical fragments >= 1 cell: 0 for 18, C 6. Open items for
the next round: C's arm-front chunks (needs the net; 74 fragments without it), bob's ring (~50-particle chunk without
the net), clean single-run speed numbers, and the user's [[cleanup-after-v10]] sweep once v10 is passed. Nothing is
running on hyde06; the other user holds all four GPUs (only light renders fit on GPU 0).

11:20 CLEANUP DONE (user asked now, not at v10): server 244 -> 109 GB (69 runs' npz deleted, logs archived in
output/logs_archive_20260918.tgz; live/ pruned to the 19 v7 runs = 44 GB; old report folders removed); local
output/ = only the two v7 page folders (35 MB). NEXT ROUND (user 11:15, six items): (1) ejection zero WITHOUT the
net, (2) C without the net, (3) net-free across materials (soft dragon 343), (4) surface bumpiness / visible
particles during the morph -> fix in the Gaussian renderer (covariance), FIVE HYPOTHESES first, (5) loss jitter
near the optimum: optimizer vs early stop, (6) camera: every mesh looks down (orientation/up-axis).

12:10 NEXT ROUND progress: item 6 = ASSET UP-AXIS (11/19 assets z-up: bunny spot nefertiti teapot dragon armadilo
heart A C V bob -> x-90; table assets/orientation.json; physmorph/sampling/orientation.py; loader rotates, archive
records `orient`, renderers rotate old archives at render time) — code edited, NOT yet committed/deployed/tested.
Item 5 = cfg.reject_stop=3 (early stop on 3 consecutive rejections) — edited, not yet tested. Item 3 = soft dragon's
343 reatt were the box trap (25660 hits); matw_{soft,nu45}_dragon + matw_soft_bunny (40k, walls, net) RUNNING on
GPU 0 (markers RT matw_*). Item 4 (surface bumpiness): plan = F-anisotropic density kernel (Sigma = sigma0^2 F F^T
from F_samples) in render_photoreal + 5 hypotheses (H1 blur below spacing irregularity, H2 thin features need
anisotropy, H3 low iso inflates, H4 surface offset/discreteness, H5 CIC grid aliasing flicker). Items 1-2: bob ring
sheds at windows 16-27 (expansion), C arm fronts — need runs (GPUs held by nurshat; GPU 0 has ~20 GB free).

13:30 items 5+6 DONE and deployed (7ac43e7, 43b6201): orientation table lives in physmorph/sampling/orientation.json
(deploy skips assets/); reject_stop=3. Item 4: renderer hypotheses FALSIFIED with numbers (F-carried Gaussians
bumpier: 19-22 deg vs 15; blur/grid variants <=10 %); bumps = loss-cell-scale STATE -> objective-side experiment
running: rv_bunny (--render_res 128 --render_views 12) vs rvb_bunny (base) at 40k on GPU 0 (markers RT rv_bunny /
RT rvb_bunny). Item 3: matw_soft_dragon under walls ~41 reatt (windows 23-37; pre-walls 343) — chain matw_* running
(waiter beui0jthx). Item 1: nn150_bob's 54 end fragments are NOT >=20-particle chunks (chunk probe found none) —
blob_probe with min_n 1 at frames 200/300/420 pending.

14:00 ITEM 1 STRUCTURAL FIX deployed (f410ee9, tests 26 pass): bond decoupling flag evaluated EVERY STEP (was fixed
at the window start -> single-particle leaders bonded a window late). Running on GPU 0: nnw150_bob then
nnw150_dragon (150k, no net, walls + per-step bonds; markers RT nnw150_*). Also running: matw chain (soft dragon
done: 41 reatt pre-fix; nu45 dragon + soft bunny carry the fix), rv_bunny/rvb_bunny (item 4 objective experiment:
render_res 128 x 12 views vs base). Docs: experiments.md "next round, first readings" section holds items 1-6.

14:40 RUNNING on GPU 0 (all 40k unless noted): matw chain (nu45 dragon, soft bunny), rv_bunny -> rvb_bunny (item 4
objective test), nnw150_bob -> nnw150_dragon (150k no net, per-step bonds), cc40_C (--continuity, historically
FALSIFIED v2 2026-09-16: ejecta move WITH their neighbours), ce40_C (--assim 0.1) -> ce25_C (--assim 0.25) (C fronts:
more elasticity holds the neck?). Waiters: beui0jthx (matw), be86m7xqx (C trio); Monitor bjdd4h1q6. KEY INSIGHT
(item 2, docs): the control is a STRESS -> an isolated chunk has zero net force -> detached material can never
return; "no net" == "no fracture" (prevention only). C chunks with walls sit 1.7-2.3 wu outside the target, static,
although the transport map still pulls (probe: 4 wu inward at 2-3 wu out).

11:20 CDT RESULTS: per-step bonds WORK — nnw150_bob (150k no net) 54 -> 1 fragment, 0.932 -> 0.950; matw_nu45_dragon
95 -> 0 reatt, 0.941 -> 0.967; matw_soft_dragon under walls 343 -> 41 (pre-fix bonds). rv_bunny (128px x 12 views)
= rvb (no end-shape gain); bumpiness stills pending (bg). Still running: nnw150_dragon, matw_soft_bunny, then the
C trio (ce40 assim 0.1, cc40 continuity, ce25 assim 0.25) after rvb (waiter byxqdu4v8). GPU 0 OOM'd at 6 of ours +
nurshat 27 GB: keep <= 3 concurrent runs.

11:55 CDT ROUND STATE: item 4 CLOSED with numbers — target clouds render at the same bumpiness (14.5 deg) as the
morphs: the texture is the 150k sampling noise, not the state; Gaussian/F-covariance route falsified (worse);
render-loss resolution no effect; cell-scale lumps need a surface-aware objective (future). Item 2 (C) OPEN: assim
0.1/0.25 and continuity all freeze C at ~20 windows; C keeps the net (ledger in docs). Items 1+3: per-step bonds
(f410ee9) -> bob 54 -> 1 (no net), nu45 95 -> 0, soft bunny 12 -> 1; nnw150_dragon ~1 fragment (finishing);
matwb_soft_dragon (walls + per-step bonds) launched (marker RT matwb_soft_dragon DONE) to close the soft-dragon 41.
Items 5+6 DONE. NEXT: final round summary to the user; later a v8 gallery re-run (walls + per-step bonds + early
stop + orientation) when GPUs free up; user's cleanup rule after v10 [[cleanup-after-v10]].

12:10 CDT ROUND DONE (docs/experiments.md "next round, first readings"): net-free 150k: bunny 0 / dragon 1 / bob 1 /
C 74; materials with net: soft dragon 1 (matwb), nu45 0, soft bunny 1. Nothing running on hyde06. Code state:
walls + per-step bonds + reject_stop 3 + orientation + renderer rules (iso auto, mass, cavity, bridge) — all
deployed; the v7 gallery pages were built BEFORE per-step bonds/orientation (a v8 re-run would carry them).

12:45 CDT PLAN A/B/C (user approved "진행해 줘"): A = loss_res ladder lr{64,96,128}_{bunny,dragon} (40k, running; waiter
by7y92169) + lump measure scripts/probes/lump_amplitude.py (surface-referenced; lumps ~0.01 wu = 1/3 spacing at 150k,
wide error ~1 spacing; the user's screenshot lumpiness was the z-up orientation). B = v8 gallery as a NO-NET pass of
all 19 (fragments = the ejection ledger), after A decides loss_res; <= 3 concurrent on GPU 0 (nurshat holds 27 GB).
C = documented as the limit case (method 10.10a). Ladder runs are 2 chains on GPU 0.

14:00 CDT A DONE: loss_res ladder — lump amplitude does NOT follow the cell (bunny 0.0084/0.0085/0.0085, dragon
0.0104/0.0105); recipe keeps loss_res 64; item 4 closed (texture = sampling floor; lumps 1/3 spacing; screenshot =
orientation). B LAUNCHED: v8 gallery = NO-NET pass (EXTRA="--archive_stride 8", no --reattach) of all 19 at 150k:
chains A (h150v8 bunny dragon bob teapot heart spot), B (h150v8 A V armadilo C; n150v8 cow homer), C (n150v8
maxplanck nefertiti fandisk ogre beast cheburashka bimba, starts after lr128_dragon). Markers H150V8/N150V8 <T>
DONE, media report_h150v8/<T>, report_n150v8/<T>. Then: photoreal_batch (iso auto + mass + cavity + bridge, oriented),
gridfrag/cavity sidecars not needed (renderer counts cavities itself now), build with a v8 narrative (fragments
column = the no-net ledger), publish to 2f348b78 / 4b18fc06, md reports. ETA ~8-10 h under contention.

14:10 CDT v8 RESTRUCTURED after OOM (150k run = 6.6-8 GB; nurshat 27 GB; GPU 0 47 GB -> TWO 150k runs max): chain A
(dragon bob teapot heart spot) + chain B (V armadilo C, then n150v8 cow homer) running; follow-ups via
output/v8_followups.sh: A2 (after "H150V8 spot DONE": bunny; n150v8 maxplanck nefertiti fandisk), B2 (after
"N150V8 homer DONE": A; n150v8 ogre beast cheburashka bimba). False markers of the OOM'd bunny/A were removed.
Photoreal v8 waiter counts unique markers after "V8 START" (10 + 9) then renders both prefixes -> "PHOTO V8 DONE".
Monitor by34dvmhu. PITFALL hit again: pkill -f with a pattern that appears in the ssh command kills the ssh shell.
Draft narrative: scratchpad/global_v8_draft.json (placeholders __FRAG_LEDGER__ __QUALITY__ __C_LINE__ __FRAG_TABLE__
__SUMMARY__).

17:20 CDT (session restarted twice; monitors re-armed): v8 no-net done for V 0/0.963, dragon 1/0.955, bob 9/0.949
(9 = single particles + one 9-particle cluster), armadilo 1/0.938, teapot (see docs). Running C + heart; chains
A/B + follow-ups A2/B2 + photoreal waiter all ALIVE on hyde06 (verified 17:18). ~14 targets left at 2 per ~50 min
-> ETA ~23:00 CDT. post_run.sh now writes <T>_gridfrag.txt (grid_fragments probe) — targets already post-processed
(V dragon bob armadilo teapot) need the probe run by hand before the v8 build.

19:45 CDT v8 10/19 done (V 0, dragon 1, bob 9, armadilo 1, teapot 0, C 3/0.905, heart 0, spot 0, cow 1, bunny 0).
nurshat LEFT GPU 0 -> follow-ups A2/B2 killed and replaced by 4 parallel run_batch chains: [maxplanck nefertiti],
[fandisk ogre], [A], [beast cheburashka bimba] (+ homer finishing) — 5 x 8 GB on GPU 0. Photoreal waiter (unique
markers after V8 START, 10 + 9) still armed -> "PHOTO V8 DONE". Then: gridfrag sidecars exist via post_run for the
new runs (backfilled for the first five; check heart/spot/cow/bunny/C were post-processed AFTER ae44525 deploy —
C/heart/spot/cow/bunny post_runs ran 17:38-19:40, deploy was 17:22, so yes). Build: fill global_v8_draft.json
placeholders (__FRAG_LEDGER__ __QUALITY__ __C_LINE__ __FRAG_TABLE__ __SUMMARY__), cp to report roots, build both,
fetch_page.sh h150v8 / n150v8, publish to 2f348b78 / 4b18fc06 (re-read both first — new session), md reports.

23:05 CDT v8 ALL 19 DONE (no net): 12 targets 0 fragments; dragon/armadilo/cow/beast 1; C 3; bob 9, ogre 21 (one
collapsed clump each); total 39 vs v7 814 netted / v6 2018; grid probe >= 1 cell: 0 frames everywhere. Quality
within 0.7 pt of v7 except bob -2.0, beast -2.6, armadilo -1.1, C -3.5. PHOTOREAL v8 batch RUNNING (started
22:57; 19 videos; marker "PHOTO V8 DONE"; monitor b6tc9fb38). Drafts DONE: scratchpad/global_v8_draft.json,
global_n150v8_draft.json (both to scp + validate). Both artifacts READ this session (publish OK). NEXT: after PHOTO
V8 DONE -> cp drafts to report_h150v8/global.json + report_n150v8/global.json, build both (titles "PhysMorph 150k
v8 — 안전망 없이" / "PhysMorph new meshes 150k v8"), fetch_page.sh h150v8 / n150v8, publish to 2f348b78 / 4b18fc06
(figs/ not _figs; the main page has no figs this time unless copied), md docs/highres150_v8_report.md +
newmesh150_v8_report.md, docs summary, memory, Korean summary to the user.

2026-09-19 01:05 CDT *** V8 PUBLISHED *** (no-net gallery): main -> 2f348b78, new meshes -> 4b18fc06; md
docs/highres150_v8_report.md + newmesh150_v8_report.md; local pages output/report_{h150v8,n150v8}_page. Ledger: 39
end fragments over 19 targets (12 zero; dragon/armadilo/cow/beast 1; C 3; bob 9 + ogre 21 = one collapsed clump
each); grid probe >= 1 cell 0 frames everywhere; detached drawn pieces 0 frames everywhere. Quality vs netted v7:
within 0.7 pt on 15, bob -2.0 / beast -2.6 / armadilo -1.1 / C -3.5. Pitfall fixed: render_pbr/render_iso_video
dict conversion lost .files (9b21e21). Nothing running on hyde06. OPEN for the next round: C's arm fronts and the
thin-feature quality without the net (bob, beast); the surface-aware objective; v10 cleanup rule
[[cleanup-after-v10]]. Server output holds v7 + v8 runs and reports (v7 report folders can go at the next sweep).

**Ops facts:** logs are block-buffered until exit (judge progress by markers); never put a run name
unbracketed in `pgrep -f` inside the same ssh command; kill in a separate ssh call. Server output was
cleaned 2026-09-17 15:30 (kept h150y/n150y/h150q/n150q + report_h150p; other logs in
output/logs_archive_20260917.tgz); local output/ keeps only the three page folders.

2026-09-19 09:00 CDT *** SURFACE ROUND (user: "surface를 매끈하게", find 5-6 methods) *** Candidates S1-S6
pre-registered (experiments.md 2026-09-19); module physmorph/render/surface_recon.py; renderer flags --kernel pca,
--surface poisson|imls|surfel, --post bilateral, --pull, --layer grad|density, --poisson_cell, --bulk; probe
scripts/probes/surface_gt.py (recon of the TARGET cloud vs the true asset mesh: distance / bias / completeness /
normal dev / roughness at 2 spacings); runner scripts/ops/surface_test.sh <gpu> <run> <frame> [cands]; results in
$OUT/surf_test/{run1..run5, current}. FINDINGS: (a) the sampling floor = voxel fill 110^3 + jitter, 0.56-2.1
particles per fill voxel with replacement -> shot noise ~14 % at the 1.5-spacing blur -> 0.6-spacing bumps of
3-spacing width (the 14.5 deg floor); (b) BUG: rho_bulk = median of occupied voxels is ~0.5 of the particle
median (blur halo) -> the "0.283 bulk" level was ~0.14 of the interior density -> every gallery surface sat
+1.6-1.9 spacings outside the truth; fixed (2512410, --bulk particle; MC now +0.5-1.2, the level's own offset);
v8 videos carry the old bulk (fat by ~0.7-1 spacing) -> re-render pending; (c) WINNER S2: screened Poisson of the
plane-pulled outer layer (layer = |grad rho|/rho >= 0.285/spacing, PCA normals oriented by the gradient, octree
cell 1 spacing, trim 3 sigma): target distance 1.16-1.19 -> 0.20-0.29 spacings, bias 0, roughness 12.3-13.2 ->
8.4-9.7 deg (targets), 12.5-13.2 -> 8.5-12.9 (morph frames); costs: 10-13 raw comps/frame (flaps, dropped by the
mass rule), ~4x slower; (d) FALSIFIED: S1 PCA kernels (same bumps; sigma0 0.7 worse + cavities), S3 IMLS (spurious
crossings), S4 surfel triangulation (positions good, mesh not), S5 bilateral (-1.5 deg only), density-rule layer
(morph frames break), cell 2/3 or 1/2 (rougher, fragmented), "legs lost" (legs missing in MC too). RUNNING: three
full Poisson videos bunny/dragon/cow -> $OUT/surf_test/videos/<T>_poisson.mp4 (+ .components.txt), markers
"SURFVIDEO <T> DONE" (wait task b7ftr1pjc). NEXT: per-frame QA of the three videos; if clean -> photoreal_batch
with --surface poisson (+ bulk fix) for all 19 v8 targets -> rebuild both pages -> republish 2f348b78/4b18fc06
(read first) -> method.md 10.12 written, experiments.md summary, Korean summary with paper sources.

2026-09-19 09:50 CDT surface round, whole-trajectory test: pass 1 failed twice — (1) the 3-sigma Poisson trim opened
the surface -> flaps counted as drawn pieces (cow: drawn>1 in 143/292 frames vs MC 7) and an open body can vanish
(|svol| < dx^3) -> trim removed (--poisson_trim 0, closed surfaces, mass rule works); (2) Open3D 0.19 Poisson
SEGFAULTS nondeterministically across frames (exit 139 at frame 126 then 51) -> Poisson runs in a spawned child per
frame (physmorph/render/poisson_worker.py), retry single-threaded, else that frame = level set + sidecar record.
Pass 2 of bunny/dragon/cow videos running (markers "SURFVID2 <T> DONE", $OUT/surf_test/videos/, old tries in
videos/try2). Gallery re-render script ready: scripts/ops/surface_rerender.sh <gpu> <prefix> <targets...> (keeps
v8 files as <T>_v8mc_*; h150v8: A armadilo bob bunny C dragon heart spot teapot V; n150v8: beast bimba cheburashka
cow fandisk homer maxplanck nefertiti ogre). Machine load ~350 on 128 cores (other users) -> Poisson ~15-20 s/frame.

2026-09-19 11:20 CDT surface round: pass 2 videos finished (isolation OK) but QA failed (drawn>1 in 4/28/141 frames)
because the 10.10 rules read the LEVEL SET -> three fixes reading the drawn mesh: enclosure by occupancy (+1 spacing
tolerance), shortest-chain bridge (Dijkstra), mass = enclosed particle count per component (body = heaviest, light
component inside the body = cavity). Test frames all one drawn piece. Pass 3 of bunny/dragon/cow videos running
(markers "SURFVID4 <T> DONE", ~15 s/frame). After pass 3 QA (target: drawn>1 frames ~ v8 MC 0/0/7): launch
scripts/ops/surface_rerender.sh lanes for the 19 v8 targets -> rebuild pages -> republish.

2026-09-19 12:05 CDT pass 3 QA: drawn>1 10/21/79 frames (unbridged 6/15/16, all in the expansion phase = leader
clusters within one cell of the body) -> bridge link radius = max(2.5 spacings, dx) (c74543c); stills bridged.
GALLERY RE-RENDER LAUNCHED: 6 lanes of scripts/ops/surface_rerender.sh over the 19 v8 targets (GPU 0), logs
$OUT/surf_rerender_logs/lane1-6.log, markers "SURFRR <prefix> <T> DONE" (19 expected), ETA ~4-5 h. NEXT after
19 DONE: check every sidecar footer (drawn>1 frames, fallback frames), cp the v8 global.json drafts (already
installed in report_h150v8/report_n150v8), python scripts/ops/build_report150.py for both prefixes (titles
"PhysMorph 150k v9 — 매끈한 표면" / "PhysMorph new meshes 150k v9"), fetch_page.sh h150v8 / n150v8, READ then
republish 2f348b78 / 4b18fc06, md reports docs/highres150_v9_report.md + newmesh150_v9_report.md, Korean summary
with the paper sources (Triangle Splatting 2505.19175, TS+ 2509.25122, 2D TS 2506.18575, 3DGT ECCV26 2607.10690,
SplatSurf, SuGaR, 2DGS, Yu&Turk 2013, Kazhdan&Hoppe 2013, RIMLS 2009).

2026-09-19 14:00 CDT first re-render batch stopped at 8/19: spot showed "drawn>1 unbridged" 36 frames = closed
Poisson sheets INSIDE the body (gradient rule fires at interior density steps) with >= ppc mass -> rule: every
component the body encloses is interior (fde333f). Batch RELAUNCHED 13:58 (6 lanes, markers SURFRR; baseline count
before relaunch = 8, done when 27; poll task b73odbd7u; logs $OUT/surf_rerender_logs/lane1-6.log). v9 page drafts
written: scratchpad/global_v9_draft.json, global_n150v9_draft.json (install as report_h150v8/global.json and
report_n150v8/global.json, validate JSON with $PY on the server).

2026-09-19 15:30 CDT USER: "full render보다 N=40000부터 확인" -> 150k batch STOPPED (8 of 19 had finished with the
pre-interior-rule code; report folders hold mixed files, v8 kept as *_v8mc_*). 40k check on lr64_bunny/lr64_dragon/
or40_C: two fixes (Loop subdivision to the render voxel, same-side plane pulling; 43252a4). Stills good for
bunny/dragon; C 40k = thin-sheet limit (ragged rim). 40k Poisson vs MC videos running ($OUT/surf_test/videos40,
markers SURFVID40B). NEXT: compare per-frame QA (drawn>1, unbridged) -> report to user in Korean with the 40k
stills -> user decides on the 150k re-render (surface_rerender.sh lanes; ~5 h).

2026-09-19 16:50 CDT 40k videos: Poisson passes QA on bunny/dragon/C (drawn>1 0 frames each; MC 0/0/9 bridged).
Comparison page published: artifact 0c05b8c6 (read before republishing). USER QUESTION: "smoothing kills detail;
shouldn't the render gradient capture it?" -> answered: sub-spacing detail absent from the target cloud (fill
pitch), 1-3 sp band is 14% shot noise (sampling with replacement), loss cell ~1.5-2 sp; PROPOSED structural lever =
noise-free sampling (one particle per fill voxel / Poisson-disk) for source+target, then finer loss blur + Poisson
cell; test = resample targets without noise + acid test at cell 0.5 sp. Waiting for the user's decision; 150k
re-render stopped (8/19 done with pre-interior-rule code; needs a full rerun if resumed).

2026-09-19 17:30 CDT USER asked: give the render gradient the flattening role + survey how the triangle-mesh methods
get surfaces -> docs/surface_gradient.md written (facts: flagship render channel = soft silhouette render_res 64 +
D_pbr shading ON with w_pbr 1.0 (CLI default; config default 0 is overridden by the arm), d_pbr ~0.02 vs d_render
0.17; loss grid follows the MPM cell under --loss_units density: provenance loss_res 37, dx 0.308 wu = 3.6 sp at
150k -> shading normals from that grid cannot see 1-3 sp bumps; survey table SuGaR/2DGS/Gaussian Surfels/TS/TS+/
2DTS/3DGT with formulas; mechanisms G1 (D_pbr target from the denoised target surface + normals on the render-pixel
grid), G2 (normal consistency + plane pulling on the outer layer, 2DGS/3DGT analogue), G3 (depth distortion), G4
(Sobolev), G5 (noise-free sampling)). Recommendation: G5 sampling test first (no runs), then G1.

2026-09-19 18:30 CDT GRADIENT-STAGE ANALYSIS (user: analyse physics/render gradients per stage; protocol not "add
a loss", e.g. external force): --grad_dump + probes/grad_stage.py on 40k bunny 8 windows: render covector 99% on
the outer layer, 65% rough at 2 sp; after the adjoint every channel's control gradient has the grid correlation
(0.86@2sp, 0.6@4, 0.2@8); render channel alone ROUGHENS (0.474->0.50), physics smooths (0.458). External FORCE
on layer particles is averaged away by P2G/G2P (2.4% of a bump per window) -> protocol = per-step POSITION
projection of the rough plane residual, frac 1/T (--layer_relax; kernels k_layer_resid/k_layer_project; warp
gotcha: no division by loop accumulators inside kernels or the adjoint blows up). Tests pass (directional FD).
RUNNING: gs40lr_bunny (8 windows, --layer_relax --grad_dump) -> compare plane-residual RMS vs 0.45 baseline.
NEXT: full 40k bunny/dragon runs with --layer_relax -> Poisson+MC renders, roughness/QA/silIoU vs lr64 baselines
-> report; then decide the 150k re-render. docs/surface_gradient.md §6 has the numbers.

2026-09-19 18:55 CDT LAYER RELAXATION RESULT (40k full runs lrx_bunny/lrx_dragon = recipe + --layer_relax, frac
1/T): outer-layer plane-residual RMS over the morph bunny 0.442 -> 0.290 sp (end 0.457 -> 0.270), dragon 0.403
-> 0.279 (end 0.371 -> 0.261); target clouds' own floor 0.339 / 0.325 -> the morph is smoother than its target.
Cost: silIoU 0.962 -> 0.957, 0.964 -> 0.952; chamfer unchanged; detFmin better; frac=1 diverges (falsified).
RUNNING: stills (surf_test lrx_*) + Poisson/MC videos of lrx runs (markers SURFVID40X). NEXT: compare stills
(s40x vs s40/s40b), video QA, decide default (--layer_relax in RECIPE?), report to user in Korean; 150k later.

2026-09-19 19:30 CDT sampling test (bunny 40k): stratified (no replacement) MC 13.2->12.3 deg, Poisson 5.7->4.9,
layer RMS 0.352->0.286 -> replacement noise is ~1/10 of the level-set roughness; the rest = jittered arrangement
through the 1.5-sp blur (the level-set floor). Layer projection smooths the PARTICLE surface (RMS -35%) but the
MC render stays 13.8; Poisson 6.1->5.6 / 7.8->7.2. CONCLUSION: smooth+detailed at 40k = --layer_relax in the
physics + Poisson deliverable. docs/experiments.md entry "the gradient at each stage" written. Waiting on lrx
Poisson/MC video QA (SURFVID40X). Open decision for the user: adopt --layer_relax in the RECIPE (IoU -0.5..-1.2 pt
at 40k) and Poisson as the deliverable; then the 150k re-render (lanes ready) or new 150k runs with --layer_relax.

2026-09-19 19:50 CDT USER: "render gradient가 구조적으로 아래를 보게 하려면?" -> answered (actuator + reference, not
signal) and user said OK -> IMPLEMENTED G1 (--pbr_denoised: target shading from the target's Poisson-surface
normals, morph normals on the render-pixel grid blurred 1.5 sp; TargetPack pgmin/pdx/pdims/pblur) and A
(--layer_ctrl: per-window normal displacement leaf u on the outer layer, 1/T per step in k_layer_project, second
Adam leaf, clip one spacing, u through warp_mpm_ext(u_t=) and PersistentAdjoint.apply(dfc, u)); tests pass (5).
RUNNING: g1_bunny/g1_dragon (relax+G1) and g1a_bunny/g1a_dragon (relax+G1+A) at 40k (task prints end metrics +
layer RMS for lr64/lrx/g1/g1a). docs/surface_gradient.md §7 written. NEXT: stills + roughness for g1a, video
QA, report; then decide recipe (layer_relax, pbr_denoised, layer_ctrl) and the 150k plan.

2026-09-19 18:50 CDT (new session) G1/G1+A bunny 40k results: g1 (relax+G1) = lrx (silIoU 0.9568, RMS 0.288);
g1a (relax+G1+A) silIoU 0.9614 (recipe 0.9623), chamfer 0.1196 best, detFmin 0.758, layer RMS 0.363, end frame
vs true mesh d_abs 1.00 / n_dev 21.9 (lowest of the four) / Poisson rough 6.24 (vs 5.9) -> extra roughness is
STRUCTURE. lrx_bunny Poisson video QA: drawn>1 0 frames. Dragon analysis running server-side (marker DRAGON_DONE
in $OUT/surf_test/endframes/dragon.log). surface_gt.py has --gt_all now.

2026-09-19 19:05 CDT dragon 40k four arms: silIoU 0.9642 / 0.9519 (relax) / 0.9493 (G1) / 0.9594 (G1+A); chamfer
.1222/.1237/.1249/.1235; detFmin .663/.705/.743/.778; layer RMS .403/.279/.265/.341; Poisson mid-frame rough
7.8/7.2/7.0/8.5 (comps 1/7/7/10). Same pattern as bunny. End-frame-vs-true-mesh probe for dragon running
(poll task bap0xurpk; server log $OUT/surf_test/endframes/dragon.log marker DRAGON_DONE). lrx_dragon Poisson video
at 351/629. NEXT: fill dragon end-frame columns, commit, report to user; recommend recipe = --layer_relax
--pbr_denoised --layer_ctrl + Poisson deliverable; then 150k plan (new runs, not re-render).

2026-09-19 19:10 CDT dragon end frames vs true mesh: n_dev 25.1/24.6/24.8/24.9 (all equal), rough 7.9/7.8/7.5/8.6
-> A's extra roughness on the dragon NOT shown to be structure (bunny: slightly better n_dev). Proposed to user:
recipe += --layer_relax --pbr_denoised --layer_ctrl + Poisson deliverable (or relax only for max smoothness);
150k = NEW runs. Waiting for the user's decision. lrx_dragon Poisson video still rendering (QA later).

2026-09-19 19:25 CDT USER ACCEPTED: recipe = RECIPE_V8 + --layer_relax --pbr_denoised --layer_ctrl (hyde06_env.sh
updated, RECIPE_V8 kept), Poisson deliverable; 150k ONLY after the whole pipeline is verified at 40k; resolve
ambiguities. Ambiguity RESOLVED with surface_gt detail_analysis (hp_res = high-passed residual to the true
surface, dcorr = correlation with true detail): A's bumps = structure (lowest hp_res both targets), relaxation
removes noise, IoU cost is at the pixel. RUNNING: p40_bunny / p40_dragon (new recipe, 40k, markers "P40RUN <T>
DONE"). NEXT: post_run.sh p40 bunny/dragon + photoreal_batch.sh 0 p40 bunny dragon (SURFACE=poisson default) +
build_report150.py $OUT/report_p40 -> fetch_page.sh p40 -> publish as the 40k verification page; check every
sidecar/QA column parses with the Poisson "# surface" line; lrx_dragon Poisson video QA still pending.

2026-09-19 19:40 CDT p40 (new recipe via hyde06_env) 40k: bunny silIoU 0.9628 chamfer 0.1196 detF 0.766 (9.6 min);
dragon 0.9555 / 0.1235 / 0.736 (11.4 min). Twin spread vs g1a = +-0.4 IoU pt. POST RUNNING in parallel: post_run.sh
p40 bunny/dragon + photoreal_batch.sh 0 p40 <T> (Poisson; markers POSTP40/PHOTOP40 <T> DONE; ~2 h for the videos),
then build_report150.py $OUT/report_p40 "PhysMorph 40k 파이프라인 검증 — 새 recipe" $OUT/p40_report.md (global.json
installed from scratchpad/global_p40_draft.json). Wait task bsxeotn83. NEXT: fetch_page.sh p40 -> read QA table ->
publish the verification page (new artifact) -> report to user; if every column clean -> plan 150k new runs.

2026-09-19 19:50 CDT lrx_dragon Poisson video QA: 629 frames, drawn>1 0, unbridged 0 (PASS). p40 post_run stage
verified (gates PASS, gridfrag 0, reatt 0, census clean); build_report150 dry-run on partial p40 folder OK. p40
Poisson videos rendering (photoreal_batch, ~2-2.7 h, progress lines filtered by the batch's grep); wait task
bsxeotn83 then builds the report. After: fetch_page.sh p40, publish new artifact "PhysMorph 40k 파이프라인 검증".

2026-09-19 21:15 CDT *** 40k PIPELINE VERIFIED end to end with the new recipe *** p40_bunny / p40_dragon: gates
PASS, gridfrag 0, reatt 0, Poisson videos drawn>1 0 frames (502 / 656 frames), fallback 0, report built
(report_p40, docs/pipeline40_report.md), page fetched (output/report_p40_page) and published as a NEW artifact
(title "PhysMorph 40k 파이프라인 검증 — 새 recipe"). Photoreal Poisson video time at 40k: bunny 45 min, dragon 100 min
(CPU-bound). Waiting for the user's go for the 150k gallery = NEW runs with $RECIPE (19 targets, lanes) + Poisson
post; the v8 pages stay published until then. Local output/ has report_p40_page (8 MB).

2026-09-21 09:40 CDT USER: "render gradient는 지금 어때?" -> gs40new_bunny (new recipe, 8 win, --grad_dump with u):
covector unchanged (97% layer, ~50% rough; shading rough 0.67->0.55 with G1); stress path unchanged (corr2 0.89);
u path keeps half the 2-sp content (corr2 0.46, corr4 0.21), render owns u (10x physics); accepted u RMS 0.585 sp,
19% at the 1-sp clip, corr2 0.31 -> in the expansion phase u does pixel-scale TRANSPORT at capacity and raises the
layer RMS 0.43->0.47 that the relaxation undoes. Proposed next test (NOT run): project the u-gradient onto the
layer's smooth subspace (same W, h=2 sp). Docs §7 updated. Still waiting for the 150k go.

2026-09-21 10:10 CDT USER DECISION: 150k is EXCLUDED from the experiments for now (no 150k runs, no re-render; the v8
pages stay as they are). All work at 40k. User asked for the gradient results VISUALISED: probes/grad_stage_viz.py
(7 figures from the three dumps old/relax/new: covector maps, smooth/rough split, stress vs u paths, corr vs scale,
single-channel responses, per-window traces, the accepted u) -> figures in $OUT/gradviz, fetched to scratchpad/
gradviz; page to publish next (embed JPEGs, Korean captions). fig6 title fixed (rerun needed).

2026-09-21 10:10 CDT gradient figures published (artifact "Render gradient 단계 지도"; source page scratchpad/gradviz/
index.html, figures $OUT/gradviz + scratchpad/gradviz). Findings shown: silhouette covector 98% on the layer, 83%
normal, 40% rough at 2 sp (G1 cut it from 65%); stress path corr2 0.87 for every recipe (grid property); u path
corr2 0.46-0.50; u alone moves the layer 0.19-0.31 sp and the interior 0.001-0.005 sp; u at the clip 61% in
window 0 -> 0% from window 4 (transport in the expansion phase, correction later). STATE: 150k excluded; the next
proposed test (u-gradient projected onto the layer's smooth subspace, same W) not yet run; awaiting the user.

2026-09-21 10:30 CDT USER: "covector half noise — was it always so?" (yes: old 0.65/0.67, relax 0.52/0.56, new 0.53/
0.55; sources = target shot noise + CIC rasteriser per-particle kernel derivative) + "run the next candidate" ->
--layer_ctrl_smooth implemented (u Adam step <- W step, W = relaxation weights h=2 sp; 878afd0). RUNNING:
ps40_bunny/ps40_dragon (RECIPE + --layer_ctrl_smooth), gs40smooth_bunny (8-win dump); server chain
$OUT/ps40_analysis.sh -> $OUT/ps40_analysis.log (marker PS40_ANALYSIS_DONE; poll task bexa2y5jh): stage probe, end
metrics, layer RMS, mid stills, end frames vs truth (lr64/lrx/g1a/p40/ps40). NEXT: read, docs §7, report; decide
whether --layer_ctrl_smooth joins RECIPE.

2026-09-21 10:40 CDT projection test (--layer_ctrl_smooth) RESULT: on the channel as designed (u RMS 0.585->0.397,
clip 19->8.5%, corr2 0.31->0.71, u no longer roughens), morph-mean layer RMS -7..-9%, dragon mid Poisson 8.5->7.8
deg; BUT bunny end-frame detail worse (hp_res 0.177->0.199, dcorr .28->.24), IoU -0.4/-0.2 (spread), early morph
slower -> FALSIFIED as a recipe addition (redundant with the relaxation's W on the state). RECIPE unchanged
(v8 + --layer_relax --pbr_denoised --layer_ctrl). 150k excluded. Open next steps at 40k: (a) G5 noise-free
sampling of source+target (the only lever below the pixel), (b) thin-feature targets (bob, beast, C) through the
new recipe, (c) the CIC rasteriser's per-particle kernel derivative as a noise source (a smoother splat kernel?).

2026-09-21 11:05 CDT USER: "check all three" -> RUNNING at 40k (GPU 0): thin targets t40_{bob,beast,C} (RECIPE) +
t40v8_* (RECIPE_V8); G5 g5_{bunny,dragon} (RECIPE --sampler stratified; NOTE flag is --sampler, --sample is
volume/shell); quad kernel k40_* launched by the chain after the t40 batch (RECIPE --sil_kernel quad); dumps
gs40_bunny_{g5,quad}_dump. Server chain $OUT/all3_analysis.sh -> $OUT/all3_analysis.log with markers STAGE_A
(dumps) / B (k40 launch) / C, C2 (thin: metrics, RMS, post_run, QA, Poisson videos) / D (g5) / E (quad) / ALL3_DONE.
Pre-registered: G5 adopt iff covector rough share down AND end-frame hp_res down / dcorr up beyond the twin spread
(0.01 / 0.02); quad adopt iff gx_sil rough down, IoU/chamfer within spread, hp_res/dcorr not worse; thin targets:
QA columns 0, silIoU within spread of the v8 twin. Poll task running locally. Code: 0644604, bd8a0fe.

2026-09-21 11:15 CDT stage A of the three-way check: G5 stratified -> covector rough share UNCHANGED (sil 0.52,
pbr 0.54): target shot noise is NOT the source. QUAD splat -> sil covector rough 0.53->0.33, norm -1/3, shading
0.55->0.42, u-gradient rough 0.35->0.22 / corr2 0.46->0.61, 8-win IoU 0.9105->0.9181: the rasteriser WAS the
recipe-independent half. Chain relaunched after stale markers (first crashed launch wrote DONE markers; removed
with sed, backup $STATUS.bak_20260921). Stage D (g5 full runs) split into $OUT/d_only.sh -> d_only.log so it
does not wait behind the thin-feature videos. GOTCHA: pkill -f inside ssh kills the ssh shell if the pattern's
text appears elsewhere in the same command line — use a variable for the name.

2026-09-21 11:25 CDT G5 ADOPTED (full runs vs p40 twins): end frame vs true mesh hp_res 0.177->0.159 / 0.200->0.185,
dcorr +0.28->+0.36 / +0.32->+0.34, n_dev 21.8->17.6 / 25.1->23.6, IoU +0.4/+0.6, chamfer -3/-4%, layer RMS morph
-25%/-13%, Poisson mid 6.2->5.4 / 8.5->8.1. Caveats recorded: bunny target itself better (open base, 44^3 fill no
spurious floor); covector rough share unchanged (G5 helps the target/source, quad helps the gradient). Launched
gq_{bunny,dragon} = RECIPE + --sampler stratified + --sil_kernel quad (the combination) for the final decision.
surface_gt fix 174b824 (source sampler -> v_src). Pending: quad full runs (k40, after beast), thin-feature stage C.

2026-09-21 12:05 CDT THREE-WAY VERDICTS: G5 stratified ADOPTED (RECIPE += --sampler stratified; hyde06_env.sh);
quad splat FALSIFIED (covector smoother but bunny end-frame hp_res 0.177->0.212, layer RMS up; with G5 worse, early
stop 882 frames, d_95 4.2); combination not adopted; thin targets under the recipe: bob +0.45, C +7.6 IoU, beast
within spread, all QA columns 0 and census cleaner. Pending: stage C2 (thin-feature Poisson videos QA) in
$OUT/all3_analysis.log (the main chain also re-runs D/E — harmless). RECIPE now = v8 + --layer_relax --pbr_denoised
--layer_ctrl --sampler stratified. 150k still excluded. Next candidates: none pre-registered; report to user.

2026-09-21 12:25 CDT USER: "prove why there is no noise + visualise" -> scripts/probes/sampling_noise.py (theory:
replacement = Poisson process, Var = rho int K^2, rel (p/sigma)^1.5/sqrt(8 pi^1.5) = 5.9% at sigma = 1.86 p_vol;
stratified = dipole field of the jitter, rel (p/sigma)^2.5/sqrt(64 pi^1.5) = 1.1%, spectrum x k^2 p^2/12).
UNIT GOTCHA: project "spacing" (8-NN median) = 1.24 x volumetric spacing for a Poisson process. Measured: cube
5.5->1.1% (theory 5.9/1.1), bunny 8.1->2.9, dragon 5.2->2.2; dispersion index 1.01->0.26 / 1.30->0.31 /
1.06->0.28. Figures $OUT/noiseviz + scratchpad/noiseviz; page template written (publish next). docs §9 written.

2026-09-21 13:05 CDT noise-proof page PUBLISHED: "층화 샘플링의 잡음 증명" https://claude.ai/code/artifact/b4bb5282-65cb-45ce-97be-c8186554f01b
(source scratchpad/noiseviz/template.html -> index.html, figures from $OUT/noiseviz; docs/surface_gradient.md §9;
code d415bf8). Stage C2 (thin-feature Poisson video QA, report_t40): bob 441 frames and C 101 frames -> drawn
components>1 in 0 frames, Poisson fallback 0; beast (895 frames) still rendering at 13:05 — log the beast line to
docs/experiments.md when it lands. Not yet in docs/experiments.md: the bob/C Poisson QA lines. 150k still excluded.

2026-09-21 13:50 CDT stage C2 COMPLETE: beast 896 frames, drawn pieces>1 in 2 frames (219, 225), both bridged by a
particle chain (unbridged 0), Poisson fallback 0; bob 0/441, C 0/101. Logged in docs/experiments.md (2026-09-21
entry). Nothing pending from the three-way check; the main chain's D/E re-run is harmless. Nothing else running
for me on hyde06 except the chain's tail. 150k excluded. Next: none pre-registered; awaiting the user.

2026-09-21 14:10 CDT three-way chain ALL3_DONE (A/C/D/E, no tracebacks; the D/E re-run reproduces the recorded numbers).
NOTHING of mine running on hyde06. USER QUESTION answered: "does the current proof show rendering meaningfully
affects physics?" -> answer: the §9 noise proof is a sampler proof, unrelated; render->control->outcome proven
beyond spread under v3/v6/v7 (g_share ~35 %, 150k triples 4-10 spreads, 40k v3 +1.4-4.6 IoU) but NO lambda=0
twin under the CURRENT recipe; through the stress path the render gradient is grid low-passed (cell scale); the
sub-cell path is u (kinematic, bypasses the grid). Proposed (not launched): 40k 2x2 factorial render x u + a
control re-run on bunny/dragon/bob, judged against the pre-registered spread (IoU 0.4, hp_res 0.01, dcorr 0.02).

2026-09-21 14:40 CDT USER: "prove everything unproven first, then organise the final implementation" -> the render x u
FACTORIAL at 40k, pre-registered in docs/surface_gradient.md §10 (commit 8811616): per target (bunny/dragon/bob) six
runs, same seed/particles: fx_11 (RECIPE), fx_11c (identical re-run = spread), fx_01 (--lambda_auto 0), fx_10 (RECIPE
minus --layer_ctrl), fx_00 (minus --layer_ctrl, lambda 0), fx_cut (--render_until 20/20/8). Chain
scripts/ops/factorial40.sh 0 2 RUNNING on hyde06 (bunny then bob on GPU 0, dragon on GPU 2; 3 at a time), log
$OUT/fx_analysis.log, markers "FX fx_<cell>_<T> DONE" in $STATUS, FX_RUNS_DONE / FX_DONE in the log. Readings:
fx_summary.py (g_share, lambda, cos, D_vol), layer_rms, QA, surface_gt --gt_all end frames (target ply bunny/dragon
= surf_test/g5_<T>/poisson_target.ply, bob made fresh), render_effect --ctrl fx_11c (figs $OUT/fx_figs). Verdict
rule in §10: "render meaningfully changes physics" needs H-A (outcome beyond spread) + H-B (D_vol lower, g_share)
on 2 of 3 targets; the DETAIL claim needs the u-dependent part of H-C beyond the spread. AFTER the verdicts: write
the final-implementation plan (user's request), publish a page, log in experiments.md. Sanity checked: fx_01 logs
render=OFF, stratified 44^3 identical. ETA ~2.5-3 h from 14:35.

2026-09-21 15:30 CDT FACTORIAL DONE (18 runs, 70 min; §10 readings + verdicts written; experiments.md entry; page next).
VERDICTS: H-A confirmed 3/3 on silIoU (+1.8-2.0 / +1.0-1.5 / +0.6-0.7 pts; spread 0.23/0.53/0.08), chamfer within
spread; cut at a third loses the gain (dragon/bob all, bunny half). H-B partly: g_share 0.39-0.40 (w1-20) / 0.35-0.38
deterministic; D_vol end lower only on bunny; det F less compressed on dragon/bob. H-C: outline gain needs no u
(fx_10-fx_00 = +1.6/+1.6/+0.9); physics path carries detail down to the cell on dragon (dcorr +0.28->+0.35 u off).
H-D falsified: cut twin divergence == identical re-run (0.40 vs 0.41, 0.76 vs 0.63, 0.43 vs 0.44 sp) -> chaos.
U CHANNEL: +0.4-1.4 IoU (render on), +0.7-1.0 (render off); MAIN ROUGHNESS SOURCE (layer RMS 0.19-0.22 -> 0.25-0.33;
bob end 0.44 with physics-only u); necessary on bunny (d_95 5.4 -> 1.4 without/with), neutral dragon, harmful bob
(n_dev 5.6->7.0, dcorr .32->.26). Mechanism: u projection never updates F -> no strain energy -> nothing resists.
PLAN docs/final_plan.md: P3 (u through F: F <- (I + grad delta) F on the layer, least-squares gradient over the
layer neighbourhood, recommended first), P1 (render-only u, ablation), P2 (cell-sum saturation gate, fallback), P5
drop u rejected; 9 runs ~40 min; then lambda=0 twins, freeze recipe, method.md §10.13, 19-target 40k gallery +
Poisson videos + page, cleanup sweep (cleanup-after-v10 milestone), 150k excluded. NOT LAUNCHED: awaiting the user.
2026-09-21 15:50 CDT factorial page PUBLISHED "Render × u 요인 실험" https://claude.ai/code/artifact/2bb0b1a9-f807-4612-b4fb-71f2a5ba554d
(source scratchpad/fx/template.html + jpg/; docs committed e894e5e). Waiting for the user's decision on final_plan.md (P3 first).

2026-09-21 16:40 CDT USER: "start with P3" -> IMPLEMENTED (commits c95b503, 1af4090; deployed): --layer_F = u channel
through F: kernels.k_layer_F (G = sum_a (delta_a - delta_p) x g_a + delta_p x n / depth; F <- (I+G) F on the layer;
k_update writes Fu[t+1], k_layer_F writes F[t+1]); surface_recon.layer_grad_weights (tangential least-squares
weights, normal ridge; depth = one spacing); traj layer tuple (mask,nrm,nbr,w,frac[,g,depth]); optimizer builds g
when cfg.layer_ctrl and cfg.layer_F; 7 layer tests pass on hyde06 (linear u field -> its gradient; dL/du through F
vs FD). §11 pre-registered (predictions P3-a..e, falsifiers). RUNNING: scripts/ops/candidate40.sh 0 P3 --layer_F
(fx_P3_{bunny,dragon,bob}, GPU 0, log $OUT/cand_P3.log, markers "FX fx_P3_<T> DONE", CAND_P3_DONE at the end;
readings read against fx_11/fx_11c/fx_10). GOTCHA fixed: run() needed `shift` (target name leaked into "$@");
stale markers removed (backup $STATUS.bak_p3). Next: verdict vs §11 -> adopt or P1/P2; then lambda=0 twins,
recipe freeze, method.md §10.13, gallery, cleanup. ssh to hyde06 drops intermittently today (retry works).

2026-09-21 17:30 CDT P3 FALSIFIED (§11): fx_P3 runs collapse (10/24/16 windows, silIoU 0.81/0.47/0.60, det F min
0.0005-0.008, layer RMS 0.40-0.73, strays 1.5-8 %): sub-cell strain written into F cannot be relaxed by the 3.6-sp
grid, the volumetric part compounds ((1-1/T)^T = 0.36 per window inward). Structural: MPM at this dx cannot carry
a sub-cell actuation through the physics (force averaged away / strain never relaxed). MEASURED: the bunny far
region without u is the crevice between the front legs (x~0, all inner side, 4.7 % of the surface), a concave
target feature the cell-scale control leaves 1.5 cells short — u = concave finishing; bob = noise on a saturated
ring. RUNNING: fx_P3t (tangential-only, --layer_F_depth 0, GPU 0) to close §11; P2 IMPLEMENTED (commit 39e5ed6,
deployed, 8 tests pass): --layer_gate = u only where |rho_morph - rho_target|/rho_ref at sigma 1.5 sp exceeds
2 x the Poisson floor (11.8 %); telemetry u_gate per window; §12 pre-registered (P2-a bob layer <= 0.22 / n_dev
<= 6.0; P2-b bunny d_95 <= 1.5; P2-c IoU within 0.5; P2-d dragon within spread). RUNNING fx_P2 on GPU 2
(cand_P2.log, CAND_P2_DONE). Then: verdicts -> adopt (freeze RECIPE += --layer_gate) or re-derive the floor from
the morph's measured dispersion; then lambda=0 twins, method.md §10.13, gallery, cleanup.

2026-09-21 18:50 CDT P3t (tangential-only) FALSIFIED (layer worse 0.32/0.31/0.40, det F 0.15/0.015/0.12, bunny hp_res
0.383). P2 (density-residual gate) NOT ADOPTED: bob harm halved only (layer 0.269->0.229, n_dev 6.95->6.45), bunny fold
lost (d_95 5.38 = u-off), IoU -0.4..-0.9; gate share falls 90 % -> 8-19 % as the outline arrives. PROBES: bunny's far
region is NOT a density defect (target uniform; residual at/behind the layer 0.03-0.06 < 0.118): it is the front-bottom
FOLD (u-off still shows a notch) + the non-watertight true mesh (base holes) -> bunny d_95 is not a clean metric.
P1 IMPLEMENTED (--layer_u_render_only: composite from separate gp/gr every iteration, g[u] = lambda gr[u]; commit
+ deployed) and RUNNING: candidate40.sh 0 P1 (fx_P1_*, cand_P1.log, CAND_P1_DONE). §13 pre-registered (bob layer <=
0.23 / end <= 0.15 / n_dev <= 6.5; bunny fold closed + IoU within 0.5; dragon within spread; u RMS < 0.585, clip < 19 %).
After P1: write the candidate-round verdict + page; decision on the recipe is the user's (options: fx_11 recipe as is,
u off, P1 if it passes). Remaining plan phases unchanged (lambda=0 twins, method.md, gallery, cleanup; 150k excluded).

2026-09-21 19:40 CDT P1 FALSIFIED: render-only u is ROUGHER (layer morph/end bunny 0.316/0.366, dragon 0.282/0.257,
bob 0.313/0.385 vs fx_11 0.27/0.32, 0.28/0.20, 0.27/0.17), chamfer +4-6 % (bunny 0.1214, dragon 0.1259), dragon
det F 0.615; bunny fold not closed (d_95 5.45). => the physics gradient through u was a regulariser, not noise.
SEAM DETECTOR probe (opposite-facing layer walls within 2 sp): ZERO seam particles on every run -> the bunny "fold"
is not a crack. Open question being probed (probe_deficit.py via a retrying Monitor; ssh to hyde06 dropping
repeatedly): is the bunny TARGET cloud short of the true mesh in the far region (fill deficit of the non-watertight
bunny)? If yes: bunny's true-mesh metrics are target-limited and the u-on/u-off difference there is about matching
a defective target. CANDIDATE ROUND SO FAR: P3 x (collapse), P3t x, P2 partial (bob halved, IoU -0.4..-0.9),
P1 x. Best composite remains fx_11 (RECIPE as is); u off = best true-mesh surfaces on bob/dragon (-1..-1.6 IoU).
Decision for the user after the probe: keep u / drop u / P2 compromise. Then lambda=0 twins, method.md, gallery.

2026-09-21 20:20 CDT CANDIDATE ROUND CLOSED (all falsified; docs committed 'candidate round closed'): P3 collapse,
P3t rougher, P2 half of bob's harm at -0.4..-0.9 IoU, P1 rougher + chamfer +4-6 %. FOUND: bunny TARGET has interior
low-density pockets (4.5 % of deep interior, column above the base of the non-watertight mesh; bob 0.00 %, dragon
0.45 %); the reconstruction draws interior sheets on interior density gradients -> bunny true-mesh metrics
contaminated ("u finishes bunny" = artefact). Non-watertight assets: bunny, dragon, beast, armadillo.
REVISED ORDER (final_plan.md §6): 1 fix the fill of non-watertight meshes (acceptance: deep-interior low-density
share < 0.5 % on every target, probe_column.py), 2 exterior test in the reconstruction (surfel with particles on
its outward side within 2 sp = interior; also in surface_gt), 3 re-read bunny fx_11/fx_10/fx_11c, 4 u decision
(user: keep u = IoU +0.4..1.4 / drop u = best true-mesh surfaces on bob & dragon), then twins/method/gallery/cleanup.
Page "마지막 구현 후보 라운드" being published. Flags kept as records: --layer_F, --layer_F_depth, --layer_gate,
--layer_u_render_only. hyde06: nothing running.
2026-09-21 20:30 CDT candidate-round page PUBLISHED https://claude.ai/code/artifact/bd571a40-98de-490e-bdd0-b7cf79fa7d74 (source scratchpad/fx/cand_template.html). Awaiting the user: proceed with final_plan §6 steps 1-2 (fill fix + exterior test)?

2026-09-22 14:30 CDT USER: "proceed (final_plan §6 1-2) and ALWAYS summarise the rendering influence" (memory
always-report-render-influence.md). DONE: (1) FILL FIX (commits 312cf16, 62fc403, bcd1fa3, deployed): the cause of
bunny's interior 40 %-density comb = the orthographic fill's 3-axis intersection fails in the columns above the
base HOLES (no bottom surface along y) -> _fill_ortho_reliable: boundary loops projected per axis = hole
footprints; a voxel is filled if enclosed along every reliable axis, but only where >= 2 axes are reliable (one
axis alone streaks: beast regression); + _fill_pockets (majority of 6 neighbours, iterated) + streak strip;
FILL_MODE "legacy" reproduces pre-2026-09-22 archives (surface_gt retries with it). fill_check.py acceptance
(deep-interior low-density share < 0.5 %): bunny 4.79->0.42, maxplanck 9.95->0.33, beast 0.00, armadillo 0.00,
watertight targets 0.00; dragon 0.54 (borderline, far from any loop; left). (2) EXTERIOR TEST (surface_recon.
exterior_surfels, wired in render_photoreal before oriented_layer): a surfel with >= 2 particles in its outward
2-sp cap beyond 0.75 sp is an interior density step -> dropped (bunny target: 2907 dropped). Tests pass.
RUNNING: $OUT/resheet_gt.sh (old fx bunny/bob/dragon end frames re-reconstructed with the exterior test ->
surface_gt; RESHEET_GT_DONE) and $OUT/fxbunny.sh (fx_11n/11cn/01n/10n bunny on the FIXED target, GPUs 0/2, then
readings; FXN_DONE). Then: §14 addendum + method.md §10.12 + sampling note, the u decision page WITH the
rendering-influence block (fx_11n vs fx_01n, g_share), report. GOTCHA: pkill patterns must not appear literally
anywhere in the same ssh command (killed the remote shell twice).
2026-09-22 15:40 CDT bunny on the FIXED target (fx_*n): 11n 0.9638 / 11cn 0.9637 (spread 0.01 pt!) / 01n (lambda 0)
0.9445 / 10n (u off) 0.9558; chamfer all 0.1153-0.1157; layer RMS morph/end 0.257/0.226, 0.258/0.207, 0.254/0.219,
u-off 0.191/0.221; g_share 0.39-0.40 (w1-20), 0.36-0.37; D_vol end 0.0036 vs 0.0041 (lambda 0). Surface columns
pending (fxbunny.log end-frame section; resheet_gt.log rerun after the legacy-fill fix f1d0c48).
2026-09-22 15:55 CDT RE-READ DONE: old end frames with the exterior test: bunny u-off d_95 5.39 -> 1.53, hp_res
0.295 -> 0.175 (the interior sheet WAS the "u finishes bunny" reading); u-on 0.94-1.10; bob/dragon move 0.3-0.5 deg,
verdict unchanged (u off better there). NEW bunny (fixed target): 11n/11cn IoU 0.9638/0.9637 (spread 0.01), 01n
0.9445, 10n 0.9558; surface 11n 0.88/15.7/0.154/+0.409, 11cn 0.83/15.4/0.152/+0.421, 01n 0.99/17.1/0.171/+0.329,
10n 0.89/16.3/0.164/+0.386 (d_95/n_dev/hp_res/dcorr). U DECISION EVIDENCE: u better on bunny (surface + IoU),
worse on bob/dragon surfaces, +IoU everywhere -> user's call. RENDER INFLUENCE (fixed bunny): g_share 0.39-0.40/
0.36-0.37, cos 0.01-0.02; vs lambda 0: IoU +1.9, n_dev -1.4 deg, hp_res 0.154 vs 0.171, dcorr +0.41 vs +0.33,
D_vol 0.0036 vs 0.0041. Next: page + report; then the user decides u; then twins/gallery/cleanup.
2026-09-22 16:20 CDT decision page PUBLISHED https://claude.ai/code/artifact/5246e365-a485-4e80-a490-772b6e99e43d (scratchpad/fx/decision_template.html). Docs committed 1b2adf7. hyde06 idle. AWAITING the user's u decision (A keep u / B drop u); then final_plan §4 (lambda=0 twins under the frozen recipe, method.md, 19-target 40k gallery + Poisson videos + page, cleanup sweep). Every report carries the rendering-influence block.

2026-09-22 17:30 CDT USER: "proceed until engineering/algorithm are all closed" (u KEPT per my recommendation; recipe
frozen in hyde06_env.sh). RUNNING on hyde06: $OUT/gallery40.sh -> g40 gallery (19 targets, batches A/B on GPU 0,
C/D on GPU 2 after the bob/dragon twins fx_11n/fx_01n), then photoreal_batch (Poisson videos, 2 per GPU) and
gallery_post.sh (surface_gt end frame + target, layer_rms, render telemetry -> report_g40/<T>/<T>_{surface_gt,layer,
render}.txt); markers G40 <T> DONE / POSTG40 / PHOTOG40 / GPOSTG40, G40_RUNS_DONE / G40_DONE in gallery40.log.
DONE meanwhile: build_report150.py table 2c (surface vs true mesh + rendering influence, page + md), method.md
§10.14 (relaxation, u, G1, stratified), tests/test_fill_reliable.py (2 pass), full pytest running (pytest_all.log).
NOT deployed: the footprint area rule (changes armadillo/ogre fills by 5-8 voxels) — reverted; keep as is.
NEXT after G40_DONE: write global.json for report_g40 (assessment, render_html = the rendering-influence block with
fx_11n vs fx_01n on bunny/bob/dragon, summary), build_report150.py on hyde06, fetch_page.sh g40, publish, docs
(experiments.md gallery entry, overview.md current state), then the CLEANUP SWEEP (archive logs first; list in
experiments.md), memory update. 150k excluded.
2026-09-22 18:10 CDT full pytest 231 passed / 2 skipped (smoke deselected). Twins on the fixed fill: bob 0.9774/0.9710 (+0.6), dragon 0.9635/0.9448 (+1.9), bunny +1.9; g_share 0.36-0.40. twins_post.sh running (surface columns, TWINS_POST_DONE). global_g40.json drafted in scratchpad (placeholders __BOB_*__/__DRA_*__ for the twins' surface columns). Gallery 9/19 runs at 18:05.
2026-09-22 18:45 CDT twins surface (fixed fill): bob 11n/01n n_dev 6.5/6.7, dcorr +0.23/+0.25 (silhouette only); dragon 22.1/25.0, dcorr +0.36/+0.29 (render improves the surface, like bunny). global_g40.json complete (scratchpad). cleanup_sweep.sh dry run: ~129 GB (38 live dirs + h150v7/n150v7/h150y/h150v7pw/lr96/lr128/or40/rv/rvb/matw*/nnw150 + report_*v7) — execute after the g40 page. Gallery 13/19 runs at 18:40.

2026-09-22 16:20 CDT (after the user's local restart; save conflict = VS Code's stale buffer, resolved by Revert):
pipeline map PUBLISHED https://claude.ai/code/artifact/f61f534f-baf2-4fc5-b5e8-2c2cf03f9232 (scratchpad/pipeline_page.html;
6 SVG diagrams: run flow, window loop, the two render->physics paths, the surface pipeline, server workflow, recipe
table, rules, what is established). Gallery state: 19/19 runs, post 19/19, videos 3/19 at 15:55 (load down to ~105
after the Poisson thread cap); monitor re-armed (G40_DONE). Then: build_report150 on hyde06 with report_g40/global.json
(already uploaded), fetch_page.sh g40, publish, experiments.md video QA + surface table, cleanup_sweep.sh (dry run
listed ~129 GB), memory. Pending user: nothing (u kept). 150k excluded.
2026-09-22 17:40 CDT WHERE page PUBLISHED https://claude.ai/code/artifact/9672af6f-806d-4748-a39c-6039a77b14b6 (grad_where.py on gd_bunny 56-window dump; scratchpad/gw/). Key: render dominates dFc at 1.6-5.2 % of particles = 11-23 % of the outer layer (ears, paws, base rim, back outline); 43-58 % of the render pull on the layer; per-particle share 0.25-0.29 layer / 0.09-0.13 interior vs global norm share 0.38-0.42. Docs committed 123d2cc. g40 videos 8/19, load 13.
2026-09-22 18:20 CDT USER: 'first make sure the pipeline works; the rest is comparison'. g40 video QA so far (9/19): all drawn pieces>1 = 0 except cow (24 frames, 20 bridged, 4 unbridged = 267/270/438/480). Diagnosed: not physics (grid fragments 0, single connected component at r = dx) but the bridge rule walking through FREE particles only; enclosed-to-enclosed contact within r had no edge. FIXED (render_photoreal filament_bridges: direct anchor-anchor edge; verified frame 267 -> bridged 1; committed + deployed). TODO after the batch: re-render cow's video (and any other target with unbridged frames) with the new rule, then build/publish g40. SIGGRAPH assessment given (not yet: effect size, baselines, 150k, u framing); user chose to verify the pipeline first.
2026-09-22 18:40 CDT mid-morph stills (homer 144, ogre 123, armadillo 102) clean: single closed bodies, no floating pieces. cow video re-render launched with the fixed bridge (g40_photo_cow_redo.log; overwrites report_g40/cow). Videos 10/19. After G40_DONE: check every sidecar's 'not bridged' count (expect 0), build, fetch, publish, docs, cleanup.

2026-09-22 20:00 CDT *** 40k ENGINEERING CLOSED *** g40 gallery PUBLISHED https://claude.ai/code/artifact/a19b0010-0b54-4e7f-949d-99bc18b0442a
(19 targets, unbridged pieces 0/19 after the bridge fix 4aa31cc, fallback 0; page tables 2/2b/2c; md twin
docs/gallery40_report.md). CLEANUP SWEEP DONE: hyde06 output 241 -> 123 GB (logs_archive_20260922.tgz; deleted
h150v7/n150v7 + report folders, h150y, h150v7pw, lr96/lr128, or40, rv/rvb, matw*/nnw150, 38 live dirs); local
v7 page folders removed. Docs: experiments.md entries, final_plan.md status (§4 done). Other pages today:
pipeline map f61f534f, where-dFc 9672af6f, decision page 5246e365. NEXT (user's call; SIGGRAPH assessment given):
comparison work — with/without-render videos, baselines (Xu et al. C++, geometric morph, 3DGS deformation),
150k heroes (u OFF for the paper headline, u on for the deliverable), arbitrary source->target pairs, image
metrics on held-out views. 150k still excluded until re-opened. hyde06: nothing running.
2026-09-22 20:40 CDT USER raised mid-morph surface quality (flicker, false breaks at thin necks). Diagnosis: per-frame independent Poisson re-fit (jitter) + necks 1-2 particles thick at 40k (physics levers exhausted, thin-feature dossier); voxel size is not the lever (fat level set). IMPLEMENTED render_photoreal --track (commit 9f30036; method.md 10.15): vertices advected with k=8 nearest particles (Gaussian 1 sp), pulled to the fresh reconstruction at alpha 0.3, re-mesh on topology change or drift > 1 sp; sidecar gains refit_jitter_sp / track_drift_sp / remeshed columns. TESTING: output/track_test/{cow_track,cow_fresh,ogre_track}.mp4 via scripts/render_photoreal_track.py (test copy on the server; the repo file is committed). Next: read jitter numbers, view videos, decide default (--track in photoreal_batch), re-render g40 videos if adopted.
2026-09-22 21:20 CDT tracking v1 result: the per-frame Poisson re-fit jitter is only 0.024 sp (image flicker 0.08 %): the visible defect is TOPOLOGY EVENTS (pieces/bridges toggling for a few frames at thin necks), not jitter. v2 policy (commit above): re-mesh only on particle-confirmed topology change / median drift > 1 sp / every 60 frames; gated pull (<= 1 sp). Running: track_test/cow_track2, ogre_track2 (monitor). Next: compare stills at cow video frames 71/89/146 (archived 213/267/438) fresh vs tracked; if the tube persists, adopt --track in photoreal_batch and re-render the 19 videos. GOTCHA again: pkill pattern must not appear literally in the same ssh command.
2026-09-22 22:10 CDT TRACKING v2 VERIFIED on cow: re-meshes 10 (periodic + early cavities), none at the neck episodes; legs continuous at 213/267/438; jitter 0.024 sp; ADOPTED as the video default (photoreal_batch TRACK=1, commit + deploy). ogre_track2 still rendering. Page 'Morph 중의 표면 추적' being built (scratchpad/gw/track_template.html). Next: re-render the 19 g40 videos with --track (photoreal_batch 4 concurrent, ~3 h), rebuild/republish the gallery page.
2026-09-22 22:30 CDT tracking page PUBLISHED https://claude.ai/code/artifact/6d800e4a-284e-440c-acaa-c2de4b2363be. photoreal_batch.sh fixed (TRACK default; commit 4d9ca0a, deployed). RUNNING: $OUT/g40_track.sh = tracked re-render of all 19 g40 videos (4 batches, PHOTOG40 markers cleared first, backup rcp_ladder_status.log.bak_track; G40_TRACK_DONE in g40_track.log; ~3-4 h). ogre_track2 test still rendering. After done: rebuild report (build_report150 on hyde06 with report_g40/global.json), fetch_page.sh g40, republish to the SAME artifact (file path output/report_g40_page/index.html, url a19b0010...), experiments/memory. The monitor for the re-render is armed for 1 h at a time (re-arm).
