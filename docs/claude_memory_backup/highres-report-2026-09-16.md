---
name: highres-report-2026-09-16
description: "In-progress 2026-09-16 high-resolution report job — 10 sphere→target examples at 40k --ppc 8 (render + physics-only twins) on hyde06, post-processing watcher, report builder, artifact; what remains"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-16T12:35:52.552Z
---

User request (2026-09-16): judge what remains (mass ejection, high-res + diverse examples, speed),
then a document with ~10 high-res examples, PBR stills, gradient reach (% of surface, magnitudes,
heatmaps), loss curves + wall-clock, physics-only comparison, hi-res GIFs; later: each result
viewable in the 3D viewer (every run writes `--live_dir` packets → `scripts/viewer_serve.py`).

State of the job: COMPLETE (batch h 07:20–09:08 CDT; all 20 runs post-processed; artifact published; docs committed).
- Batch h on hyde06 (`/tmp/run_hr_20260916.sh`, also `scripts/ops/`): GPU 0 = bunny armadilo dragon
  spot bob; GPU 2 = teapot heart A C V; each target = `hr_<T>` (render) then `hr_<T>_phys`
  (`--lambda_auto 0`); recipe `--n 40000 --ppc 8 --loss_units density --warm_start --w_kin 5
  --w_kin_var 200 --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000
  --live_dir $OUT/live`. Started 07:20 CDT; ~40 min per run when two runs share the box → ~7 h.
  Markers in `$OUT/rcp_ladder_status.log`: "HR <T> render DONE", "HR <T> phys DONE", "BATCH h1/h2 DONE".
- Watcher `/tmp/hr_watch.sh` runs `/tmp/hr_post.sh <T> <gpu>` per finished pair → `$OUT/report/<T>/`
  (hi-res GIFs res 240 stride 6 — the artifact budget is 64 MB per version and 15 MB per file, res-320/stride-2 GIFs were 10 MB each; PBR stills via `scripts/render_pbr.py` az 35/215 for render/phys/target,
  `scripts/probes/grad_field.py` (3 commits render, final phys), scatter_probe2, loss_curves);
  marker "POST <T> DONE"; logs `$OUT/report_post_<T>.log`.
- Local report builder: scratchpad `build_report.py` reads `output/report/<T>/…` + `global.json` →
  `output/report/index.html` (PBR PNG → JPEG 800 px, heatmaps → 1000 px, done in build_report.prep) → published as artifact https://claude.ai/code/artifact/3a1a66fe-6100-4ccf-9e54-21c02a1d0c8a (title "PhysMorph 고해상도 보고서"; republish the same file path with the new per-target files map).
- Done this session: sampler streak fix (0ccc43a), speed −27–31 % (5a3b8ee, device-side
  allocation), probes grad_field/loss_curves (5a3b8ee), render_pbr (f0ef818), asset survey (car
  fails to fill; 14 others fill; S takes 45 s).
- Speed-profile facts (40k ppc8, 12 commits): allocation 40 %, MPM fwd+adjoint 31 %, `.item()`
  syncs 13 %, PBR-lite views 8 %, silhouette 5 %.
- Gradient facts (rcpr_40k fixed target): render gradient on x touches 5–7 % of particles (17–20 %
  of the surface set, <2 % interior; surface/interior magnitude ratio 40–80×); D_vol touches all;
  through the adjoint the image gradient reaches 37–53 % of particles (29–49 % of the interior),
  reach by depth 0.56 → ~0.25 mid → 0.45 core.

**How to apply:** if resumed after a break, first check `rcp_ladder_status.log` and
`$OUT/report/*/`, fetch finished folders with scp into `output/report/<T>/`, run build_report.py,
publish; then write docs/experiments.md "2026-09-16 high-res gallery" + memory update.

Findings (2026-09-16): 6/10 converge (teapot, heart, spot, bunny, A, armadillo: chamfer 0.111–0.116, silIoU
0.94–0.98); render vs phys there: silIoU +0.4–3.6 pt, chamfer ±1 %, thin mass +0.001–0.019. bob (ring)
and V (sharp re-entrant): render WORSE than phys and ejects 3–5× more (580 vs 127; 123 vs 42). dragon:
both silIoU 0.8, strays 0.4–0.7 %. C (ring): both arms stall at commit 9 (outer-merit guard rejects all
candidates) — topology-change limit. Ejection is target-driven (thin appendages), not a render-channel
artefact except on hole targets; ejected particles never return. Gradient: image loss touches 6–9 % of
particles directly, 50–90 % through the MPM adjoint. Speed: 10 → 6.3–6.7 s/commit after 5a3b8ee; phys
3.5–3.8 s/commit (render channel ≈ 2.8 s/commit). Next ladders: v_max (needs CLI), near-band re-coupling on
ejected set, λ ramp, w_spray 0 on bob/V, outer-merit relaxation on C.
