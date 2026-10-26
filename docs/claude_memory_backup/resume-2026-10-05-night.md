---
name: resume-2026-10-05-night
description: "Handoff 2026-10-06 13:20 CDT — D105 relief reference passed (300k + gallery); D107 gradient bands; D108 forensics; D109 refuted+reverted; D111 render-written display layer; D112 Xu baseline = our pipeline with --baseline xu / xu_spray (spray cleanup only); 300k baseline runs going (d112c.sh)"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-06T18:02:20.293Z
---

State at 2026-10-06 13:20 CDT, branch settled-base, HEAD a26af41, all pushed; branch `xu-baseline` (303603a, release-based C++-loss reproduction, superseded). Authoritative record: docs/experiments.md D104–D112.

- **D105** (`--layer_relief`, reference = relaxation operator on the layer's feet): passed 300k + 40k gallery (D106). Not yet made default (user decision pending).
- **D107**: relief born on arrival, plateau after; the window's free move (half-assimilated elastic spring, assim 0.5) takes back 1.3–2.6 %/window of fine-band detail. Proposed D110 (assimilation 1 with warm decay 1−assim) NOT run yet.
- **D108**: mouth lumps = mesh teeth at 300k (not an artifact); early Gaussians = particles torn out by the concentrated render gradient through per-particle dFc (windows 3–6).
- **D109** (dFc on grid nodes): refuted, reverted.
- **D111** display layer (user approved "4번"): massless discs on the exterior lattice at the 1.5M pitch with height h fitted by the render (mean of four 1.5M draws, h = S v with the one-pixel tent, same-side neighbours, base display's averaged normals, per-disc Adam state carried, step ∝ 1/sqrt(fits)). Stage 0 passed (bunny 5.4-pitch relief 0.55→0.75 = the 1.5M sample's 0.77). Stage 1 (`child_detail_morph.py`, post-hoc over a run's frames) 1e: flicker 1.35× D105 display. Fair comparison (same post-hoc fit on both): render-in-loop −35 % sil / −8 % shading vs physics-only + fit (bunny). Dragon PVfit result pending in output/gpu/d111/morph1e_dragon_PVfit.
- **D112 baseline** (user: Xu et al. = their TVCG 2025 paper; "physics만 돌리고 나머지 아무것도"): `--baseline xu` in OUR pipeline = same simulator/windows/Adam with Xu's EndLayerMassLoss alone (C++ form: sim grid, cubic B-spline, cell mass 1, ×5 out-of-target gradient); nothing of ours. 40k bunny: silIoU 0.906, detached particles 1.92 % (ours full 0.02 %). 300k bunny/dragon running (`tmp/d112b.sh`, output/gpu/d112/*300k_XU*, repo_r91), then runeval3. Next: D111 post-hoc fit on the XU frames, then the comparison table Xu+fit vs physics-only+fit vs full+fit; optionally the ablation ladder (user asked about SIGGRAPH scope: recommended 3–4 core contributions, rest implementation details, ladder decides).
- **xu_spray** (user: "입자 튀는 안정화 장치"만 켜기): Xu loss in its own units + our spray cleanup / ot_scale; 40k bunny detached 2.04 % → 0.68 %, silIoU 0.904 → 0.940, ears still merged/frayed. 300k launched 13:18 via tmp/d112c.sh (repo_r92): bunny/dragon XS on GPU0/1, X bunny→dragon on GPU3, outputs output/gpu/d112/*300k_XS / *300k_X, then runeval3. Next: D111 post-hoc fit on XS frames; table Xu+fit vs physics-only+fit vs full+fit.
- Physics-only twin ≠ Xu: it carries the whole target in 3D (OT to the 300k sample, surface proximity, relief) — why it is close to the render arm.
- Server: repo_r89 = D105 code, repo_r91 = HEAD (baseline mode); output 30 GB after cleanup (logged). hyde01 key may need re-adding by the user if `ssh hyde06j` says publickey.

Related: [[render-must-beat-physics-300k]], [[competitor-landscape]], [[no-parameter-only-fixes]], [[hyde06-ops-etiquette]]
