---
name: competitor-landscape
description: "2026-09-23 night literature scan — the closest competitors (TVCG 2025 MPM morphing by Xu/Song/Levin/Hyde = the ancestry of our physics arm; PhysMorph-GS arXiv 2511.16988 = F-control MLS-MPM + 3DGS silhouette/depth losses, the nearest to 'rendering controls physics'), the runnable baselines (ICLR 2025 implicit velocity-field morph, 4Deform, volume-preserving SDF morph), what is covered vs plausibly ours, where we lose"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-15T23:13:37.108Z
---

The user (2026-09-23, discouraged after the 300k defects) asked whether the research is meaningful
against FEM-based morphing and the recent ICLR papers. The scan is recorded in
docs/related_work.md ("Competitors and baselines, 2024–2026 venues").

**THE USER'S OWN PRIOR WORK (the user, 2026-09-23: "불편한 사실 두 개 다 내가 한 거야"; 2026-10-06: the user is the CORRESPONDING AUTHOR of the TVCG 2025 F-control paper: "F Control 한 것도 나야 내가 그 논문 교신저자야" — never write "cite Xu et al." as if it were someone else's; say "our prior work / our F-control formulation"):** *A
Differentiable MPM Framework for Shape Morphing* (Xu, Song, Levin, Hyde; TVCG 2025 / SCA 2024;
per-particle F control, topology change, a log mass loss, a geometric target) and *PhysMorph-GS*
(Song, Hyde; arXiv 2511.16988; MLS-MPM 32³ + 3DGS silhouette/depth/edge losses, F control, 21–30
min per episode) are the user's papers; this project is their follow-up that closes what
PhysMorph-GS lists as open (thin features, watertightness) and characterises the render channel.
Never frame them as competitors; frame the contribution as the improvement over them.

**Correction (user, 2026-10-06 13:40 CDT: "우리가 PhysMorph-GS야"):** THIS project IS PhysMorph-GS — arXiv 2511.16988
is its own earlier preprint and the SIGGRAPH paper is PhysMorph-GS itself. Never propose PhysMorph-GS (or its legacy
configs) as a baseline. The external baselines are Xu et al. TVCG 2025 (physics only, `--baseline xu_spray`, D112), a
render-only method (proposed: Large Steps, Nicolet 2021, from a sphere), and the geometric neural morphs (ISD, 4Deform).

**Runnable baselines (public code, minutes per pair):** ICLR 2025 "Implicit Neural Surface
Deformation with Explicit Velocity Fields" (the Cremers group; topology change; ~20 min), 4Deform
(CVPR 2025; 8–10 min), Volume Preserving Neural Shape Morphing (SGP 2025; equal volumes needed).
The image-through-MPM works (PAC-NeRF, GIC, NeuMA, OmniPhysGS, PhysGaussian) fit materials or run
forward — not morph baselines.

**Plausibly ours:** the trajectory as optimal control with Sinkhorn transport + the volumetric
cell sum; exact mass conservation across the morph; shading supervision; the λ=0 twin; the
spacing-derived constant contract; the measured limits of the render channel. **We lose on:**
surface quality (the SDF morphs are watertight, no sub-cell roughness), speed, memory.

**Why:** the paper needs a head-to-head that has never been run ("paper comparisons not started").
**How to apply:** before any new mechanism, run the baselines on sphere→bunny and sphere→C and put
the axes side by side (topology change, mass/volume, self-intersection, intermediate-frame
quality, surface, time); position the claim on the axes where the numbers hold. Related:
[[mid-morph-lumps-u-channel]], [[always-report-render-influence]].

**Baseline comparison IN PROGRESS (2026-09-23 23:05):** hyde06 conda env `nsd` (JAX 0.4.25, flax
0.8.1, cuDNN pinned 8.9.7.29 — the pip plugin's cuDNN 9 fails); repos + data under
`/data/relcfd/chayo/physmorph_v2/baselines/` (ISD @ e7992a5, 4Deform @ 7f2274e; data
`<pair>_none` = sphere→bunny / sphere→torus, 20k samples, no correspondences); scripts in
`baselines_prep/` (`run_server.sh env|inputs|prep|train_isd|eval_isd|train_4d|eval_4d`,
`compare_baselines.py`); the chain log `baselines_prep/train.log` (marker BASELINE_TRAIN_DONE).
Our meshes at t = 0.25/0.5/0.75/1 in `$OUT/r300/mesh/` (Poisson PLY via `--save_mesh`), the target's
surface sample `tgt_surface.npy` (outer layer of the volume sample). Our surface Chamfer / diag at
t = 1: c300 0.0123, g300 0.0109, h300 0.0111; genus 0, watertight, volume drift ≤ 2.3 %. ISD
training ≈ 13 min per pair on the Ada (20 it/s late), then eval (MC 256).

**B12 render-only baselines read (2026-10-15).** Commits a29e096 and aea2fe4 on wip/b12, not merged yet. Table in `output/results_2026-10-15/b12_render_only/b12_summary.md`.
- **Large Steps end silhouette:** reaches ours on the genus-0 targets (bunny, homer; armadilo 0.0003–0.005 below) in 18–53 s, 13–61× faster than ours. NEVER claim a better end silhouette than inverse rendering on genus-0 targets.
- **Large Steps failures:**
  - genus 0 in all 756 frames, so the dragon's loops and bob's ring end as self-intersecting surfaces;
  - 31 of 36 ends self-intersect;
  - in-betweens are not a motion: dragon volume 1.60–1.74, and 75 % of the progress happens in the first 1–6 % of steps.
- **FlexiCubes:** spurious genus 4–58 and end IoU 0.77–0.98.
- **Claim the numbers support:** comparable end silhouettes; the gain is the trajectory plus reaching genus > 0 targets without self-intersection.

**Xu et al. reference version (user, 2026-10-06 16:50 CDT: "TVCG 2025쪽으로 보는 게 더 좋을거야"):** compare against
the TVCG 2025 paper (DOI 10.1109/TVCG.2025.3591729), not arXiv v1 / SCA 2024. Author's accepted version locally at
`C:\Users\ok429\Downloads\A_Differentiable_Material_Point_Method_Framework_for_Shape_Morphing.pdf`, on hyde06 at
`/data/relcfd/chayo/physmorph_v2/tmp/xu_tvcg2025.{pdf,txt}`. Its loss: 1/2 sum (ln(m+1+1e-4) - ln(m*+1+1e-4))^2 +
sum 1{m<m_min} 10 (m_min - m)^2, plain gradient (no x5 out-of-target factor: that and w=1 are in PhysMorph-GS's C++
working copy bffb9e2:legacy/DiffMPMLib3D, which is NOT Xu's code as published). Adam + bisection line search, dt 1/120,
control every 10 steps, 4 iterations x 3 passes, 32^3 grid. Baseline protocol = the paper's (user decision D112/D118).
