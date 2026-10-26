---
name: resume-2026-10-06-late-night
description: "Handoff 2026-10-06 22:15 CDT (user restarting the local PC): D119 nearly refuted, D118 Xu verdict, D120 proposed, figure files, what runs on hyde06"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-07T03:18:04.749Z
---

State at 22:15 CDT 2026-10-06. Local branch settled-base @ 241096f (D119 code, `--relax_in_transit` A/B flag); D118 on branch xu-paper-protocol @ b502b5a (NOT merged).

**D119 (relaxation stops on arrival)**
- DONE 22:25: refuted on (c), recorded and the code reverted in c4a51ce (pushed). Lesson: the relaxation on the arrived layer is what holds it on the mesh relief; the flicker is the price of that hold.
- Good: tail flicker −58 % (ALT 0.000256 vs D105 0.000615 on the same frames).
- Bad:
  - runs 1.3–1.9× longer (bunny 170 windows, dragon 135); the stop rule relied on the tug-of-war's rejections.
  - thin uncovered roughly doubled in both arms (LV9 5.3 / 6.0 %, PV9 7.2 / 7.9 % vs about 3.3 %).
  - twins worse (PV9 0.9849 / 0.9822 vs 0.9869 / 0.9875).
  - bunny normal error median 10.5° vs 7.6°.
- 40k gallery held: tmp/d119.queue moved to d119.queue.held.
- Remaining: bands_{bunny,dragon}.log in output/gpu/d119 (d119_relief.sh); bunny PV9 runeval3.
- Then: write the result into experiments.md, revert the code as for D117 (commit), and tell the user.
- Tools: tmp/d119_bands.sh, tmp/d119_course.py, tmp/early_gauss.py.

**D118 (Xu TVCG 2025)**
- The reimplementation is faithful: at 11k it reproduces the paper (α 1e-2: −97.9 % vs −98.3 %).
- Fig. 8's α 1e-3 under-steps. Best Xu = α 1e-2 on both meshes; it is behind D105 on every measure except the net/gross momentum ratios.
- Pending: merge xu-paper-protocol into settled-base (experiments.md conflict); the ablation ladder.

**D120 (proposed, awaiting the user's go)**
- Early raw Gaussians at 1–3 s (D108): the render share is 0.55–0.70 in windows 3–11 against the designed 1/3, because of the λ EMA (0.3) lag (render_loss.py balancer).
- Proposal: λ = the rule's value at every window, i.e. drop the EMA.
- Display "apart" discs in the first 10 % exist in all 300k runs since D70; twins have about half (dragon).
- The kernel is CUBIC B-spline (4×4×4 stencil); I once wrongly said quadratic.
- Mouth flakes (user crop, blue on black, source unknown): likely sub-pitch teeth. Forensic pending: the user was to name the source.

**Figures** (docs/figures/, not committed)
- Confirmed final = `pipeline_overview_final.{pptx,png}` (= _v2: glossy D115 bunnies + shadows + Stomakhin-style thin MPM arrows). Also kept: _v1, _v3 (legend / 4×4), the bunny/dragon matte versions, simple (Lato), and parts/.
- In progress at the restart (figure agent, local): parts/ back to v2 (v3 parts to parts/_v3), and `pipeline_overview_final_gs` (strip bunnies as raw 3DGS, image-10 look). Re-issue both if lost.
- Dragon side video: output/video_2026-10-06/dragon_side/ (az 90, head right; tmp/side/cdm_side.py).

Related: [[resume-2026-10-05-night]], [[competitor-landscape]], [[no-parameter-only-fixes]].
