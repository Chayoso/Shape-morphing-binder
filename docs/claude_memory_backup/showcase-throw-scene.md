---
name: showcase-throw-scene
description: "User's showcase scene for the paper (2026-10-14) — two spheres thrown on parabolas, collide and bounce apart, each morphing into its own bunny, ending as tumbling bunnies; texture on the best scene later; D147 design agent"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-17T22:28:29.972Z
---

**The scene (user, 2026-10-14).**
- Two spheres are thrown on parabolic paths. They collide while morphing, stay SEPARATE (each becomes its own bunny), and the final state is two bunnies tumbling or rolling ("나뒹굴거나").
- The user rejected the easier merge into one bunny.
- Texture on the "best scene" comes later ("best scene엔 나중에 texture도 넣어야").

**What it needs (D147, design-first agent):**
- Multi-field per-body grid contact that separates (Bardenhagen / Nairn / Huang 2011 / Homel–Herbold), differentiable. The single-body default path stays bit-for-bit.
- A rigid-motion-invariant objective per body: centre of mass plus a best-fit rotation (Müller 2005 shape matching), so gravity, collision and the floor move the body freely while the controls only shape it.
- Two sources and two targets, with transport per body.
- Floor friction for rolling.
- Tests that controls inject neither linear nor angular momentum.
- A 40k prototype first, then 300k.

**D147 design (c38283b), approved by me under the autonomy rule on 2026-10-14:**
- Contact:
  - one grid field per body, Bardenhagen multi-velocity at shared nodes;
  - a shared normal from the node-mass gradients;
  - Nairn's volume gate V₀+V₁ ≥ dx³ (no constant to tune), approaching-only, Coulomb μ 0.5;
  - the floor treated as an infinite-mass body behind the same gate.
- Body frame: the Buckens / shape-matching polar frame with an exact custom backward. Measured to turn ≤1.5° on a single-body morph, so it is near a no-op; V0 will check that.
- Velocity terms work on the deformation velocity (the König split).
- Scene:
  - two bodies of 150k each, v_rel/c 0.1, G 0.05;
  - apex collision at 1.0 s with a 0.5 R offset;
  - 8 s, 48 windows, drag 0;
  - no truncation;
  - a visible grey floor with a contact shadow; the bunnies face each other.
- Plan: code by 10-17, V0/V1 on 10-18, the 40k scene by 10-21, the 300k scene by 10-24, video by 10-28.
- Cut order if late: the second seed, then friction, then the 300k twin, then the 300k scene.

**M-b prototype results (19dcddf, 2026-10-14 02:26):**
- Gated multi-field contact: clean bounce, e = 0.74 head-on and 0.68–0.69 oblique; momentum kept to 4e-6, angular momentum to 8e-5 with friction.
- Ungated Bardenhagen starts contact at a 2.6-cell gap.
- The single field does NOT stick in a fast collision (e = 0.81), but leaves a one-cell air cushion and crushes the facing layers (J 0.74). The case for per-body fields is that visible gap; slow or resting contact was not measured.
- The gate volume is fixed by definition: per-particle volume = fill volume / N.
- The prototype's slice tape gave a zero adjoint (P7, unexplained); the real path's FD test has not run yet.

**User feedback, 2026-10-14 ~14:30, after the 40k o100 video:** "지금 예제는 거의 morph가 다 되고 나서 부딪히는 거라 조금 아쉽네". Requested:
- the spheres launched 4–5× farther apart;
- a slower morph, so they collide mid-morph.

Plan: a target progress s(t) along the OT displacement interpolation, or similar. Pre-registered target: morph progress at first contact 0.35–0.65. 40k first, then 300k; the old spec is not run at 300k.

**2026-10-14 ~15:20, throw40k_a (scene_throw):**
- The gap between the spheres is 4.5× (5.6 wu); first contact at 4.05 s.
- Target progress s(t) along the transport, with t_m 7.6 s; morph progress 0.41–0.42 at contact.
- The bodies part at 6.3 s and land at 8.9 / 11 s; end error 0.029 / 0.039; tumble 30–40°. Passes in 2/2 runs.
- The ending bounces high (drop ~11 wu) and starts a second mid-air collision at 16 s.
- USER: option (A), keep it and cut the delivered video at ~14 s, before the second collision. The cut frame comes from a rule on the 300k run.
- The 300k run (2×150k) starts at 19:00: s97, then twin, then s99.

**Why:** a physics-preserving morph as the SIGGRAPH video's opener. It resolves the 10-01 open "body-frame objective" question for this scene.

**How to apply:** the paper must be finished by the end of November ([[siggraph-target]]), so keep the minimal path and cut if late.
- Texture idea for later: per-particle colour, carried from the source or assigned from the target's texture through the transport map, drawn by scripts/render_deliverable.py.
- The multi-field contact could later also answer the narrow-gap beads ([[d137-cell-from-n]]).

**State 2026-10-14 14:10 CDT (D147 agent; docs bea151d, code wip/d147).**
- The long-flight scene (`specs/scene_long*.json`):
  - `--assim 0`, plus the floor's position constraint (FloorTrajectory);
  - parts in the air 8 of 8 at 40k;
  - the tumble bar (>30° after landing for each body) is a run-to-run draw, because the scene is nondeterministic for the same seed.
- The pick by the rule, after the timeline reading fix (the first parting, not the last impulse): `scene_long_o100` (vertical offset 1.0 R).
- Chain 10 runs on hyde06 from 19:00 out of repo_d147l: 300k (2×150k), seed 97, then the render-off twin and seed 99, the 4K render, read_300k.txt.
- Coordinator's rule: deliver the 300k seed that passes; if none passes, deliver the one with the largest tumble and say the bar was passed by chance.
- ΔL split: the pair's L is kept to 1e-5 through the contact. The −17,900 was the spins summed: friction turns orbital momentum into spin. u, spacing and relaxation each carry ≤ 2 % of the spin. Proposal only: a spin term in u's projection.
- Renderer: the camera now follows the pair (render_scene.framing).
- Before settled-base is fast-forwarded: the suite runs in chain 12 (repo_d147m).

**2026-10-14 ~16:50 CDT (continuous target, a17133a):**
- The late snap came from switching targets at s = 1. It is fixed by Z_s = Y + (1−s)·D_Y(Y).
- Passes at 40k in 2/2 runs: contact progress 0.45 / 0.39, snap ratio ≤ 2.9.
- Static camera fitted to the box corners.
- USER kept t_m 7.6 s ("지금 그대로"): at contact the bodies read as short-eared bunnies.
- 300k s97 on this definition started at 16:24 (~56 min).
- Defect: in throwY40k_a a 13-particle face piece of body 1 tears off mid-air at 12.5 s, during the recovery from the landing squash, and is still out at the cut.
- Attribution queued before the 300k twin/s99, pre-registered: u_off, render-off twin, D146b. If D146b cleans it, the showcase 300k waits for the re-freeze.
- 300k s97 (17:08): judge PASS, end error 0.021. One particle at 3.18 sp in one frame (NOT clean by the letter).
- Amended cut: the earlier of the first grid contact impulse after parting and one spacing; 15.33 s.
- Attribution at 40k: u_off disintegrates; the render-off twin and D146b both keep 1–4-particle excursions after landing. The 13–24-particle droplet is a run-to-run event.
- DECISION (mine): s97 is the interim showcase, with a 4K fallback render on hyde01. The final scene 300k (arm, twin, s99) runs once on the re-frozen recipe.

**State 2026-10-14 18:10 CDT (supersedes the 14:10 block).**
- Spec: `specs/scene_throw.json`.
  - Launch gap 4.5× (t_contact 4.3 s), lift 4, 16 s, morph_time 7.6.
  - Target on the transport read from the target side: Z_s = Y + (1−s)·D_Y(Y), in `scene/schedule.py`. This removed the late "snap". The user keeps the short-eared bunnies at contact.
- Interim showcase: 300k s97, `output/results_2026-10-14/d147_throw/300k_s97_4k/` (cut 0–15.30 s plus a contact close-up).
- Cut rule (amended): the earlier of the first grid contact impulse after parting and the bodies coming back within one spacing.
- Camera: still by default (`render_scene` corner-projection fit). The contact close-up is fitted over its own span. Renders go to hyde01 GPU 1 via `scripts/ops/hyde01_render.sh --scene` (+res=1080 for previews).
- Droplet attribution:
  - u_off: the bodies fall apart.
  - render-off and D146b: they still shed 1–4 particles to under 4 spacings.
  - Run a's 13–24-particle droplet is a one-run draw.
  - s97 has one particle at 3.18 sp: NOT clean by the letter.
- Next: the final 300k (s97, twin, s99) runs once, on the re-frozen recipe. wip/d147 is merged with refreeze-prep (16910d6); the full suite on that merge is not yet run.

**State 2026-10-15 22:00 CDT (final scene on tag freeze-2026-10-15; supersedes the next-step line above).**
- Code: wip/d147 caff236+ (the tag merged cleanly; only the scene files differ). Deployed as repo_d147fin. Suite 435 passed; the known-flaky settled-contracts test failed once on the first suite.
- Runs: `output/gpu/d147/runs/final300k_{s97,s97_off,s99}`. Local copies in `output/results_2026-10-15/d147_throw_final/` (videos, judge_table.md, README_ko.md).
- Judge FAIL:
  - s97 passes, end error 0.033;
  - the twin passes, 0.042;
  - s99 fails, 0.093.
  - None is clean. s97 is WORSE than the interim (600 vs 2 far particle-frames; end error 0.033 vs 0.021).
- Forensic (scripts/probes/d147/forensic.py):
  - The landing squash is equal in all runs, the interim included, and the grid moves it, not u or the relaxation.
  - The final runs lose control during the recovery: commit-replay nulls in landing windows, and the loss-raster trap. Body 0's rebound passes 5.28 wu in its own frame, the transport goes inf, and both bodies run free toward the sphere (`--assim 0`).
  - Not attributable from one run each.
- Proposed, NOT launched (the coordinator decides): the 09b path vs the tag, seed 97, 2 runs each (~2.9 card-hours).
- Scene-definition questions: the per-body loss raster is too small for the landing; one body's inf term stops both bodies; a replay null commits zero control.

**State 2026-10-16 14:00 CDT (the four fixes and their 40k gate).**
- The user approved all four fixes ("1–4 다 고쳐서 40k 확인"). They are implemented on wip/d147: fa0d091 plus test fixes up to ea41f6f, timeline.py and gate.py after. Deployed as repo_d147fix.
  - (1) `scene/raster.py`: the per-body raster grows to cover the uncontrolled continuation.
  - (2) A non-finite body is left out with zero controls.
  - (3) On a replay mismatch the window is solved again, up to the patience.
  - (4) `mpm/bodies.py`: each body's deformation velocity is damped at 0.9/s in P2G, keeping P and L.
- Tests: suite 446 passed, exit 0 (the digest included).
- 40k gate (chain20; `runs/gate_40k.txt`): FAIL by the letter on item 3 (clean). s97 has a 13-particle piece and the twin a 10-particle piece; both are body 1's ear tips 2.2–3 s after landing. s99 passes everything.
- All other items pass in all three runs:
  - no zero-control window, the raster never grew, no body left out;
  - 5–8 replay retries a run, all resolved;
  - heads 0.61–0.74 sp against the D135 references' 0.93–1.57;
  - judge PASS, end error 0.025–0.028;
  - momentum kept.
- Cost: +50 % (17 min vs 11.4). The damping's single-address atomics and their adjoints double the gradient time.
- Two readers were fixed after the runs and are stated in the docs: timeline.py (read40k.py crashed on the missing scene40k_L.npz), and gravity's torque taken out of the free-flight momentum item.
- The 300k was NOT launched (the rule: on a fail, stop and report). Next: the coordinator's decision.

**State 2026-10-17 12:30 CDT (ear-tip forensic + damping speed-up; user: "끝 조각을 먼저 없애는 거랑 속도 개선을 한 번에").**
- Forensic: 40k dump runs (`scene_run --dump`, `scene/dump.py`; `runs/dump40k_s97*_dump`, 2.7 GB each) read by `dump_forensic.py`.
- Root cause, which is in the FROZEN path, so the work STOPPED and was reported:
  - Body 1 rests squashed on the floor for 2 s after landing (8.9–11 s).
  - The grid flow draws the ear tip out 3–16× while the stored F stays 0.95–1.13. The frozen F smoothing s = 0.955 keeps 4.5 % of each step's increment.
  - The controls sit at the frozen cap `dfc_clip` 0.02 over the whole body.
  - The grid (G2P) opens the link at lift-off. u and the relaxation do not, and later pull the piece back.
  - The same particles (300, 23414, 37974) detach in every run with the same body-1 sample.
- Damping speed-up done: block sums plus a hand adjoint (8f2af48, repo_d147fast). Suite 448 passed. Equal within the run-to-run spread. 5.13 s per window vs 7.75 (old damping) vs 4.8 (no damping).
- 2026-10-17 13:45, after the user chose the stiff material ("재료 단단하게"): the stiff 40k gate is read (chain23).
  - Setup: `--young 9.8e5` (B8's E×7) plus `specs/scene_throw_E7.json` (G/7, v_rel/c ÷ √7). The world-unit motion stays the same; only c grows ×√7.
  - Physically excellent:
    - no far piece in any run;
    - ear-tip stretch 1.07–1.16 vs 2–3 before; heads 0.52–0.54;
    - judge PASS, end error 0.023–0.026;
    - 12.4–13.9 min per run;
    - elastic bounce (11–20 sp), more tumble, not at rest.
  - By the letter the gate FAILS: the amended impulse cut fires on the first contact's tail (5.5 / 5.8 s), which empties the head window.
  - Proposed: the impulse cut counts only an approaching contact (no constant). Waiting for the coordinator's go before the 300k.
- 2026-10-17 17:30 CDT: DELIVERED. The user approved the cut amendment ("진행"); `cut.py` c05c409 counts a grid impulse only while the bodies approach. Under it the stiff 40k gate passes.
  - Stiff 300k (chain24; repo_d147fast 8f2af48; spec scene_throw_E7 + `--young 9.8e5`):
    - s97 passes every item and is delivered: end error 0.015, silhouette 0.985 / 0.984, no piece, heads 0.52, tumble 68° / 127°, bounce 16–34 sp;
    - the twin passes (one 4-particle piece);
    - s99 fails narrowly: body 1 landed on its head (recovery mean 1.20 > 1.007), free-flight angular momentum 1.05e-3 > 1e-3.
  - Deliverables: `output/results_2026-10-17/d147_throw_fixed/` (15 videos, judge_table.md, README_ko.md, before/after sheets). Docs 87739e6.
- (Earlier) Not run: no new 40k gate, no 300k. Waiting for the user's choice:
  - accept the small pieces;
  - a scene-only material or cap change (frozen-algorithm territory);
  - a gentler landing (geometry);
  - a stiffer scene material.

**DELIVERED 2026-10-17 (stiff material). Docs 87739e6; code wip/d147 8f2af48 + the cut amendment c05c409.**
- Setup: spec `scene_throw_E7.json` = B8's stiff material E×7, with gravity and the throw kept in world units. Plus the four fixes (raster, per-body inf, replay retry, deformation-only damping 0.9/s) and the damping speed-up.
- Amended cut rule (user-approved after the stiff 40k gate): a grid contact counts as the second collision only while the bodies approach.
- **The delivered run is stiff300k_s97.** It passes every item.
  - end error 0.015; silhouette IoU 0.985 / 0.984;
  - 0 pieces; head 0.52 / 0.52; 0 zero-control windows;
  - tumble 68° / 127°; bounce 16 / 34 spacings; 50 min.
  - Files: `output/results_2026-10-17/d147_throw_fixed/stiff300k_s97_4k_cut.mp4` (16 s whole run) and `_contact.mp4`.
- Gate overall FAIL by the letter, in s99 only: the head recovery window catches the tail of a head-first landing squash (1.20 against bar 1.01), and momentum is 1.05e-3 against the 1e-3 bound. The twin passes.
- The bodies are still tumbling at 16 s, i.e. not at rest. This matches the user's "end tumbling".
- Root cause of the earlier ear-tip pieces: the frozen F smoothing keeps only 4.5 % of each step's deformation as elastic strain, so the soft material flows under the landing squash. The stiff material removes the squash.

**2026-10-21 (user):** "gravity 이거 내가 원하던 게 아닌데... 그냥 난 엄청 위에서 떨구는 sim을 원했던 건데".
- The B9/B9b gravity runs, where the body morphs standing on the floor, are NOT the user's intended gravity demo.
- The intended one: one body dropped from high up, morphing during the fall, landing and bouncing / tumbling.
- D147 agent: `scene_drop.json`, using the throw-scene machinery, stiff E×7, the morph ramp ending about at landing; 40k then 300k; studio render.
