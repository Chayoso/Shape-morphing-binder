---
name: render-own-contribution
description: "User goal 2026-10-21 — the paper needs a contribution only the render can make; why the render only polished so far (D156/D158) and the three running tracks D159 (detailed targets), D160 (colour layout = physics null space), D161 (render-carved relief)"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-23T16:05:08.675Z
---

User, 2026-10-21 evening: "이제는 진짜 '랜더'만의 뭔가 contribution이 필요해". They chose the colour-layout track (D160), then added their own idea: "색이나 surface를 깎아서 진짜 그걸 만들게 하는 건 어때" (D161).

**Why the render only polished so far (measured):**
- The render's pictures carry the same information as the 3D target, so the transport dominates and the render adds sub-pitch rim polish (D157: Boundary IoU −16 %, F@0.5 −36 %, PSNR +0.7–1.0 dB at 300k).
- D156 (visual hull as 3D target + true pictures, 40k): FAIL.
  - The render could not carve concavities: the silhouette pushes outward, the shading is depth-blind, and the transport holds the hull's mass.
  - D156b (photo hull, Kutulakos–Seitz carving as the target) is running.
- D158: the in-loop target pictures are drawn from 300k target samples through the same blurry field, so they carry no detail and the render does not ask for any.
  - It was not the picture resolution (188 px sees 84–95 % of the effect) and not the u resolution (u holds 72–88 %).
  - D159 (detailed target pictures, shell offset measured) is running.

**Key argument for the paper:** every physics term (Sinkhorn transport, proximity, end drift, released motion, cleanup) depends only on the SET of particle positions, so it is blind to material identity. Which material ends where (colour or texture layout) is therefore a null space of the physics objective, and only the render can choose it.
- D160 tests this: particle albedo is fixed and advected, never optimised; RGB shading; with white albedo the run is bit-identical to the tag.
- D161 (relief) is NOT a null space. The proximity term and the relaxation resist relief, so the agent measures a response curve first.
- Precedents: Alexa & Matusik 2010 "Reliefs as Images", Weyrich 2007, Shadow Art 2009.

**The user's widened criterion (same evening):** "뭔가 그 dragon의 비늘이나 bunny의 주름 이런 것들을 만들 수 있지 않을까 ... 일단 rendering이 physics에 영향을 줘서 physics의 방향을 바꿀 수 있는 방법이면 뭐든지 돼".
- Anything by which the render changes the physics' trajectory counts, not only render-only information.
- The concrete wish is the real mesh's fine features grown through the physics: dragon scales, bunny wrinkles.
- D159 now reads the scale and wrinkle regions on the particles' own surface. D159 and D161 also report, per frame, the full-vs-twin surface difference against the twin-vs-twin seed spread.

**TOP PRIORITY (user, 2026-10-21 ~23:00):** "일단 최우선 목표는 rendering이 유의미한 결과 낼 때까지 계속 진행해 줘".
- The render experiments get cards first; SHOWCASE dropped to one card.
- D161's response curve (40k bunny, w = 4 pitches): the physics hold on sub-cell relief is ~180× the render's push. The terms, largest first: surface proximity 1.06e-5, spray 5e-6, transport 1.3e-6, near band 1.2e-6, against the render's 1e-7. The equilibrium relief is 0.04–0.05 a, where the render alone asks 0.18–0.44 a.
- The candidate root definition is that proximity and cleanup act below the scale the 3D target resolves. D161b is to name the scale and test "physics owns the loss-cell scale, the render owns sub-cell". D159 is told to read the per-term hold too.

**D160 result (2026-10-22 00:40, 40k): FAIL, r = 0.01–0.09.**
- The null space was verified exactly: the twin's trajectory is bit-identical under any colouring, and white albedo is bit-identical to rt1.
- Colour is decided only during the flow (0.6–2.4 s).
- After arrival λ falls ~120× together with ‖g_phys‖ (0.31 → 0.0026), and each window moves about 0.008 pitch.
- Leading hypothesis: the λ rule ties the render step to the physics gradient, so the render is starved exactly in the null space; the stop rule may also end the run early.
- Next: D160 step-1 limiter measurement, then D160c (the pre-D81 once-calibrated λ, an existing option) or D160b (the D155 moving target).
- D159 and D161 also read λ and the steps after arrival.

**2026-10-22 night: the limiter is confirmed by three tracks.**
- D159 (detailed pictures, 300k): FAIL. After arrival λ and the step fall 20–70×, and the detail band ends at the seed floor. The particles already hold ~4/5 of the relief, and the r = 3 display field hides it.
- D161 plain: 8–19 % of the relief ceiling.
- D161b (proximity radius = berth): worse, because λ falls with the hold.
- **D161c (λ held, `--lambda_ema 0`): 47–58 % of the ceiling carved into the PARTICLES, and "PM" is readable in the display at 40k.** It costs rest (v_mean 3–11× the twin) and shape (Chamfer +0.04 / +0.12).
- The λ rule ties the render weight to the vanishing physics residual. The physics hold is constant, so "proximity holds" and "the render is starved" are the same ratio.
- **D162 (led by the D161 agent):** (1) the render weight referenced to its own excess above a measured floor, plus (2) the placement terms read at loss-cell resolution. One merged branch wip/d162 (appearance + render_tgt + detail pictures). 40k relief and colour plus a 40k gallery regression, then 300k scales.
- Also running: D156g (uniform-solid carve, CPU) and a check of whether the full arm lowers its render term by internal arrangement.

**D160c (λ held, colour, 40k): the colour MOVES.**
- band r 0.91/0.93, turn 0.63, half 0.89/0.79; held-out colour PSNR +3–5 dB.
- In the turn sheet (output/results_2026-10-21/d160_appearance/d160c/sheets/bunny_turn_97_end_lambda_held.png) the twin keeps the transport's top/bottom split, while the full arm rotates toward the target's front/back split. This is the strongest visual render evidence so far.
- Costs: C2 shape (40 of 80 cells out of range), C3 rest (end KE 5–80×), erratic window counts.
- Next definitions:
  - D160d (D160 agent): the colour term keeps its first-window weight, the geometry render terms keep the rule λ.
  - D162 (D161 agent): the render weight from its excess over a measured floor, plus the placement terms at the loss cell.

**The full arm's render term is partly arrangement (D156h/h2, 2026-10-22, tag 300k seed 97).**
- The full arm's own render term falls BELOW the floor of the true mesh's uniform samples: bunny 0.46–0.58×, dragon 0.69–0.79×. The twin sits at 1.4–1.9×.
- At least 43–56 % (bunny) and 21–26 % (dragon) of the full-vs-twin render-term gap is internal arrangement: an aligned, smoother exterior, mostly in the silhouette term.
- **Paper claims about shape must use held-out views or the exact-mesh distance (D157), never the training render term.**
- D162's floor-referenced weight must be checked for arrangement eating the floor.

**User 2026-10-22 ~14:00:** "VANDERBILT나 SIGGRAPH 글자 … 가능한 모든 방법을 사용 해 보자".
- **D163** (new agent): sphere → a smooth plaque, with "SIGGRAPH" / "VANDERBILT" embossed in the render pictures only.
  - Arms: twin, rule, λ held, D162 F/FC. 40k quick check, then the 300k deliverable.
  - hflip at encoding, because StudioRaster mirrors the image and text would read backwards; raking light.
- **D164** (user-approved frozen fix): the commit_replay tolerance. It nulls 4.1 % of the tag's windows through atomics noise. Built on wip/d164 from rt1 with a 40k gallery and a 300k pair check; no tag without asking.
- **B9b gravity:** NOT pursued. The user: proceed only if it is the fallen-bunny example, and B9b is the standing scene. The drop scene's E×7 is affected only indirectly.

**D162 / D162b (2026-10-22 evening): the weight's reference is not the blocker.**
- D162 (floor read from the state, F): arrangement drove the term to the floor. The weight flipped on and off, and 6/8 relief runs ran to the window cap.
- D162b (the excess read from a uniform resample of the solid, S): the flipping is gone and runs stop normally, but:
  - b's 3-pitch field erases 1-pitch letters, so S reads "done" before the letters are carved (relief 12–42 %);
  - arrangement still eats the term (0.60× the floor while the shape stays flat).
- Band colour passes with F and S, but D160d (colour term held) is still best.
- D162c is measuring how the render moves the outer particles: normal vs tangential, in-band vs sub-pitch. It contrasts the successes (D161c, D163 H, D160d) with the failures, to choose a normal-only or band-limited push.

**D162c (2026-10-22 night, measurement): the push PATH is the same in success and failure; the weight's MAGNITUDE after arrival decides.**
- The render's push is mostly normal and about half in-band in every run.
- The control filters out the sub-pitch part.
- Successful carving (D161c, D163 H) moves the outer layer 2–30× the twin, mostly as in-band tangential flow: material flows along the surface to form the letters.
- Failed runs (F, S) move only as much as the twin, because their weight switched off early.
- So a normal-only push would kill the flow that carves; band-limiting changes nothing.
- **D162d (running via the gpu1 queue):** F's form with L_r and L_floor read from pictures low-passed at one pitch, so arrangement cannot cross the floor and the letters stay visible.
- **gpu1 queue:** tmp/gpu1_queue/ (runner holds the gpu1 lock; jobs NN_agent_name.sh). Order: RGV render-authority run → D162d → D164 → RI videos.

**User 2026-10-22 night:** "예전처럼 0.0012 이딴식으로 줄이는 게 아니라 확실히 surface의 주름을 만들던지 SIGGRAPH/VANDERBILT를 보이게 하던지… 이런게 dissertation에 들어가야 할 듯. 지금 거의 다 surface control u에 따른 결과 같아."
- The dissertation must show VISIBLE render results: letters, image-only wrinkle relief, colour layout.
- It must also answer "render or u?": the twin has u and stays smooth, and a u-off arm with render authority tests which channel writes.
- Plan: D162d at 40k, plus the u-off arm, then 300k deliverables through the gpu1 queue (jobs 22–29): the plaque, a wrinkle-relief scene and colour. Afterwards Chapter 4's method must include the render's authority (the weight definition) and these results.

**User 2026-10-22 late:** "끝에서 덜 멈추는 대가와 가중치 정의 문제 … 이 문제까지 해결해서 rendering loss 쪽 무조건 해내자."
- The rest cost under render authority must be SOLVED, not documented.
- Next: D162h-settle, a zero-control continuation from the λ-held end states. It decides between (A) a premature stop by the noisy render-dominated merit (fix: a stop rule that requires rest and settles with zero controls) and (B) tension, where the letters are held only by live controls (fix: placement_at_cell D162g, or plastic baking).
- Dissertation plan: finish the edits with the existing results now, then swap in each GPU result as it lands.

**User 2026-10-23:** "이전이 Render-Guided의 특성이 컸는데, 지금은 Engineering 짬뽕이 된 느낌… 지금 하는 실험이 그걸 증명해야… 무조건 해 보자."
- Ch4 must become render-centric again.
- Queued in this order, results in output/gpu/d162/auto/<job>/summary.md:
  - 27: settle test;
  - D162i (auto if 27 reads (A));
  - 26: H+C;
  - 29: stiffness ×3 / ×7 for the lettering;
  - 29b: the render-centric minimal recipe (render authority + w_c = 0 + placement_at_cell) on lettering, colour and 6 gallery meshes, against the tag.
- If 29b holds, the method can drop the cleanup and lead with the render. That also answers the user's earlier "spray 후처리를 render가 하게".

**User 2026-10-23:** up to **500k particles allowed for render-influence experiments**. This is an exception to [[algorithm-before-n]] for this track only. Use it when the card memory allows, since nurshat shares gpu1.

**User 2026-10-23: TOP PRIORITY = D165 stamp test** (D147 agent).
- Ours (the D163 H carved plaque, letters in the material) against physics-only plus a post-hoc display decal (the twin plus a D115-style fit to the same relief pictures).
- Both are pressed into a soft slab. Ours should imprint the word; the decal cannot.
- Answers the expected reviewer question "physics-only + post-hoc surface recon/fit?". Summary: output/gpu/d165/summary.md.

**D165 RESULT (2026-10-23 11:30): the stamp test PASSES at 300k.**
- Ours (the H carved plaque) pressed into a soft slab prints a readable SIGGRAPH in the slab's depth map: contrast 0.13 a0, letter corr 0.46.
- The twin with the post-hoc decal leaves a flat print (corr −0.28).
- Our letters survive with the controls off (0.30 → 0.35 after settling).
- The decal still looks lettered too, so only the imprint separates the two. That is exactly the reviewer answer.
- Caveats: faint in the studio close-up; at 40k nothing reads; pressed without gravity.
- Files: output/results_2026-10-23/d165_stamp/ (imprint_depth.png, videos, decal_compare.jpg); docs a966aa18.
- Also from 22: the coin letters are written about 2/3 through u, while the bunny flank is written by dF_c alone. 26 (H+C) carves the most (0.55–0.83). Rest still fails; the settle rerun is job 50.

**How to apply:** judge every new render experiment by whether the full arm differs visibly from "dF_c + layer u" for a reason the physics objective cannot see. Never fall back to "limitation" ([[no-limitation-escape]]). Arm names follow [[arm-naming]]. Related: [[render-must-beat-physics-300k]], [[physical-detail-not-display-trick]], [[siggraph-target]].
