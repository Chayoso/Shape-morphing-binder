---
name: resume-2026-10-22-early
description: "Handoff 2026-10-22 early morning before a local shutdown: TOP PRIORITY render; λ-coupling root cause confirmed (D159/D160/D161); D160c colour moves and D161c carves letters with λ held, but rest/shape break; D160d + D162 + D159c running; D156g CPU; B9b step 1; SHOWCASE 1 card; pending user decisions"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-22T18:44:53.327Z
---

**Top priority (user, 2026-10-21 23:00): keep going until the render gives a meaningful result.** The full detail is in [[render-own-contribution]].

**Where it stands:**
- **Root cause, measured by three tracks:**
  - The λ rule ties the render weight to the vanishing physics residual. After arrival λ falls 20–380×, while the physics hold and the fixed-weight cleanup (near band, spray) stay.
  - D159 (detail pictures, 300k): FAIL, the same starvation.
  - D160 step 1: the cleanup out-pushes the colour after arrival.
  - D161: proximity holds the relief.
- **With λ held (`--lambda_ema 0`, the existing pre-D81 option), the render really acts:**
  - D160c colour: r 0.63–0.93 on band and turn, +3–5 dB held-out PSNR. The turn sheet shows a top/bottom → front/back split.
  - D161c: 47–58 % of the relief ceiling, "PM" readable at 40k.
  - Both break shape (C2/P3) and rest (C3/P4).
- **Running (each agent writes tmp/<exp>/RESUME*.md):**
  - **D160d** (D160 agent): the colour term held, the geometry render terms at the rule λ. 40k chain PID 383480.
  - **D162** (D161 agent, wip/d162 merged branch): the render weight from its excess over a measured floor, plus the placement terms at the loss cell. Then 40k relief + colour + a 40k gallery regression, then 300k scales.
  - **D159c** (D158/D159 agent): D159 pictures + λ held at 300k bunny/dragon, as an upper bound. It starts after the D159 renders and close-ups, which include an r = 2.5 display reading of the scales.
  - **D156g** (D156 agent, CPU): a uniform-solid carve for image-only targets, plus a check of whether the full arm lowers its render term by internal arrangement.
  - **B9b step 1** (40k, the settled load deformation; lowest card priority).
  - **SHOWCASE** on 1 card (remaining runs and renders). hflip at encoding, because StudioRaster mirrors the image (fix the renderer later, for all deliverables at once).
- **Pending user decisions:**
  - (1) The λ-rule change, once D160d/D162 show a definition that keeps rest and shape.
  - (2) The stop rule: C stops early at 300k on 2/2 seeds. It is a slow tangential drift that would settle by itself in 2–4 s. Option (a): don't count null commits as stale while moving. Option (b): the stop rule requires rest.
  - (3) The cell default (`--cell_follows_n`): it opens the teapot genus and makes the surface more accurate, but leaves early holes and far pieces. 5–7 h of extra runs are needed before deciding.
  - (4) A tail stop at 90 % of the time.
  - (5) Whether the earlier "후처리를 render loss가 하게" meant the spray cleanup. D160 step 1 found the cleanup to be the after-arrival blocker.
- **Added later on 10-22:**
  - **D160d done:** with the colour term held, r 0.67–1.00 on every scene and +3.5–5.6 dB PSNR. It fails 12/80 shape cells (7 of them in turn) and C3 rest, because the colour is unfinished when the run stops. Next: the colour weight follows its own excess over its floor, which is what D162 tests.
  - **B9b on hold.** The root cause is the frozen F smoothing: 0.955 keeps 4.5 %, so under sustained load E acts about 22× softer. The drop scene's E×7 was compensating for it. User decision: (a) the grid static-equilibrium subtraction, or (b) the smoothing's behaviour under sustained load.
  - The dissertation zip agent is updating Chapter 4 and the figures and drafting the abstract. The backup is docs/Changyong___Dissertation_orig_2026-10-22.zip.
  - 20 GB of server cleanup (D147 v0 runs, RGV first-design viz): 457/500 GB.
- **13:45 CDT state:**
  - **D162** chains run detached (alpha 3066387, beta 3067953, after 3067961). The A1 scene runs finished at 13:40; the A2 and B gallery runs continue. The agent is resumed to read.
  - **D162 found a frozen-code issue:** the commit_replay tolerance is max(1e-7, 10 × replay_rel) from one replay pair at rest, against atomics noise, and it nulls 4.1 % of the tag's windows (d145t_b1 222/5370). That is the user's call.
  - **D156i:** FAIL. Cause: the visual hull sits up to ½ px inside the true silhouette, which also confounds D156's 36 runs. D156j (conservative hull, +½ px) was resumed.
  - **D159c:** FAIL. With λ held the particles still do not carve the scales: they are 0.12–0.41 pitch deep at 300k, below what the representation holds. Rest, thin and pieces get worse.
  - **SHOWCASE** done: 27 examples, montage, 8 recommended (output/results_2026-10-21/showcase300k/).
  - **Dissertation zip:** done. New Ch4 and figures, PDF at docs/dissertation_build/dissertation_2026-10-22.pdf, abstract drafted (347 words). The user has to decide: paragraph numbering, the Ch1/2/6 inconsistency with the new Ch4, 8 Ch4 TODOs, and replacing the silhouette-residual claim with the D157 exact-mesh numbers.
  - Disk 426/500 GB.
- **Rule slips in this period:** local `python3 -` (D160 agent) and `python3 --version` (B9b agent), both without effect. All agents were told again.
