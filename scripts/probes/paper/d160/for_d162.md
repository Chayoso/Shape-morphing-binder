# D160 -> D162 note (agent D160, 2026-10-22 ~03:00 CDT)

Code: `--appearance SCENE` on branch wip/d160 (rt1 6c9b0e9 based; 209ba6d at the D160c runs), default off. Scenes in
physmorph/pipeline/appearance.py (white / half / band / ears / turn). The physics never reads an albedo; white = rt1 bit
for bit; the lambda = 0 twin's trajectory is the same with or without paint (tests/test_appearance*.py).

## What limits the colour (D160 step 1, docs e75895f9)
- At the colour front the colour's push is 1/5-1/10 of the physics and 1/8-1/25 of the cleanup after arrival; the
  transport/proximity are orthogonal to it (cos ~0); the fixed-weight cleanup pushes back along the needed direction.
- The lambda rule (lambda |g_render| = 0.5 |g_phys| every window) keeps the render at the physics' scale: lambda
  0.31 -> 0.005 by arrival; the colour moves only while the body flows. Stop rule / outer rejects are not the limiter.

## D160c: lambda held at the first window (--lambda_ema 0), docs 1ef311f6
| scene | r rule lambda | r lambda held | mis twin -> held | held-out colour PSNR twin -> held |
|---|---|---|---|---|
| half  | 0.04 / 0.07 | 0.89 / 0.79 | 0.057 -> 0.009 / 0.059 -> 0.015 | 18.80 -> 22.20 / 18.73 -> 21.57 |
| band  | 0.06 / 0.09 | 0.91 / 0.93 | 0.099 -> 0.015 / 0.104 -> 0.013 | 16.77 -> 21.19 / 16.64 -> 21.49 |
| ears  | 0.04 / 0.03 | 0.19 / 0.40 | 0.128 -> 0.105 / 0.128 -> 0.078 | 16.28 -> 16.79 / 16.23 -> 17.67 |
| turn  | 0.01 / 0.02 | 0.63 / 0.63 | 0.433 -> 0.168 / 0.436 -> 0.169 | 10.70 -> 14.73 / 10.67 -> 14.65 |
| dragon band | 0.03 / 0.03 | 0.73 / 0.34 | 0.195 -> 0.056 / 0.192 -> 0.129 | 15.22 -> 18.27 / 15.30 -> 15.50 |
(r = recovered share of the mis-coloured surface; seeds 97 / 99.)
- The price: C2 40/80 geometry cells worse than the tag's worst V + spread (F@0.5 9/10, Boundary IoU 6, studio PSNR 6,
  yardstick 5+5); kinetic record at the end 5-80x (G3 passes; still rearranging); one piece; windows 19-128 (D81 mode:
  render-dominated merit, some early stops while the colour was still falling).
- Data: output/gpu/d160/gate_H/{table.md,gate.json}, read_colour_H.jsonl, video/*_97_H/.

## D160d (running next on wip/d160, option --appearance_weight held)
- The geometric render terms are exactly the tag's (silhouette + grey shading) on the rule's lambda (PCGrad, balance).
- The colour term (the colour pictures' error, its geometry held at the window's discs, so its gradient goes only
  through the discs' albedos) is a separate fixed-weight term beside the cleanup: weight = the render weight's first
  calibrated value, held (no new constant). It lives in the physics' null space, so it is not balanced against it.
- Question: does the colour move as in D160c while the geometry and rest stay as in D160 (rule)?
- If D162's weight (the render term's excess above its floor) is adopted, the colour term would be one more render term
  whose floor is the target sample's own colour error; it could merge with D160d's treatment.
