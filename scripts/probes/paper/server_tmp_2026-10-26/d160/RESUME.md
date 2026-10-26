# D160 RESUME (agent D160; updated 2026-10-22 00:25 CDT)

DONE. Nothing running, no card held. Read: docs/experiments.md "D160 read" (settled-base 2981079b).
Verdict: FAIL on C1 (r 0.01-0.09 on every rearrangement scene; pass 0.5); C2 holds (5/80); C3, C4 miss by one run each.
Gates passed (suite 446 exit 0 after 209ba6d; white = rt1 bit for bit; twin blind to the paint).
Code: wip/d160 209ba6d (origin). Results: output/gpu/d160/ (runs40 frames12 kept, 1.3 GB; gate/, video/ mp4 + end.png,
read_colour.jsonl, pieces.jsonl, d157_L.jsonl). Local: output/results_2026-10-21/d160_appearance/.

Measured cause: the colour is placed while the body flows (gain largest at 0.6-0.8 s on turn / dragon, half lost at
arrival); after arrival lambda falls 120x with the physics gradient, the window moves ~0.008 pitch, the front stops.
Not locality alone (the 3.7-pitch scene fails too). Same pattern as D161.

Proposed next (not started; needs the coordinator's go):
1. state-level split of the colour push at the front (transport/proximity cancel vs step pace), CPU/card, minutes;
2. D160b = D160 on D155's moving target (wip/d155 ramp option), 40k, the same five scenes and criteria;
3. if still stalled: DROT-style OT colour term on (pixel position, colour).
Re-read commands: tmp/d160/d160_read.py OUT JOBS ; tmp/d160/d160_gate.py OUT_DIR (PYTHONPATH=repo_d160, CPU).
Note: one stray local `python3 -` call happened ~17:30 on 10-21 (no effect); not repeated.

## 2026-10-22 00:51 CDT: step 1 read, D160c running
- Step 1 (docs e75895f9): the colour push is 1/5-1/25 of every other force at the front; it moves material only while
  the body flows; after arrival the fixed-weight cleanup pushes back along the needed direction; stop rule / transport
  not the limiter. Files: output/gpu/d160/limiter_A.json, step1/ (split.jsonl, dumps ~350 MB: delete after the read).
- D160c (pre-registered 00:49): PID 3105260 `bash tmp/d160/d160c_chain.sh` -- waits for a card (3/1/0, 10 idle min),
  runs *_H{97,99} (--appearance S --lambda_ema 0), card readers, read_colour_H.jsonl, videos video/*_97_H, releases the
  card, then CPU pieces / gate reader / D157 reader and `d160_gate.py output/gpu/d160/gate_H H`; status lines "[D160c]".
  Re-run if killed: same command (finished runs are skipped).

## 2026-10-22 02:45 CDT: D160c read, nothing running
- D160c (docs 1ef311f6): C1 PASS (band r 0.91/0.93, turn 0.63/0.63), gate FAIL on C2 (40/80 cells), C3 (kinetic 5-80x;
  G3 passes), C4 (1 piece). The render-weight coupling is the colour limiter; holding the whole weight (pre-D81) breaks
  rest and the exact-mesh surface.
- Next = D160d, waits for the user: the geometric render terms on the rule's lambda, the colour term held at its
  first-window weight. Not started.
- No card held; the D160c chain (3105260) has finished.

## 2026-10-22 02:53 CDT: D160d running
- D160d pre-registered (docs 97426e5a). Code wip/d160 d3aa85e9 (--appearance_weight held, default rule), deployed
  repo_d160 / repo_d160s. Note for D162: tmp/d160/for_d162.md.
- PID 383480 (restarted 03:51 after the V4 log fix 963cd2b1; try 1 4170064 stopped at V1, try 2 138006 at V4 log check) `bash tmp/d160/d160d_chain.sh` -- waits for a card (3/1/0, 10 idle min); V gates (gates_d/), runs *_D{97,99},
  card readers, read_colour_D.jsonl, surfdisp.jsonl (D/H/L vs P + floors), videos video/*_97_D, releases the card,
  then CPU pieces / gate reader / D157 reader, `d160_gate.py output/gpu/d160/gate_D D`. Status lines "[D160d]".
  Re-run if killed: same command (finished runs are skipped; the gates rerun).

## Checkpoint 2026-10-22 (local shutdown)
- Running: PID 383480 `bash tmp/d160/d160d_chain.sh` (setsid, own session; its CPU readers and the gate run inside it
  after the card is released, no separate after-script). Ends with "[D160d] gate read: ..." and "[D160d] done" in
  output/gpu/d160/status.txt; results gate_D/, surfdisp.jsonl, read_colour_D.jsonl, video/*_97_D/.
- Code wip/d160 963cd2b1 (origin), deployed repo_d160 / repo_d160s. Docs: D160d pre-registration 97426e5a.
- Next on resume: read gate_D/table.md + surfdisp.jsonl (D vs L / H vs P, floors), write "D160d read" in
  docs/experiments.md (note the code 963cd2b1 = d3aa85e9 + two non-path fixes: the stand-in config default, the printed
  line), copy to local results d160_appearance/d160d/, report in Korean. Re-run if killed: same command.

## 2026-10-22 10:30 CDT: D160d read, nothing running
- D160d (docs 09757a62): C1 PASS on all five (r 0.67-1.00), gate FAIL on C2 (12/80, mostly F@0.5; turn), C3 (7/10 runs
  above the kinetic bound, those whose colour still flows at the end), C4 (one 10-particle piece, dragon, as the tag's).
  The surface reading (surfdisp.jsonl): circulation under a surface moving 1.1-1.4x the rule arm's (turn/dragon 1.4-1.9x).
- Next: the colour's weight following its own excess above the paint's floor = D162's definition; D160d is its
  reference. Twins *_P* and D160d *_D* frames12 kept for D162; L/H/step1 frames deleted (cleanup.log).
- No card held, no process running.
