---
name: resume-2026-10-15-afternoon
description: "14:30 CDT handoff before a local shutdown. Every job runs detached on hyde06; on resume, read the D151 gate, finish the RGV renderer edit, merge B13, collect B12/B14/B15"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-15T19:29:05.167Z
---

This is the state at 2026-10-15 14:30 CDT (server clock), just before the user shut down the local machine. Every agent dies with the local machine. The server jobs below are nohup'd and keep running. Disk: 299 G of the 500 G limit.

**D151 `--layer_link` gate. Top priority: read it tonight; the user decides before the 10/18 tag.** See [[freeze-plan-2026-10-18]].
- Code: wip/d151, dd3d959 + bc1feac, on refreeze-cand 169f251. Local worktree: C:\dev\smb_d151. Server copy: repo_d151. Suite: 418 passed.
- Pre-registered in 555c42e; launch notes in 7e5b644.
- Server jobs under /data/relcfd/chayo/physmorph_v2:
  - `tmp/d151/d151_chain.sh dragon` on GPU 1: NL, then NP.
  - `d151_chain.sh bunny`, waiting for GPU 3. CDX's render (tmp/d148/q_cdx.sh) must finish first.
  - `tmp/d151/d151_post.sh`: the CPU readers, visible_apart and why_floating.
  - `tmp/d151/d151_base_render.sh`: D149's SL/SP side renders, which run last.
  - Logs: tmp/d151/*.out.
- On resume:
  1. Check the chains finished.
  2. Run `scripts/probes/d151/d151_gate_read.py` and judge each item as pre-registered.
  3. Copy the table and the side-by-sides (base | layer_link, dragon and bunny side) to output/results_2026-10-15/d151_gate/.
  4. Commit the reading to settled-base.
  5. Report to the user in Korean.

**RGV (layout fix + labelling re-render).**
- The agent was mid-rewrite of the four-panel renderer (scripts/render_gradient_viz.py: body_arrows, arrow alpha) and its README when shut down.
- Run `git status` / `git diff` on resume. The edit may be partial.
- Server: `tmp/rgv/rgv_queue3.sh`, status in output/gpu/rgv/status.txt (prep/proof dragon rc 0; hooktest2 waits for a card idle 10+ min).
- Queue: dragon 1080p, bunny 1080p, bunny 4K side/front, dragon 4K.
- The fixed dragon 1080p goes to the user for LinkedIn.

**B13 (force-control ablation).**
- `tmp/b13/b13_chain40k.sh` is running: CUDA tests, deterministic proof, suite, 40k smoke.
- When it passes, merge wip/b13 (49d8706) into refreeze-cand.
- The 300k production runs (b13_chain300k.sh) wait for the tag.

**B12 (Large Steps / FlexiCubes).**
- Server: `baselines/b12/scripts/b12_after.sh` + `b12_queue.sh bunny`; log baselines/b12/logs/after.log.
- Done: bunny, dragon, armadilo; homer was next, then the bunny re-run.
- Large Steps:
  - armadilo 0.982–0.985, ours 0.985–0.987; 3–7 self-intersecting frames per run;
  - dragon: genus 0 in every run, so it cannot open the loops; its loss ends 6–10× the bunny's;
  - wall time 19–53 s against our 7–18 min.

**D145 pools (r1–r3) on GPUs 0/2.**
- `tmp/d145/main.queue`, 494 lines: B14 ISD/4Deform (homer running), B15 NISE, Buonomo seeds 2–3.
- The D145 agent itself is finished. The post-tag queue (posttag_local.sh TAG) waits for the 10/18 tag.

**B5 Buonomo:** all six pairs are done. Seeds 2–3 run detached via master2.sh.
