---
name: gallery-single-run-spread
description: "40k gallery, one run a mesh: C stops early while moving in 3 of 4 runs of any code, nefertiti sometimes; thin share differs by up to 6 points between two runs of the same code; read a gallery against TWO runs of the baseline"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-03T09:12:48.406Z
---

Measured 2026-10-03 (D25, D27, D29, D31; docs/experiments.md): two runs of the same code (repo_r35b, seed 97) on the 19-mesh 40k gallery differ by a median 0.0005 in silhouette IoU (max 0.0022) and 0.9 point in thin share, but single meshes swing far more: bimba's thin share 8.2 vs 14.3; C runs 67 windows in one run and stops after 16 windows still moving (last kinetic record 3e-2, silhouette 0.974–0.9765) in three others, whatever the code; nefertiti's old-code run once stopped after 16 windows at 1e-2.

**Why:** in D25 and D28 I read C's early stop and bimba's thin share as effects of the change under test; the baseline's second run showed the same. My reading, not the user's.

**How to apply:** judge a gallery against two runs of the baseline (`tmp/d29_eval.py` on the server does this: D19's 96-px arm and D27's old arm) and report a mesh as changed only if it leaves their range; never attribute C's early stop or a 3–4 point thin swing on one mesh to a change from one run. Related: [[generalise-beyond-bunny]], [[floaters-and-15min-until-solved]].
