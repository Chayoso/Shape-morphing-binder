---
name: resume-2026-10-16-morning
description: 2026-10-16 ~11:00 CDT state — overnight post-tag evidence (B1 44/44, A6 75/75, B7 done, B2 19/72), B13 26/36; 88 GB cleanup (348 G); agents launched (D145 post-tag reading, RGV on the tag); user approved D147 fixes 1–4 (loss raster, per-body inf, replay null, deformation-only damping) at 40k first
metadata:
  type: project
---

State at 2026-10-16 ~11:00 CDT; the user said "계속 진행 하자".

**Overnight, on tag freeze-2026-10-15** (outputs in `output/gpu/d145t_*`):
- B1 300k is complete: 44 JSONs (6 meshes × L/P × seeds 97/99/101, plus bunny/dragon 103/105).
- A6 ablations complete: 75 (NRD, NUC, NSS).
- B7 RG97 bunny/dragon done (dumps + counterfactuals, rgv_patch2).
- B2: 19 of 72.
- B13 300k: 26 of 36 done (`output/gpu/b13/status.txt`).
- The pre-tag RGV dragon 4K finished: `output/gpu/rgv/render7_dragon_2160_*`.

**Cleanup at 10:47.** 88 GB deleted, logged in cleanup.log; 436 → 348 G.
- d147 scene40k_* prototypes (42 G);
- d148 term dumps;
- rgv RG97 dumps and counterfactuals + the v1design viz;
- d146g window dumps.

**Agents launched:**
- D145 post-tag reader: B1 table L vs P, the comparison with the pre-tag D135 numbers, the A6 table, and the render influence. Output to `output/results_2026-10-16/d145_posttag/` and docs.
- RGV on the tag: from the B7 data, A·B2·C·D·U with every non-render term counted. Output to `output/results_2026-10-16/render_gradient_viz_tag/`.
- D147 resumed with the user's go ("1–4 다 고쳐서 40k 확인"), scene code only:
  1. the per-body loss raster covers the landings;
  2. an inf term stops only its own body;
  3. no zero-control commit on a replay null;
  4. damping on the deformation velocity only, per body, at 0.9/s (keeps L and P per body).
  - Order: tests + suite → pre-register the 40k gate → 40k → 300k s97/twin/s99 → renders to `output/results_2026-10-16/d147_throw_fixed/`.

**Still to do:** the B13 reading once 36/36 are done (its agent died; spawn a reader). B12 is not merged (wip/b12). Floater display is deferred.

**D145 post-tag reading (889585b, ~12:10 CDT; tables in `output/results_2026-10-16/d145_posttag/`).**
- **B1, L vs P:** the render cuts the yardstick silhouette error by 31–57 %, shading by 19–35 %, and display front 1−IoU by 18–33 % on all 6 meshes (seed ranges apart). It does NOT move silhouette IoU or chamfer against the independent sample, thin, centre of mass or rest.
- **L worse than P:**
  - bob net rotation +389 %;
  - largest angular momentum higher on bunny, bob and homer (the drag's P2G form);
  - early visible blobs: armadilo 14 vs 6, homer 11 vs 2;
  - homer cost +31 %.
- **Tag vs pre-tag D135:**
  - thin −16…−30 % on all 6;
  - centre-of-mass drift −53…−82 %;
  - end kinetic energy −29…−52 %;
  - render lead kept;
  - COST worse: minutes +22…+85 % (dragon 27.3 vs 18.3 min, homer 29.0 vs 15.7);
  - early apart particles up (teapot ×3.1).
- **A6:** each part earns its place.
  - relax_driven: wobble ×2 without it;
  - u_adam_carry: blobs ×2 without it, dither;
  - spacing_symmetric: bunny drift ×3.3 without it.
- **Floaters:** nothing serious.
- **Volume Σ J V0:** bunny 0.997–1.013, dragon 0.992–1.031.
- **The agent's slips:** `python3 --version` run locally once; repo_settled/__pycache__ deleted.
- **Pool queue:** the 57 B3 V items moved ahead of B2, plus a one-minute end-IoU job (logged). The 40k A6 reading is due about 13:30.

**D147 fix, 40k gate (~15:00 CDT; docs 2592131, 904853c; code wip/d147 fa0d091; server repo_d147fix).**
- The four fixes work: 0 zero-control windows; head recovery 0.6–0.74 sp against 0.93–1.57 for the D135 references; the judge passes in all 3 runs (end error 0.025–0.028); per-body momentum within 5.7e-4.
- FAIL on clean in 2 of 3 runs: body 1 sheds an ear-tip piece 2–3 s after landing, during the squash recovery (s97 13 particles at 3.35 sp; twin 10 at 4.0 sp). The same class exists in the D135 refs. The grid's elastic squash moves those particles.
- Cost +50 % (17 vs 11.4 min), caused by single-address atomics in the damping sums and their adjoints.
- **USER decision:** "끝 조각을 먼저 없애는 거랑 속도 개선을 한 번에 같이 해 줘".
  - D147 now does: the ear-tip forensic, then a definition fix in scene code (stop if the cause sits in the frozen single-body algorithm), plus the exact damping speed-up (block sums and a hand adjoint).
  - Then a new 40k gate, then 300k.

**D152 done (0a776fa).** The render's effect is real; the IoU is blind to it.
- The binary 128 px IoU is 1.8 pitches a pixel, and 72–89 % of the twin's error is the floor.
- The render's own silhouette term goes below the floor (bunny 3.7 s, dragon 10.8 s).
- Late on, the render maintains the state: a window without it undoes 67–81 %.
- PCGrad cuts only 3–10 %.
- A paper paragraph is drafted. Proposed follow-ups: mesh-exact silhouettes (no floor); late render on/off switching as the causal test.

**D153 running:** the tag profile, plus the user's question about a fixed 5–6 s horizon (quality against t from the B1 kept frames).

**END OF DAY 2026-10-16 (~14:30 server clock).** User: "일단 실험들만 돌려놓을래? 오늘 여기까지 하자". The local agents die with the session. The server chains below are detached (parent 1). Disk 362 G.
- **D147:**
  - `tmp/d147/chain21.sh`: the dump runs for the ear-tip forensic.
  - `repo_d147fast/scripts/probes/d147/chain22.sh`: the damping speed-up chain.
  - Logs: output/gpu/d147/chain21.log, chain22.log.
  - On resume: read both; do the forensic (root cause of the post-landing ear-tip shedding), the definition fix in scene code (stop if it is in the frozen single-body path), then pre-register the new 40k gate (same items, clean < 10 particles), then 300k.
- **RGV on the tag:** `tmp/rgv_tag/rgv_queue_main.sh` (idle-card renders). On resume: collect into `output/results_2026-10-16/render_gradient_viz_tag/` and check the frames.
- **B13:** `tmp/b13/b13_chain300k.sh`, 36 runs. On resume: spawn the reader (pre-registration on wip/b13 + docs ce39674).
- **D145 pool:** runs B2 / B3 / B8–B10. max_cards is 1 until restore_cards2.sh sets it to 2 at 23:59. The 40k A6 reading (`a6_40k.py`) is due once B3 V is done.
- **D153 and D152:** their GPU parts ended (no chain listed). On resume, check output/gpu/d153 and d152 (own_floor.jsonl, yard_floor.jsonl) and re-run D152's tables/figs if the floors landed.
  - D153's answer to the user's fixed 5–6 s horizon question is pending: quality against t from the B1 kept frames, and the minutes per mesh.
- **Pending user decision:** a fixed 5–6 s horizon (wait for D153's numbers). Floater display deferred.
