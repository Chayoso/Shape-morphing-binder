---
name: material-and-external-force-phase
description: "user directive 2026-10-01 night — characterise the frozen recipe under other materials and under external forces at 40k; what exists, what was measured first, the open design decision"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-02T05:07:54.342Z
---

User (2026-10-01 ~23:35 CDT): "이제 material property에 따른 것도 봐야 할 거 같아. 그리고 외력이 주어져도 morphing이 되면서 움직여야 하고. 그 부분도 40K 기준으로 실험 돌려야 할 거 같아. 근데 이건 되려나?"

**What exists (commits 8c40b24…, repo_r33/r34 on hyde06):** run flags `--young --poisson --assim --drag` (material) and `--f_ext X Y Z`, `--floor` (floor at the source's lowest point, target moved to stand on it), `--floor_friction`; the window record carries `com` and `v_com`. All default to the frozen recipe; the objective is untouched. Queue worker `tmp/fq.sh GPU SERIES REPO` + `tmp/<series>.queue`, evaluation `tmp/m1_eval.py <series>`.

**Experiments (docs/experiments.md):** M1 = one factor at a time on bunny/dragon/armadilo/bob (E ×0.1…7, ν 0…0.45, assimilation 0…1, drag 0, 3); F1 = floor alone, floor + gravity at G = ρ g H / E = 0.02 / 0.1 / 0.5, friction, sideways acceleration without a floor. Effective dynamics at every N: ρ 864, c 13.4 wu/s, CFL 0.183 (the 300k log's 6300 / 4.97 / 0.068 is the unit-mass report, not the dynamics).

**First results (bunny, 00:05 on 2026-10-02):** every E level and ν 0/0.4, assimilation 0.25 reached (sil 0.975–0.978; softer E costs thin 11–12 % vs 9.3 %); floor alone and G = 0.02 reached; G = 0.1 and 0.5 stop after 5 and 2 windows (sil 0.87, 0.72) because the centre of mass falls (0.7 wu in 4 windows at G = 0.1: a sphere on a point contact is not held up until it flattens) and the world-fixed selection merit worsens → brake → 3 rejections.

**Open decision for the user (my recommendation, not yet agreed):** a body that moves while morphing needs the objective in the body's frame (target and cameras carried by the centre of mass; settling terms on the velocity relative to it). Internal stress cannot change the centre of mass, so this is a change of definition (Galilean invariance), a no-op on the current gallery. Rotation and contact-rich cases are beyond that. Do not implement before the user agrees: the formulation is frozen ([[formulation-stop-rule]]).

**How to apply:** report M1/F1 with the render influence (λ of window 0, g_share) per level; keep these as characterisation, not adoption experiments; see [[no-parameter-only-fixes]] before proposing any fix for the gravity failure.

**Final M1 (2026-10-02 00:55, 56 runs):** 47 of 48 non-base runs reach the target, no guard; stiffer = fewer uncovered thin parts (E × 7 best on every mesh), softer = worse thin; assimilation 0 still morphs; ν 0.45 only costs windows; the armadilo at E × 0.3 froze on the parked `domain` ejection. λ of window 0 follows E (calibrated on the physics gradient); g_share 0.78–0.92 throughout. All numbers in docs/experiments.md (M1, F1).
