---
name: d137-cell-from-n
description: "2026-10-09 — crevice beads / neck web in the volume-conserving body = MPM cell too coarse for the target's gaps; user approved cell-from-N (dx ∝ N^-1/3 above 40k) as D137, beast's −0.0007 accepted as recorded cost"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T18:34:01.460Z
---

D137 (2026-10-09): the visible beads in the dragon's hind-leg crevice at 3.6–4 s (D135 side view, 7th row of the strip) and the late neck–body opening come from one cause. The MPM cell was fixed from the source shape (diag/26 ≈ 6 pitches at 300k, 183 ppc), and the cubic kernel reaches ~12 pitches, so material in gaps 10–27 pitches wide shares nodes with both sides and can only tear apart into beads. The old path hid it by changing volume. The cause is not the render (the twin beads equally) and not the transport assignment (91–98 % go to the correct side).

Test with a halved cell (`--cell_diag 52`, ~23 ppc): the gaps open 2–3 s earlier with no beads, and there is no time or memory cost, because the grid (44³ → 83³) is small next to 300k × 64 stencil work. Side effect: more apart particles in windows 5–7.

**User decisions 2026-10-09:**
- Pre-register and run D137: above 40k, dx = (diag/26)·(40000/N)^(1/3). The u gate and the thin set stay on the shape's cell; 40k is bit-for-bit unchanged.
- The gate is run at 300k: bunny/dragon arm+twin, plus teapot/bob/armadillo/homer.
- Beast's twin lead (3-seed mean −0.00071 against the rule's −0.0005) is accepted as a recorded cost.
- Pass → freeze = stash@{0} defaults + D137, then tag. Fail → report; the two-field split (Nairn/CPIC) needs the user's go.

**Outcome 2026-10-09 evening:** the gate failed by the letter, then each miss was explained:
- At rest: within D135's own two-run spread.
- Armadillo floaters: the same arrival population two windows earlier. Lower on all 6 meshes at matched phases; no visible piece.
- Bob: a real sub-pitch inward rim shift (silhouette +49 %, thin 8.2 vs 5.7 %). The display surface shifts inward on all 6 meshes; only bob gets worse.
- The dragon crevice now tears at 1.9–2.5 s (a ring of holes, a strand) instead of beading at 3.6–5 s.

USER DECISION: adopt D137 and freeze (tag freeze-2026-10-09). Bob's shift and the 1.9–2.5 s remnant are accepted costs. Removing the remnant would need the two-field velocity split, a new mechanism the user did not pick.

**REVERSED the same night (~22:30 CDT):**
- The user disliked the frozen bunny's 0–1 s. The face and ears are "printed" as wrinkled sheets on the sphere's front, instead of D105's whole-body rise with the ears growing like a tongue. The user also said "솔직히 트레젝토리 자체는 dragon도 D135가 좋은 거 같아보여".
- D139 proved the cell is the cause. Same frozen code, only `--no-cell_follows_n` → D135's rise.
- USER DECISION: the default goes back to the shape cell (D135). `--cell_follows_n` stays as an option. Re-freeze as tag `freeze-2026-10-09b`, which supersedes `freeze-2026-10-09`.
- The trade-off for the paper: the cubic kernel low-passes velocity over ~2 cells. The coarse cell gives a coherent rise but tears late at gaps narrower than ~2 cells. The fine cell opens the gaps but prints surface features ahead of the bulk.
- Reviewer point: ~183 ppc at 300k. Answer it later with a cell-size comparison (26/36/52) run as evidence, not as tuning.

**Next, the user's ask (2026-10-09 ~22:30):** "D135로 일단 알고리즘 자체는 fix. 근데 저기서 살짝 떠다니는 disc를 없앨 수 있을지 고안 해 보자".
- The runtime work goes on with D135 as the base.
- D140 = inventory of D135's floating/bead discs. Classify each as (i) physically apart particles or (ii) a thin sheet the surface field breaks into blobs; in the D135 crevice it was 18 of 56 (i).
- Then papers. For (i): multi-field contact MPM (Bardenhagen, Nairn, CPIC, Homel–Herbold, CD-MPM). For (ii): anisotropic reconstruction (Yu–Turk) and thin sheets (Ando).
- Then ONE definition change per class, proposed and only run on the user's go.
- Hard constraint: D135's trajectory (0–1.5 s look, coarse cell) must not change.

**2026-10-14, USER DECISION: D148 gap split.** The user saw the remaining dragon crevice foam (3.3–4.2 s) and the two detached particles (5.1–5.4 s) in the final zbs side video and chose "틈에 두 속도장 적용 시도".
- The method reuses D147's per-body multi-field contact, with side labels taken from the transport's target side (D137: 91–98 % correct) or Homel–Herbold field-gradient partitioning, and splits only at gap nodes.
- Behind `--gap_split`, default off. Gate at 300k plus the gallery; the D135 0–1.5 s rise must hold.
- On pass: re-freeze in the 11-02–06 slot. Agent launched 2026-10-14 ~09:40 CDT.

**D148 verdict, 2026-10-14 (704871f): FAILED items 1, 2, 4 and 5.**
- The 3–6 s crevice is clean on all 3 dragon runs, but:
  - new 1–3-particle beads at the early crevice opening (1.5–2.25 s), the loop (2.4 s) and the upper hole (10–14 s);
  - the bunny 0–1.5 s is out of band;
  - floating pieces slightly over;
  - cost +43 %.
- Cause: coin-toss labels within one cell of the cut, because the transport is blurred to about 6 pitches; mislabelled particles are pulled out into the gap. The split also opens the crevice early (1.3–1.6 s vs 3.3 s).
- USER: "D148b를 시도하되, 다른 접근 방법도 봐봐". In progress:
  - an exact cost speed-up (hand deposit adjoint, early exit);
  - D148b (loss-grid labels);
  - a survey of alternatives: (a) soft sides from the plan, (b) Homel–Herbold geometric partition, (c) thin-only split, (d) others;
  - D148c pre-registered from the best of these, then a 40k and 300k gate.
- Also fixed on the way: the memory leak c38951c; traj.py back to 413 lines (c5db4ac).

**User priorities, 2026-10-14 afternoon:** "속도랑 지금 저 둥둥 떠다니는 거 잡자".
- First the D148 cost speed-up, then the mid-run detached beads (D148b/c, soft sides first).
- The dragon's convergence speed comes later ("나중에 수렴속도 좀 더 빠르게", parked).
- FALLBACK if D148 can't be fixed: the deliverable state = tag deliverable-2026-10-13, i.e. D135 sim (freeze-2026-10-09b) plus the D144 display-only Solenthaler field.

**Why:** the user wants the visible defect gone before the freeze, and asked about cost ("격자 또 줄이면 cost 올라가는 거 아니냐"). The measured answer was no cost.

**How to apply:** a cell from N is consistent with [[all-means-until-solved]] ("resolution comes with N"). That rule forbids a dx cut at fixed N, not a dx that follows N. See [[pipeline-freeze-2026-10-07]] and [[runtime-10min-goal]].

**2026-10-14 ~16:40 CDT: D148 label hunt CLOSED at WL (walls).**
- D148d (nearest wall) failed its CPU check (b, c). D148e (wall carried through the stencil) failed (c, d): its single-field fallback covers 86–100 % of the would-split mass.
- The stop rule I set holds: no further label variants.
- WG99: fewer beads than A, not zero; cost passes.
- Running now: the D146b + WL gate on repo_d148rf (RW, RWT, RWB, RWBT), plus RWX (term dump for the loop beads).
- Next: the user chooses WL vs fallback (D146b alone + D144 display) from side-by-side videos.
