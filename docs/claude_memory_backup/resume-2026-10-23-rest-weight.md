---
name: resume-2026-10-23-rest-weight
description: 2026-10-23 afternoon handoff — D166 (rest under render authority) + D167 (λ weight definition) experiments queued with auto-summaries; on return read ONE file output/gpu/RENDER_REST_WEIGHT_SUMMARY.md
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-24T06:37:30.563Z
---

User 2026-10-23: "남은 문제 1, 2에 대해서 일단 실험을 최대한 많이 돌려놓자. token 다 써도 나중에 한 번에 보기만 할 수 있게".

**UPDATE 2026-10-24 01:40:**
- Codex took over the checks, went far beyond them, and stopped every queue at 01:22. D166/D167 are PARTIAL; RERUN them ourselves later (user: "나중에 다시 우리가 실험 돌려야 할 거 같아").
- Codex's work is set aside in C:\dev\Shape-morphing-binder_codex_2026-10-24\ (README there) and on hyde06 in physmorph_v2/codex_2026-10-23/ (34 GB).
- Kept: the solve.py docstring, the settle `tr.capture()` fix in repo_d166 runner.py:315 (old D166 settle_relax results are invalid), and Codex's d166_read.py hardening.
- Restored: our dissertation (125-page PDF, zip), with figures as of 10-23 14:49 and the old Fig 4.3.

**On return, read only (hyde06, /data/relcfd/chayo/physmorph_v2/):**
- output/gpu/RENDER_REST_WEIGHT_SUMMARY.md: the master file, written by aggregator job 199, with a "What to adopt" section.
- If it does not exist yet, read output/gpu/d166/auto/*/summary.md and output/gpu/d167/auto/*/summary.md, plus tmp/d166/RESUME.md.

**What is being tested:**
- D166 (problem 1, rest): settle verdict B (job 50) = rest is reached, but relief keeps only 0.29–0.43 and colour keeps 1.00.
  - First measurement: does `--settle_windows` drop u? The kept fraction ≈ 1/3 matches δF_c's share of the letters, so the "spring-back" may just be u removed.
  - Arms: u held in settle, assimilation strength, rest-requiring stop D162i, stiffness (×3/×7 = job 51).
- D167 (problem 2, weight definition): rule / held / `--render_weight_ref floor|shape`, each ± `--placement_at_cell`.
  - Run on coin / bunny / band 40k, the 6-mesh gallery and bunny + dragon 300k.
  - Criteria: keep letters/colour share AND rest (v_mean ≤ 2× twin) AND gallery shape metrics.
- A separate reader agent is finishing job 51 (stiffness) and jobs 53–55 (500k/300k ripples, band) into output/gpu/d162/auto/{51_stiff,53_*,54_*,55_*}/summary.md.

**Known before the queue (2026-10-23):**
- Held λ writes what the twin cannot: coin 0.55–0.59 / bunny 0.66–0.83 of the ceiling vs ≈0; 300k ripples 0.73 vs 0.09; 500k plaque contrast 0.68 vs 0.03; colour r 0.93 vs PSNR 21.4/16.8.
- u-off: δF_c alone writes coin 0.18–0.19 / bunny 0.57–0.62.
- 29b render-centric recipe on 6 meshes: shape ≈ same, thin worse, v_mean 5–30× → not a default.

**How to apply:** report each verdict in a few Korean lines, with the render→control→outcome summary ([[always-report-render-influence]]). An adopted fix that changes the frozen algorithm goes to the user first ([[autonomy-until-siggraph]]). Then swap it into Chapter 4's "Costs" / "A Second Mode" sections ([[render-own-contribution]]).
