---
name: physical-detail-not-display-trick
description: "User directive 2026-10-07 00:50 CDT: the fine surface relief must come from the physics; keep finding methods until it does; the D111/D115 display-only detail layer is \"too much of a trick\" for the paper"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-09T08:39:14.278Z
---

User, 2026-10-07 (after D122 was launched): "그러한 결과가 나올 때 까지 방법 찾아서 해 줘. 후처리로 가기엔 지금...너무 trick 같잖아".

**What it means:** the sub-pitch relief now shown by the display-only detail layer (D111/D115: per-surfel normal offset fitted to a 1.5 M target reference, `child_detail_morph.py +fit=1`) is not acceptable as the paper's answer. The physics (particles + base surfels) must carry the relief. Keep running pre-registered experiments until the relief bands (D105's `ag2_bands.py` measure at 2.7 / 5.4 base pitches) rise to the dense sample's level, with the twin margins, momentum, thin coverage, roughness and runtime held.

**Ladder (in order, each pre-registered, each generalised on the 40k gallery before adoption):**
1. D122: surface-adaptive sampling — REFUTED 2026-10-07 (relief −0.05…−0.08, yardstick +80–109 %; a two-density sample breaks the transport's uniform-density premise). KEY FINDING: the same uniform run read through a finer field gives 0.60–0.62 at 5.4 pitches vs 0.55 through the base field (radius 3 pitches, D59): the particles already hold the relief; the exterior field's radius hides it, and the loss's exterior uses the same field.
1b. D123 DONE 2026-10-07 09:00 (commits 28b5f97, 63550cc, 1b70ea3): the exterior field closes down to r = 2.5 pitches (offset ∝ r, 0.667); read at 2.5 the SAME D105 particles carry relief 0.37–0.42 / 0.61–0.63 at 2.7 / 5.4 pitches (vs 0.28 / 0.55 at r = 3; the 300k sample's cap at 2.5: 0.74 at 5.4) — no fit, no target. The loss's field at 2.5 (stage 2, 300k pairs) neither adds nor costs relief; end as D105, bunny twin lead at two thirds, momentum mixed. `--exterior_radius` A/B flag, default 3; adoption (display/evaluation at 2.5; loss stays 3) = the user's call. Remaining gap to a 1.5 M sample = N (Nyquist).
2. If F = 2/4 is limited by the interior (ppc, ears starving): grow N only on the surface (surface shell at 1.5 M-equivalent pitch, interior coarse).
3. If the cell limits it (stress control is cell-scale, D109): grid refinement near the surface / smaller dx with the cost measured.
4. The relaxation's relief reference (D105) and u already act sub-cell: check they are not what caps the relief (D12–D14 found the relaxation removed fine relief before D105).

**UPDATE 2026-10-09 (user):** "post process … SIGGRAPH에서 이걸 우리가 대응 가능한 정도라면 사용하는 방향으로 가자" — the D111/D115 detail layer WILL be used in the paper's videos if defensible, framed as display-only target-detail transfer onto the arrived surface (multiresolution / displacement-map style), with: all metrics on base discs; with/without side-by-sides in the supplementary; the same treatment (or none) for baselines in comparisons; D115's arrival rule so no target detail appears on material in transit; an explicit sentence that the fine relief is the target's, not simulated.

**How to apply:** do not present the display layer as the method's detail; the paper's videos should be base-surfel renders once the physics carries the relief, or the layer must be explicitly labelled as display-only. Related: [[algorithm-before-n]] (the user postponed 1.5 M overall: surface-only growth is the compromise), [[render-look-default]], [[surface-resolution-findings-2026-10-02]].
