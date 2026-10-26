# Render rest + weight definition (D166 / D167): master summary

Written 2026-10-23 17:00 CDT; 7 of 29 job summaries present (missing: 67_stiff3, 100_twinP, 101_floor, 102_shape, 103_held, 104_rule, 110_gal_twinP, 111_gal_floor, 112_gal_shape, 113_gal_held, 114_gal_rule, 116_cross, 120_300k_T, 121_300k_F, 122_300k_S, 123_300k_H, 124_300k_R, 125_300k_TP, 126_300k_FP, 127_300k_SP, 128_300k_HP, 129_300k_RP).

## What to adopt

- D166 (rest under authority): no arm passes (authority + rest <= 2x twin + kept >= 0.8); the best is UlA10 = dF_c + layer u + render loss (full), held + settle 30, u held, assim 1.0 (kept min 0.63, rest max 0.0x).
  - Ul: kept min 0.40, rest max 0.0x, relief min 0.16, colour min 0.92
  - Rx: kept min 0.34, rest max 0.0x, relief min 0.19, colour min 0.93
  - A10: kept min 0.25, rest max 0.0x, relief min 0.10, colour min 0.93
  - A08: kept min 0.31, rest max 0.0x, relief min 0.11, colour min -
  - UlA10: kept min 0.63, rest max 0.0x, relief min 0.35, colour min 0.93
  - RxA10: kept min 0.35, rest max 0.0x, relief min 0.14, colour min -
  - A025: kept min 0.36, rest max 0.0x, relief min 0.19, colour min -

Criteria: authority = relief share >= 0.4 and colour r >= 0.8 on every 40k run (held: coin 0.55-0.59, bunny 0.66-0.83, colour 0.93); rest = end v_mean <= 2x twin; kept >= 0.8 after 30 settle windows; gallery holds = sil_iou / chamfer / thin within 0.003 / 0.002 / 0.02 of the tag on all 6 meshes; 300k against the tag rule R. Arms: dF_c + layer u + render loss (full) vs dF_c + layer u (twin).

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/60_gate/summary.md -->
# D166 V3 gate: repo_d166's defaults bit-SAME

| check | result |
|---|---|
| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | frames12 arrays 4, metrics 37, history fields compared 556: SAME |
| settle defaults vs repo_d162hs (coin, held, settle 2) | frames12 arrays 4, metrics 37, history fields compared 570: SAME |

VERDICT: PASS (the new options --settle_u / --settle_relax are off by default)

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/61_ulast/summary.md -->
# D166 (a1): held + settle 30, u held at its last committed value (--settle_u last)

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| Ul | coin | 0.40 / 0.41 | 0.33 / 0.16 | 0.82 / 0.40 | 0.0 / 0.0x | 0.9767 / 0.9749 | 0.1087 / 0.1091 | 0.197 / 0.140 |
| Ul | bunny | 0.57 / 0.76 | 0.65 / 1.09 | 1.13 / 1.44 | 0.0 / 0.0x | 0.9684 / 0.9653 | 0.1109 / 0.1107 | 0.105 / 0.144 |
| Ul | band | 0.93 / 0.95 | 0.92 / 0.96 | 0.99 / 1.01 | 0.0 / 0.0x | 0.9646 / 0.9682 | 0.1110 / 0.1107 | 0.134 / 0.111 |

Arms: Ul = dF_c + layer u + render loss (full), held weight + settle 30, u held at its last value (vs dF_c + layer u (twin)).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: Ul INCOMPLETE (authority no min 0.16/0.92, rest ok max 0.0x, kept min 0.40)

Incomplete evidence:
- Ul: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- Ul: bunny/97: exact stop raw frame not sampled; interpolation is diagnostic only
- Ul: bunny/99: exact stop raw frame not sampled; interpolation is diagnostic only
- Ul: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/62_relaxoff/summary.md -->
# D166 (a2): held + settle 30, no layer relaxation in the settle windows (--no-settle_relax)

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| Rx | coin | 0.59 / 0.40 | 0.20 / 0.19 | 0.34 / 0.47 | 0.0 / 0.0x | 0.9786 / 0.9799 | 0.1092 / 0.1088 | 0.023 / 0.082 |
| Rx | bunny | 0.63 / 0.76 | 0.27 / 0.30 | 0.43 / 0.39 | 0.0 / 0.0x | 0.9760 / 0.9714 | 0.1120 / 0.1120 | 0.038 / 0.049 |
| Rx | band | 0.93 / 0.94 | 0.93 / 0.93 | 1.00 / 0.99 | 0.0 / 0.0x | 0.9704 / 0.9737 | 0.1117 / 0.1116 | 0.075 / 0.043 |

Arms: Rx = dF_c + layer u + render loss (full), held weight + settle 30, no layer relaxation in the settle (vs dF_c + layer u (twin)).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: Rx INCOMPLETE (authority no min 0.19/0.93, rest ok max 0.0x, kept min 0.34)

Incomplete evidence:
- Rx: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- Rx: coin/99: exact stop raw frame not sampled; interpolation is diagnostic only
- Rx: bunny/97: exact stop raw frame not sampled; interpolation is diagnostic only
- Rx: bunny/99: exact stop raw frame not sampled; interpolation is diagnostic only
- Rx: band/97: exact stop raw frame not sampled; interpolation is diagnostic only
- Rx: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/63_assim10/summary.md -->
# D166 (b): held + settle 30, plastic assimilation 1.0 at every commit (default 0.5), against its own twin

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| A10 | coin | 0.40 / 0.39 | 0.10 / 0.11 | 0.25 / 0.29 | 0.0 / 0.0x | 0.9778 / 0.9769 | 0.1090 / 0.1090 | 0.129 / 0.088 |
| A10 | bunny | 0.65 / 0.68 | 0.19 / 0.33 | 0.30 / 0.48 | 0.0 / 0.0x | 0.9768 / 0.9741 | 0.1116 / 0.1116 | 0.042 / 0.050 |
| A10 | band | 0.94 / - | 0.93 / - | 0.99 / - | 0.0 / 0.0x | 0.9747 / 0.9725 | 0.1111 / 0.1112 | 0.031 / 0.053 |

Arms: A10 = dF_c + layer u + render loss (full), held + settle 30, assim 1.0 (vs dF_c + layer u (twin), twin, assim 1.0).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: A10 INCOMPLETE (authority no min 0.10/0.93, rest ok max 0.0x, kept min 0.25)

Incomplete evidence:
- A10: bunny/99: exact stop raw frame not sampled; interpolation is diagnostic only
- A10: band/99: missing/empty/invalid metric cache
- A10: band/99: end authority unavailable
- A10: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/64_assim08/summary.md -->
# D166 (b): held + settle 30, assimilation 0.8 (default 0.5), against its own twin

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| A08 | coin | 0.36 / 0.39 | 0.11 / 0.15 | 0.31 / 0.39 | 0.0 / 0.0x | 0.9814 / 0.9801 | 0.1090 / 0.1088 | 0.038 / 0.094 |
| A08 | bunny | 0.59 / 0.65 | 0.26 / 0.35 | 0.44 / 0.53 | 0.0 / 0.0x | 0.9762 / 0.9755 | 0.1118 / 0.1116 | 0.035 / 0.046 |
| A08 | band | - / - | - / - | - / - | 0.0 / 0.0x | 0.9756 / 0.9724 | 0.1112 / 0.1116 | 0.052 / 0.050 |

Arms: A08 = dF_c + layer u + render loss (full), held + settle 30, assim 0.8 (vs dF_c + layer u (twin), twin, assim 0.8).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: A08 INCOMPLETE (authority no min 0.11/-, rest ok max 0.0x, kept min 0.31)

Incomplete evidence:
- A08: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A08: coin/99: exact stop raw frame not sampled; interpolation is diagnostic only
- A08: bunny/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A08: bunny/99: exact stop raw frame not sampled; interpolation is diagnostic only
- A08: band/97: missing/empty/invalid metric cache
- A08: band/97: end authority unavailable
- A08: band/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A08: band/99: missing/empty/invalid metric cache
- A08: band/99: end authority unavailable
- A08: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/65_cross_a10/summary.md -->
# D166 (a x b): u held / no relaxation in the settle, each with assimilation 1.0

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| UlA10 | coin | 0.47 / 0.40 | 0.43 / 0.35 | 0.92 / 0.86 | 0.0 / 0.0x | 0.9775 / 0.9710 | 0.1088 / 0.1087 | 0.174 / 0.205 |
| UlA10 | bunny | 0.73 / 0.62 | 0.73 / 0.39 | 1.00 / 0.63 | 0.0 / 0.0x | 0.9716 / 0.9749 | 0.1107 / 0.1110 | 0.062 / 0.073 |
| UlA10 | band | 0.93 / - | 0.93 / - | 1.00 / - | 0.0 / 0.0x | 0.9683 / 0.9723 | 0.1102 / 0.1106 | 0.120 / 0.075 |
| RxA10 | coin | 0.55 / 0.31 | 0.19 / 0.14 | 0.35 / 0.46 | 0.0 / 0.0x | 0.9776 / 0.9826 | 0.1091 / 0.1091 | 0.152 / 0.053 |
| RxA10 | bunny | 0.65 / 0.52 | 0.35 / 0.22 | 0.54 / 0.42 | 0.0 / 0.0x | 0.9737 / 0.9727 | 0.1118 / 0.1120 | 0.046 / 0.062 |
| RxA10 | band | - / - | - / - | - / - | 0.0 / 0.0x | 0.9744 / 0.9727 | 0.1117 / 0.1115 | 0.056 / 0.050 |

Arms: UlA10 = dF_c + layer u + render loss (full), held + settle 30, u held, assim 1.0 (vs dF_c + layer u (twin), twin, assim 1.0); RxA10 = dF_c + layer u + render loss (full), held + settle 30, no relaxation in the settle, assim 1.0 (vs dF_c + layer u (twin), twin, assim 1.0).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: UlA10 INCOMPLETE (authority no min 0.35/0.93, rest ok max 0.0x, kept min 0.63); RxA10 INCOMPLETE (authority no min 0.14/-, rest ok max 0.0x, kept min 0.35)

Incomplete evidence:
- UlA10: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- UlA10: coin/99: exact stop raw frame not sampled; interpolation is diagnostic only
- UlA10: band/97: exact stop raw frame not sampled; interpolation is diagnostic only
- UlA10: band/99: missing/empty/invalid metric cache
- UlA10: band/99: end authority unavailable
- UlA10: band/99: exact stop raw frame not sampled; interpolation is diagnostic only
- RxA10: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- RxA10: coin/99: exact stop raw frame not sampled; interpolation is diagnostic only
- RxA10: bunny/97: exact stop raw frame not sampled; interpolation is diagnostic only
- RxA10: band/97: missing/empty/invalid metric cache
- RxA10: band/97: end authority unavailable
- RxA10: band/97: exact stop raw frame not sampled; interpolation is diagnostic only
- RxA10: band/99: missing/empty/invalid metric cache
- RxA10: band/99: end authority unavailable
- RxA10: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d166/auto/66_assim025/summary.md -->
# D166 (b): held + settle 30, assimilation 0.25 (the direction check), against its own twin

| arm | scene | share stop 97/99 | share end | kept | v end / twin | sil_iou | chamfer | thin |
|---|---|---|---|---|---|---|---|---|
| A025 | coin | 0.50 / 0.55 | 0.19 / 0.20 | 0.38 / 0.36 | 0.0 / 0.0x | 0.9787 / 0.9816 | 0.1090 / 0.1092 | 0.106 / 0.029 |
| A025 | bunny | 0.67 / 0.70 | 0.34 / 0.25 | 0.50 / 0.36 | 0.0 / 0.0x | 0.9723 / 0.9758 | 0.1118 / 0.1116 | 0.055 / 0.048 |
| A025 | band | - / - | - / - | - / - | 0.0 / 0.0x | 0.9739 / 0.9735 | 0.1114 / 0.1112 | 0.048 / 0.052 |

Arms: A025 = dF_c + layer u + render loss (full), held + settle 30, assim 0.25 (vs dF_c + layer u (twin), twin, assim 0.25).
Share = relief amplitude / the target's own (D161 reader; coin, bunny), colour r (D160 reader; band); stop = the last optimised committed window at raw=frame_end-1, end = delivered at raw=deliver_n-1. Stop/kept values are diagnostic interpolations when the exact raw frame was not sampled; these cannot pass retention. Missing rows, nonfinite values, failed/missing exit status, or mismatched endpoint counts yield INCOMPLETE. kept = end / stop. Criteria: authority = relief >= 0.4 and colour r >= 0.8 on every run; rest = end v_mean <= 2x twin; kept >= 0.8 (settled arms). Ref job 50 (held + settle, u zero): kept 0.29-0.43 relief, 1.00 colour.

VERDICT: A025 INCOMPLETE (authority no min 0.19/-, rest ok max 0.0x, kept min 0.36)

Incomplete evidence:
- A025: coin/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A025: bunny/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A025: bunny/99: exact stop raw frame not sampled; interpolation is diagnostic only
- A025: band/97: missing/empty/invalid metric cache
- A025: band/97: end authority unavailable
- A025: band/97: exact stop raw frame not sampled; interpolation is diagnostic only
- A025: band/99: missing/empty/invalid metric cache
- A025: band/99: end authority unavailable
- A025: band/99: exact stop raw frame not sampled; interpolation is diagnostic only

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d162/auto/51_stiff/summary.md -->
# Job 29: stiffer material under the held weight (E x3, x7; D161c is E x1: coin 0.47-0.53, bunny 0.54-0.58)

- coin E x3: relief share at the stop 0.39, settled end 0.20 (corr 0.33); end v_mean 1.51e-05 vs twin 3.33e-03 (0.0x); chamfer 0.1094
- coin E x7: relief share at the stop 0.29, settled end 0.07 (corr 0.18); end v_mean 2.31e-05 vs twin 3.33e-03 (0.0x); chamfer 0.1103
- bunny E x3: relief share at the stop 0.69, settled end 0.39 (corr 0.14); end v_mean 3.30e-05 vs twin 6.57e-03 (0.0x); chamfer 0.1135
- bunny E x7: relief share at the stop 0.66, settled end 0.32 (corr 0.11); end v_mean 6.40e-05 vs twin 6.57e-03 (0.0x); chamfer 0.1143

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d162/auto/54_band500k/summary.md -->
# Job 54: 500k colour band, D160d (full) vs twin (dF_c + layer u), seed 97; runs /data/relcfd/chayo/physmorph_v2/output/gpu/d162k/runs, read /data/relcfd/chayo/physmorph_v2/output/gpu/d162k/after/read_colour.jsonl
- mis-coloured share: full 0.005, twin 0.096, floor 0.002: r = 0.96 (300k job 55: 0.97)
- held-out colour PSNR full 14.72 vs twin 13.33 (LPIPS 0.218 vs 0.228); silIoU 0.9903 / 0.9898, chamfer 0.0463 / 0.0463
- rest: end v_mean full 1.66e-03 / twin 8.43e-04 (2.0x)
- video: none rendered
VERDICT: the colour carries to 500k (r 0.96, +1.4 dB held-out) at 2.0x the twin end speed

---
<!-- /data/relcfd/chayo/physmorph_v2/output/gpu/d162/auto/55_ripband300k/summary.md -->
# Job 55: 300k ripples (H+C vs twin T) + 300k band (D160d vs twin), seed 97; runs /data/relcfd/chayo/physmorph_v2/output/gpu/d162f/runs
- ripples: ridge share of the target own full 0.73 (corr 0.42) vs twin 0.09 (0.06); silIoU 0.9867 / 0.9873, chamfer 0.0556 / 0.0551
- ripples rest: end v_mean full 1.82e-02 / twin 1.30e-03 (14.0x)
- band: mis-coloured share full 0.005 vs twin 0.100 (floor 0.003): r = 0.97; held-out colour PSNR 15.78 vs 13.84
- band rest: end v_mean full 2.26e-03 / twin 1.41e-03 (1.6x)
- videos: ripples /data/relcfd/chayo/physmorph_v2/output/gpu/d162f/after/vid_ripple_whole/pair.mp4, /data/relcfd/chayo/physmorph_v2/output/gpu/d162f/after/vid_ripple_close/pair.mp4; band none
VERDICT: render carves the ripples (0.73 vs 0.09) but does not rest (14x twin); band recovers the colour (r 0.97) at 1.6x the twin rest

