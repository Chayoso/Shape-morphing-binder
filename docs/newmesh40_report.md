# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): https://claude.ai/code/artifact/006c79b6-4b7e-4880-8810-d167b924ffe8

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | far > 0.5 wu | max far | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 0.1475 | 0.7375 | 2.05% | 123 | 4.4 | 2.1 | 0.091 | 1.000 → 0.572 | 0.109 / 0.201 | 76 | 497 (1.242 %) | 4.66 wu | FAIL |
| bimba | 0.1113 | 0.9793 | 0.00% | 86 | 2.9 | 2.0 | 0.027 | 0.687 → 0.324 | 0.143 / 0.157 | 0 | 0 (0.000 %) | 0.15 wu | FAIL |
| cheburashka | 0.1147 | 0.9420 | 0.01% | 130 | 4.3 | 2.0 | 0.019 | 0.785 → 0.318 | 0.152 / 0.182 | 1 | 30 (0.075 %) | 2.14 wu | FAIL |
| cow | 0.1135 | 0.9560 | 0.00% | 81 | 4.7 | 3.5 | 0.027 | 0.612 → 0.311 | 0.152 / 0.173 | 2 | 3 (0.007 %) | 1.64 wu | FAIL |
| fandisk | 0.1130 | 0.9741 | 0.00% | 97 | 5.6 | 3.5 | 0.023 | 0.483 → 0.262 | 0.190 / 0.203 | 1 | 1 (0.003 %) | 0.81 wu | FAIL |
| homer | 0.1195 | 0.9555 | 0.00% | 109 | 6.4 | 3.5 | 0.048 | 0.715 → 0.335 | 0.140 / 0.173 | 7 | 7 (0.017 %) | 1.49 wu | FAIL |
| maxplanck | 0.1113 | 0.9780 | 0.16% | 105 | 5.7 | 3.2 | 0.037 | 0.488 → 0.291 | 0.204 / 0.210 | 0 | 0 (0.000 %) | 0.14 wu | PASS |
| nefertiti | 0.1173 | 0.9593 | 0.01% | 137 | 8.1 | 3.6 | 0.062 | 0.745 → 0.416 | 0.125 / 0.170 | 29 | 29 (0.072 %) | 1.67 wu | FAIL |
| ogre | 0.1196 | 0.9160 | 0.13% | 110 | 3.8 | 2.0 | 0.051 | 0.856 → 0.466 | 0.127 / 0.206 | 17 | 27 (0.068 %) | 2.25 wu | FAIL |

## Assessment

9개 중 8개가 chamfer 0.111–0.120 / silIoU 0.92–0.98(cow, homer, max-planck, nefertiti, ogre, fandisk, cheburashka, bimba); beast만 얇은 팔다리 때문에 0.148 / 0.74(hole 2 %). 이탈 조각은 max-planck·bimba 0, cow 2, fandisk 1, cheburashka 1, homer 7, ogre 17, nefertiti 29, beast 76 — 같은 recipe, 재부착 없이 돌린 값.같은 코드(52b401e 이후)이며 이탈 입자 수는 그리드 연결성 fragment 기준. 재부착(--reattach)을 켜면 끝 프레임 이탈은 구조적으로 0이 된다(armadillo·bob 40k 검증: 0개).

## Mass ejection



## Speed



## Summary



## Viewer

live packet: /data/relcfd/chayo/physmorph_v2/output/live/n40_&lt;target&gt;_render_full_dt_iso_nn/; 뷰어는 150k 갤러리와 같은 서버(viewer_serve.py --port 8765)에서 run 선택기로 연다.
