# PhysMorph new meshes 150k v8

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 0.0817 | 0.9035 | 2.70% | 266 | 108.8 | 24.5 | 0.030 | 0.992 → 0.223 | 0.136 / 0.209 | 1 | 0 | 17 (0.011 %) | 2.11 wu | FAIL |
| bimba | 0.0775 | 0.9740 | 0.00% | 181 | 24.9 | 8.3 | 0.024 | 0.713 → 0.228 | 0.138 / 0.157 | 0 | 0 | 0 (0.000 %) | 0.21 wu | FAIL |
| cheburashka | 0.0775 | 0.9626 | 0.00% | 217 | 39.8 | 11.0 | 0.024 | 0.877 → 0.238 | 0.142 / 0.186 | 0 | 0 | 1 (0.001 %) | 1.20 wu | FAIL |
| cow | 0.0779 | 0.9235 | 0.00% | 218 | 70.8 | 19.5 | 0.032 | 0.489 → 0.162 | 0.147 / 0.175 | 1 | 0 | 3 (0.002 %) | 1.03 wu | FAIL |
| fandisk | 0.0776 | 0.9697 | 0.00% | 151 | 64.0 | 25.4 | 0.033 | 0.448 → 0.198 | 0.179 / 0.205 | 0 | 0 | 1 (0.001 %) | 1.35 wu | FAIL |
| homer | 0.0783 | 0.9580 | 0.00% | 289 | 83.6 | 17.4 | 0.032 | 0.605 → 0.197 | 0.133 / 0.171 | 0 | 0 | 0 (0.000 %) | 0.19 wu | FAIL |
| maxplanck | 0.0784 | 0.9692 | 0.02% | 187 | 72.9 | 23.4 | 0.048 | 0.462 → 0.167 | 0.196 / 0.209 | 0 | 0 | 0 (0.000 %) | 0.16 wu | FAIL |
| nefertiti | 0.0767 | 0.9684 | 0.00% | 189 | 54.8 | 17.3 | 0.020 | 0.808 → 0.167 | 0.158 / 0.168 | 0 | 0 | 1 (0.001 %) | 0.88 wu | FAIL |
| ogre | 0.0800 | 0.9333 | 0.98% | 200 | 60.5 | 18.1 | 0.041 | 0.865 → 0.264 | 0.139 / 0.207 | 21 | 0 | 23 (0.015 %) | 1.67 wu | FAIL |

## Frame QA (photoreal videos)

Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.

| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 361 | 0 (0.12 cells) | 39 (6) | 0 (1) | 0 | 0 | 26 (36) | 4595 (39) | 54 | 0 | 1 |
| bimba | 239 | 0 (0.00 cells) | 12 (2) | 0 (1) | 0 | 0 | 3 (3) | 1303 (33) | 2 | 0 | 0 |
| cheburashka | 287 | 0 (0.01 cells) | 3 (3) | 0 (1) | 0 | 0 | 0 (0) | 1690 (36) | 3 | 0 | 0 |
| cow | 292 | 0 (0.01 cells) | 67 (4) | 7 (2) | 7 | 0 | 59 (93) | 1097 (27) | 18 | 0 | 1 |
| fandisk | 203 | 0 (0.01 cells) | 0 (1) | 0 (1) | 0 | 0 | 0 (0) | 360 (36) | 2 | 0 | 0 |
| homer | 390 | 0 (0.00 cells) | 12 (3) | 0 (1) | 0 | 0 | 7 (8) | 855 (60) | 11 | 0 | 0 |
| maxplanck | 250 | 0 (0.00 cells) | 110 (3) | 0 (1) | 0 | 0 | 109 (110) | 339 (66) | 21 | 0 | 0 |
| nefertiti | 251 | 0 (0.01 cells) | 10 (3) | 0 (1) | 0 | 0 | 4 (5) | 1075 (36) | 1 | 0 | 0 |
| ogre | 272 | 0 (0.25 cells) | 250 (3) | 0 (1) | 0 | 0 | 250 (251) | 1234 (33) | 12 | 0 | 21 |

## Assessment

이탈 원장(안전망 없음): 9종 합 끝 fragments 23(ogre 21 + cow 1 + beast 1), 6종은 0. v7 안전망 재부착 117회, v6 710회와 비교.품질: silIoU v7 → v8: cow 0.926 → 0.924, homer 0.958 → 0.958, maxplanck 0.968 → 0.969, nefertiti 0.972 → 0.968, fandisk 0.973 → 0.970, ogre 0.939 → 0.933, beast 0.930 → 0.904, cheburashka 0.964 → 0.963, bimba 0.975 → 0.974.비디오: 등밀도면 문턱 = 2입자 필라멘트 준위, 질량 ≥ 1 cell 규칙, 공동 제외, 입자 연결 실; 자산이 바로 선 방향으로 렌더.

## Mass ejection

안전망 없는 150k — 끝 fragments / silIoU (v7 안전망 재부착 / silIoU)targetv7 (안전망)v8 (안전망 없음)cow9 / 0.9261 / 0.924homer4 / 0.9580 / 0.958maxplanck0 / 0.9680 / 0.969nefertiti5 / 0.9720 / 0.968fandisk2 / 0.9730 / 0.970ogre23 / 0.93921 / 0.933beast68 / 0.9301 / 0.904cheburashka6 / 0.9640 / 0.963bimba0 / 0.9750 / 0.974합11723정식화: docs/method.md §10.7, §10.11; 사다리: docs/experiments.md 2026-09-18.

## Render gradient -> physics



## Material -> trajectory



## Speed



## Summary

9종 모두 안전망 없이 morph; 끝 fragments 6종 0, 낱개 2, 응집 덩어리 1(ogre).남은 것: beast의 얇은 부위 품질(−2.6점), 단독 실행 속도 표.

## Viewer

모든 런은 /data/relcfd/chayo/physmorph_v2/output/live/n150v8_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765 후 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io, http://127.0.0.1:8765/.
