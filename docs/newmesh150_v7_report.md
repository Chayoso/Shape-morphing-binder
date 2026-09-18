# PhysMorph new meshes 150k v7

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 0.0815 | 0.9301 | 2.16% | 266 | 57.3 | 12.9 | 0.030 | 1.000 → 0.219 | 0.137 / 0.209 | 0 | 68 | 0 (0.000 %) | 0.34 wu | FAIL |
| bimba | 0.0774 | 0.9749 | 0.00% | 168 | 43.2 | 15.4 | 0.025 | 0.711 → 0.228 | 0.137 / 0.157 | 0 | 0 | 0 (0.000 %) | 0.28 wu | FAIL |
| cheburashka | 0.0776 | 0.9640 | 0.00% | 231 | 79.5 | 20.6 | 0.023 | 0.876 → 0.229 | 0.143 / 0.186 | 0 | 6 | 0 (0.000 %) | 0.18 wu | FAIL |
| cow | 0.0779 | 0.9261 | 0.00% | 214 | 32.2 | 9.0 | 0.033 | 0.488 → 0.163 | 0.146 / 0.175 | 0 | 9 | 0 (0.000 %) | 0.18 wu | FAIL |
| fandisk | 0.0776 | 0.9728 | 0.00% | 160 | 64.3 | 24.1 | 0.032 | 0.447 → 0.199 | 0.179 / 0.205 | 0 | 2 | 0 (0.000 %) | 0.15 wu | FAIL |
| homer | 0.0783 | 0.9575 | 0.00% | 291 | 35.6 | 7.3 | 0.032 | 0.605 → 0.208 | 0.132 / 0.171 | 0 | 4 | 0 (0.000 %) | 0.19 wu | FAIL |
| maxplanck | 0.0784 | 0.9680 | 0.02% | 175 | 18.1 | 6.2 | 0.049 | 0.462 → 0.166 | 0.196 / 0.209 | 0 | 0 | 0 (0.000 %) | 0.17 wu | FAIL |
| nefertiti | 0.0769 | 0.9720 | 0.00% | 195 | 78.8 | 24.2 | 0.022 | 0.856 → 0.185 | 0.148 / 0.169 | 0 | 5 | 0 (0.000 %) | 0.11 wu | FAIL |
| ogre | 0.0795 | 0.9388 | 1.14% | 233 | 77.2 | 19.9 | 0.037 | 0.866 → 0.253 | 0.142 / 0.207 | 0 | 23 | 0 (0.000 %) | 0.38 wu | FAIL |

## Frame QA (photoreal videos)

Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.

| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 358 | 0 (0.07 cells) | 44 (4) | 0 (1) | 0 | 0 | 26 (36) | 4590 (39) | 35 | 68 | 0 |
| bimba | 223 | 0 (0.00 cells) | 13 (3) | 0 (1) | 0 | 0 | 4 (4) | 1305 (33) | 2 | 0 | 0 |
| cheburashka | 306 | 0 (0.04 cells) | 8 (3) | 0 (1) | 0 | 0 | 8 (10) | 1690 (36) | 6 | 6 | 0 |
| cow | 286 | 0 (0.02 cells) | 70 (3) | 0 (1) | 0 | 0 | 70 (91) | 1096 (27) | 25 | 9 | 0 |
| fandisk | 215 | 0 (0.02 cells) | 0 (1) | 0 (1) | 0 | 0 | 0 (0) | 361 (36) | 0 | 2 | 0 |
| homer | 392 | 0 (0.01 cells) | 8 (2) | 0 (1) | 0 | 0 | 8 (8) | 851 (60) | 5 | 4 | 0 |
| maxplanck | 233 | 0 (0.00 cells) | 86 (4) | 0 (1) | 0 | 0 | 86 (162) | 339 (69) | 17 | 0 | 0 |
| nefertiti | 258 | 0 (0.01 cells) | 10 (3) | 0 (1) | 0 | 0 | 9 (12) | 1216 (45) | 1 | 5 | 0 |
| ogre | 310 | 0 (0.25 cells) | 33 (3) | 0 (1) | 0 | 0 | 33 (35) | 1234 (33) | 6 | 23 | 0 |

## Assessment

이탈: 9종 합 재부착 710 → 117. 가장 큰 변화는 beast(493 → 68)와 nefertiti(152 → 5) — 둘 다 v6에서 상자 덫에 걸린 입자였다(v6 nefertiti 상자 접촉 702회, beast 155회). cow 9, homer 4, maxplanck 0, fandisk 2, ogre 23, cheburashka 6, bimba 0. 끝 프레임 fragments는 9종 모두 0.품질: silIoU homer 0.938 → 0.958, nefertiti 0.949 → 0.972, beast 0.873 → 0.930, ogre 0.929 → 0.939, cheburashka 0.954 → 0.964, bimba 0.971 → 0.975; cow·maxplanck·fandisk는 ±0.003.비디오 청결: 전달 규칙(2입자 필라멘트 문턱, 입자 연결 실, 질량 ≥ 1 cell, 공동 제외)으로 9개 비디오 모두 몸체와 떨어진 조각이 그려진 프레임 0; 렌더러 없이 격자 기준으로 센 물리 조각 ≥ 1 cell도 9개 모두 0 프레임(sub-cell 군집 ogre 2프레임 21개 입자가 최대). cow의 '떠 있는 공'(62프레임)은 젖꼭지 자리의 72입자 덩어리가 1입자 실로 몸에 이어진 것이었고(cell 척도 0.07 wu까지 한 몸), 질량 규칙으로 그리지 않는다. beast 공동 20프레임은 조각이 아니다. 표는 2b절.photoreal: Open3D/Filament PBR, IBL + 태양광, 소프트 섀도, 바닥, 두 방위 병렬.

## Mass ejection

150k, 안전망 재부착 횟수 / silIoU — v6 → v7targetv6v7cow28 / 0.9259 / 0.926homer5 / 0.9384 / 0.958maxplanck0 / 0.9690 / 0.968nefertiti152 / 0.9495 / 0.972fandisk13 / 0.9762 / 0.973ogre15 / 0.92923 / 0.939beast493 / 0.87368 / 0.930cheburashka3 / 0.9546 / 0.964bimba1 / 0.9710 / 0.975합 / 평균710 / 0.943117 / 0.956원인과 수정은 메인 갤러리 페이지 4절과 docs/method.md §10.10–10.11, 사다리는 docs/experiments.md 2026-09-17 evening / 2026-09-18.

## Render gradient -> physics



## Material -> trajectory



## Speed



## Summary

9종 모두 morph, 끝 fragments 0, 재부착 합 117(v6 710).남은 것: beast(68)·ogre(23)의 얇은 부위 재부착을 안전망 없이 0으로; 단독 실행 속도 측정.

## Viewer

모든 런은 /data/relcfd/chayo/physmorph_v2/output/live/n150v7_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765 후 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io, http://127.0.0.1:8765/.
