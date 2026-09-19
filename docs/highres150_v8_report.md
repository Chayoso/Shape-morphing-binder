# PhysMorph 150k v8 — 안전망 없이

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0823 | 0.9668 | 0.00% | 203 | 78.8 | 23.3 | 0.031 | 0.524 → 0.173 | 0.148 / 0.172 | 0 | 0 | 6 (0.004 %) | 1.63 wu | FAIL |
| C | 0.1148 | 0.9053 | 0.01% | 223 | 65.2 | 17.5 | 0.021 | 1.000 → 0.053 | 0.145 / 0.157 | 3 | 0 | 134 (0.089 %) | 1.93 wu | FAIL |
| V | 0.0796 | 0.9634 | 0.00% | 141 | 44.5 | 18.9 | 0.027 | 0.504 → 0.161 | 0.164 / 0.186 | 0 | 0 | 6 (0.004 %) | 1.21 wu | FAIL |
| armadilo | 0.0780 | 0.9380 | 0.02% | 247 | 87.5 | 21.3 | 0.025 | 0.521 → 0.187 | 0.175 / 0.209 | 1 | 0 | 5 (0.003 %) | 1.65 wu | FAIL |
| bob | 0.0776 | 0.9486 | 0.00% | 115 | 37.8 | 19.6 | 0.016 | 0.871 → 0.140 | 0.179 / 0.173 | 9 | 0 | 20 (0.013 %) | 2.40 wu | FAIL |
| bunny | 0.0801 | 0.9635 | 0.00% | 195 | 43.1 | 13.2 | 0.055 | 0.422 → 0.190 | 0.167 / 0.194 | 0 | 0 | 2 (0.001 %) | 1.70 wu | PASS |
| dragon | 0.0846 | 0.9548 | 0.09% | 245 | 87.2 | 21.4 | 0.040 | 1.000 → 0.184 | 0.129 / 0.176 | 1 | 0 | 2 (0.001 %) | 1.21 wu | FAIL |
| heart | 0.0766 | 0.9783 | 0.00% | 96 | 27.8 | 17.3 | 0.054 | 0.400 → 0.205 | 0.182 / 0.192 | 0 | 0 | 0 (0.000 %) | 0.12 wu | PASS |
| spot | 0.0778 | 0.9646 | 0.00% | 197 | 68.3 | 20.8 | 0.029 | 0.614 → 0.202 | 0.158 / 0.181 | 0 | 0 | 3 (0.002 %) | 1.34 wu | FAIL |
| teapot | 0.0771 | 0.9674 | 0.00% | 136 | 42.0 | 18.5 | 0.093 | 0.263 → 0.171 | 0.208 / 0.214 | 0 | 0 | 0 (0.000 %) | 0.19 wu | FAIL |

## Frame QA (photoreal videos)

Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.

| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 272 | 0 (0.00 cells) | 4 (5) | 0 (1) | 0 | 0 | 3 (4) | 1512 (33) | 14 | 0 | 0 |
| C | 300 | 0 (0.74 cells) | 283 (13) | 12 (2) | 12 | 0 | 281 (734) | 23127 (54) | 836 | 0 | 3 |
| V | 187 | 0 (0.00 cells) | 1 (3) | 0 (1) | 0 | 0 | 1 (1) | 1184 (36) | 13 | 0 | 0 |
| armadilo | 327 | 0 (0.01 cells) | 2 (2) | 0 (1) | 0 | 0 | 2 (2) | 741 (39) | 14 | 0 | 1 |
| bob | 154 | 0 (0.11 cells) | 29 (6) | 0 (1) | 0 | 0 | 9 (11) | 3271 (51) | 12 | 0 | 9 |
| bunny | 258 | 0 (0.06 cells) | 49 (5) | 0 (1) | 0 | 0 | 0 (0) | 272 (33) | 21 | 0 | 0 |
| dragon | 327 | 0 (0.01 cells) | 3 (3) | 0 (1) | 0 | 0 | 2 (2) | 1474 (51) | 87 | 0 | 1 |
| heart | 129 | 0 (0.00 cells) | 1 (2) | 0 (1) | 0 | 0 | 1 (1) | 36 (24) | 0 | 0 | 0 |
| spot | 267 | 0 (0.00 cells) | 1 (4) | 0 (1) | 0 | 0 | 1 (3) | 903 (33) | 5 | 0 | 0 |
| teapot | 179 | 0 (0.00 cells) | 0 (1) | 0 (1) | 0 | 0 | 0 (0) | 45 (36) | 0 | 0 | 0 |

## Assessment

이탈 원장(안전망 없음, 끝 프레임 fragments): bunny 0 · teapot 0 · heart 0 · spot 0 · V 0 · A 0 · homer 0 · maxplanck 0 · nefertiti 0 · fandisk 0 · cheburashka 0 · bimba 0 · dragon 1 · armadilo 1 · cow 1 · beast 1 · C 3 · bob 9 · ogre 21 — 19개 중 12개가 0이고, 1은 낱개 입자, bob 9와 ogre 21은 스무 개 안팎의 입자가 spacing의 1/20로 뭉친 응집 덩어리 하나(cell 미만이라 그리지 않음), C 3은 팔 앞면 잔여. 격자 기준 물리 조각(≥ 1 cell) 프레임은 19개 모두 0. 렌더러 없이 격자 기준으로 센 프레임별 물리 조각과 등밀도면 QA는 2b절.품질: 안전망을 떼면 silIoU가 대부분 0.2–0.7점 낮아진다(bunny 0.966 → 0.964, dragon 0.960 → 0.955, spot 0.972 → 0.965, heart 0.979 → 0.978, teapot 0.970 → 0.967, A 0.974 → 0.967, V 0.969 → 0.963, cow 0.926 → 0.924, homer 0.958 → 0.958, maxplanck 0.968 → 0.969, nefertiti 0.972 → 0.968, fandisk 0.973 → 0.970, cheburashka 0.964 → 0.963, bimba 0.975 → 0.974); 얇은 부위에서 안전망이 재료를 되돌려 주던 몫이 큰 곳만 더 떨어진다: bob 0.969 → 0.949, beast 0.930 → 0.904, armadilo 0.949 → 0.938, C 0.940 → 0.905무엇이 바뀌었나: 벽(v6 dragon 재부착 755 = 상자 접촉 10706회 → 0), step 결합 판정(150k bob 무안전망 54 → 1, ν 0.45 dragon 95 → 0, soft dragon 343 → 1), 조기 종료(거절 후보 12회 재제안 제거), 방향(bunny·spot·nefertiti·teapot·dragon·armadilo·heart·A·C·V·bob이 이제 바로 선다).표면: 등밀도면의 오렌지필 질감은 target 점군을 같은 파이프라인으로 렌더해도 같은 값(이면각 14.5°)인 150k 샘플링 잡음이고, 셀 척도 덩어리는 target 표면 대비 대역 진폭 0.01 wu(spacing의 1/3)로 loss cell을 절반으로 줄여도 변하지 않았다. F-공분산 가우시안 커널은 오히려 더 거칠어(19–22°) 반증.C: 안전망 없이 silIoU 0.905, 끝 fragments 3, 런 중 고립 피크 15 % — 팔 앞면에서 떨어진 재료는 되돌아올 수 없지만(§10.10a) 벽 + step 결합 판정 이후로는 조각이 모두 cell 미만(격자 프로브: 299 프레임 중 ≥ 1 cell 조각 0, 최대 63개 입자 = 0.74 cell)이라 전달 규칙이 그리지 않으며, 팔이 자라며 남은 재료를 끝에 흡수해 끝 원장은 3이다. 안전망을 쓰면 0.940. C는 여전히 한계 사례로 기록한다

## Mass ejection

안전망 없는 150k — 끝 fragments / silIoU (v7 안전망 재부착 횟수 / silIoU 참고)targetv7 재부착 / silIoU (안전망)v8 fragments / silIoU (안전망 없음)targetv7v8bunny2 / 0.9660 / 0.964cow9 / 0.9261 / 0.924teapot0 / 0.9700 / 0.967homer4 / 0.9580 / 0.958heart0 / 0.9790 / 0.978maxplanck0 / 0.9680 / 0.969spot3 / 0.9720 / 0.965nefertiti5 / 0.9720 / 0.968A0 / 0.9740 / 0.967fandisk2 / 0.9730 / 0.970V1 / 0.9690 / 0.963ogre23 / 0.93921 / 0.933armadilo28 / 0.9491 / 0.938beast68 / 0.9301 / 0.904dragon7 / 0.9601 / 0.955cheburashka6 / 0.9640 / 0.963bob126 / 0.9699 / 0.949bimba0 / 0.9750 / 0.974C530 / 0.9403 / 0.905안전망 재부착 합 814 → 안전망 없는 끝 fragments 합 39정식화: docs/method.md §10.7(step 결합 판정), §10.10a(C 한계), §10.11(벽); 사다리: docs/experiments.md 2026-09-18.

## Render gradient -> physics



## Material -> trajectory



## Speed



## Summary

v8 = 안전망 없는 150k 갤러리: 19개 중 12개 끝 fragments 0, 4개는 낱개 입자 1, bob 9·ogre 21은 응집 덩어리 하나, C 3 — 격자 기준 ≥ 1 cell 물리 조각은 19개 모두 0 프레임. 두 구조적 수정(domain 벽, step 결합 판정)이 v6의 재부착 2018회를 안전망 없는 원장 39개 입자로 바꿨다. 품질은 안전망 대비 0–0.7점, 얇은 부위 target(bob·beast·armadilo·C)은 1–3.5점 낮다.남은 것: C(안전망 없이는 팔 앞면 덩어리), 단독 실행 속도, v10 이후 정리 규칙.

## Viewer

모든 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150v8_&lt;target&gt;_render_full_dt_iso_nn/(새 mesh는 n150v8_)에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765 후 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io, http://127.0.0.1:8765/.
