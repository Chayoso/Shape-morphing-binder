# PhysMorph 40k 파이프라인 검증 — 새 recipe

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 0.1196 | 0.9628 | 0.01% | – | 9.6 | – | – | 0.455 → 0.309 | 0.181 / 0.194 | 0 | 0 | 1 (0.003 %) | 0.50 wu | PASS |
| dragon | 0.1235 | 0.9555 | 0.03% | – | 11.4 | – | – | 0.984 → 0.300 | 0.154 / 0.176 | 0 | 0 | 0 (0.000 %) | 0.39 wu | PASS |

## Frame QA (photoreal videos)

Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.

| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | 502 | 0 (0.00 cells) | 191 (8) | 0 (1) | 0 | 0 | 176 (254) | 17 (114) | 0 | 0 | 0 |
| dragon | 656 | 0 (0.00 cells) | 640 (37) | 0 (1) | 0 | 0 | 636 (3883) | 59 (105) | 0 | 0 | 0 |

## Assessment

검증 항목: 새 recipe 런 완주(300 anim, early stop), post_run(등밀도면 gif·splat gif·입자 gif·PBR 스틸·loss·census·grid fragments), photoreal Poisson 영상 + sidecar(프레임별 raw/drawn/dropped/bridged/cavities + '# surface poisson' 줄 + Poisson 대체 프레임 수), 페이지 QA 표 파싱.Poisson sidecar 읽는 법: 'sub-cell dropped'가 등밀도면보다 많다(40k bunny relax 런 522 프레임 중 72 vs 0). Poisson은 산탄 잡음 덩어리나 팽창기 spray 주위에 닫힌 작은 표면을 만들고, 질량 규칙( 1)와 격자 기준 물리 조각이 QA 열이고, 두 열 모두 0이어야 한다.남은 한계: render 픽셀(1.2–1.7 spacing) 아래는 어떤 채널도 못 본다 — target 점군의 복원 추출 잡음(G5)이 바닥. 150k는 이 페이지의 모든 열이 깨끗할 때 새 런으로 간다.

## Mass ejection

40k 사다리 (2026-09-19; 같은 seed, 같은 입자)targetarmsilIoUchamferdet F min외곽층 잔차 RMS (morph / 끝)Poisson 거칠기 (중간 프레임)끝 프레임 vs 원본: 고역 잔차 / detail 상관bunnyv8 recipe0.96230.12040.6800.442 / 0.457(MC 13.4°)0.236 / +0.28+ relax0.95670.12060.7170.290 / 0.2705.6°0.201 / +0.27+ relax + G10.95680.12030.7160.288 / 0.2325.9°0.199 / +0.27+ relax + G1 + A (= 새 recipe)0.96140.11960.7580.363 / 0.3296.2°0.185 / +0.26dragonv8 recipe0.96420.12220.6630.403 / 0.3717.8°0.203 / +0.28+ relax0.95190.12370.7050.279 / 0.2617.2°0.199 / +0.32+ relax + G10.94930.12490.7430.265 / 0.2397.0°0.197 / +0.32+ relax + G1 + A (= 새 recipe)0.95940.12350.7780.341 / 0.2638.5°0.195 / +0.33정식화: docs/method.md §10.12(표면), docs/surface_gradient.md §6(gradient 단계 분석과 투영), §7(G1 + 위치 채널, 애매함의 해소); 사다리: docs/experiments.md 2026-09-19.

## Render gradient -> physics



## Material -> trajectory



## Speed



## Summary

새 recipe = v8 + 외곽층 relaxation + denoised shading + 위치 제어 채널; 전달 표면 = 외곽층 Poisson.40k에서 입자 표면 잔차 RMS 20 % 감소(recipe 대비), chamfer·det F 개선, silIoU −0.3…−0.5점; A의 굴곡은 구조.이 페이지가 150k 전 검증이다: 모든 QA 열이 깨끗해야 19 타깃 새 런으로 간다.

## Viewer

런은 /data/relcfd/chayo/physmorph_v2/output/live/p40_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765 후 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io, http://127.0.0.1:8765/.
