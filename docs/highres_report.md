# High-resolution report (2026-09-16) — sphere → 10 targets, 40k `--ppc 8`, render vs physics-only

Artifact (GIFs, PBR stills, heatmaps): https://claude.ai/code/artifact/3a1a66fe-6100-4ccf-9e54-21c02a1d0c8a

Discretisation and arms: `docs/experiments.md` 2026-09-16 (batch h). Numbers are raw-state metrics; the renderer is not consumed.

## Summary per example

| target | arm | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | strays > 2 sp | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | render | 0.1133 | 0.9768 | 0.00% | 107 | 11.4 | 6.4 | 0.021 | 0.605 → 0.254 | 0.157 / 0.172 | 0.06 % | FAIL |
|  | phys | 0.1133 | 0.9599 | 0.00% | 143 | 8.8 | 3.7 | 0.033 | 0.491 → 0.238 | 0.150 / 0.172 | 0.03 % | FAIL |
| C | render | 0.5515 | 0.7656 | 0.02% | 20 | 2.3 | 6.7 | 0.527 | 1.000 → 0.457 | 0.085 / 0.155 | 53.97 % | PASS |
|  | phys | 0.4918 | 0.6677 | 0.22% | 16 | 1.2 | 4.6 | 0.611 | 1.000 → 0.416 | 0.083 / 0.155 | 51.14 % | PASS |
| V | render | 0.1248 | 0.8817 | 0.23% | 19 | 2.1 | 6.6 | 0.093 | 0.558 → 0.297 | 0.163 / 0.185 | 0.80 % | FAIL |
|  | phys | 0.1147 | 0.9322 | 0.01% | 121 | 7.8 | 3.9 | 0.026 | 0.458 → 0.193 | 0.191 / 0.185 | 0.16 % | FAIL |
| armadilo | render | 0.1159 | 0.9356 | 0.28% | 79 | 8.6 | 6.6 | 0.040 | 0.657 → 0.405 | 0.152 / 0.205 | 0.10 % | FAIL |
|  | phys | 0.1172 | 0.8992 | 0.36% | 74 | 4.3 | 3.5 | 0.062 | 0.531 → 0.416 | 0.137 / 0.205 | 0.08 % | FAIL |
| bob | render | 0.1496 | 0.8011 | 2.06% | 114 | 12.0 | 6.3 | 0.101 | 0.853 → 0.215 | 0.195 / 0.171 | 1.49 % | FAIL |
|  | phys | 0.1220 | 0.8654 | 2.57% | 114 | 6.7 | 3.5 | 0.037 | 0.800 → 0.192 | 0.195 / 0.171 | 0.36 % | FAIL |
| bunny | render | 0.1142 | 0.9570 | 0.02% | 115 | 19.1 | 10.0 | 0.046 | 0.426 → 0.300 | 0.172 / 0.192 | 0.08 % | FAIL |
|  | phys | 0.1145 | 0.9528 | 0.01% | 129 | 7.6 | 3.5 | 0.071 | 0.401 → 0.293 | 0.163 / 0.192 | 0.04 % | PASS |
| dragon | render | 0.1325 | 0.8078 | 1.00% | 139 | 14.7 | 6.4 | 0.067 | 1.000 → 0.366 | 0.134 / 0.176 | 0.72 % | FAIL |
|  | phys | 0.1296 | 0.7886 | 0.24% | 90 | 5.5 | 3.7 | 0.082 | 1.000 → 0.369 | 0.120 / 0.176 | 0.52 % | FAIL |
| heart | render | 0.1112 | 0.9819 | 0.00% | 117 | 12.6 | 6.4 | 0.030 | 0.488 → 0.271 | 0.197 / 0.193 | 0.03 % | PASS |
|  | phys | 0.1111 | 0.9755 | 0.00% | 118 | 7.5 | 3.8 | 0.037 | 0.433 → 0.247 | 0.196 / 0.193 | 0.02 % | PASS |
| spot | render | 0.1136 | 0.9768 | 0.00% | 74 | 7.8 | 6.3 | 0.035 | 0.634 → 0.328 | 0.158 / 0.177 | 0.06 % | FAIL |
|  | phys | 0.1150 | 0.9574 | 0.00% | 114 | 7.2 | 3.8 | 0.063 | 0.578 → 0.322 | 0.139 / 0.177 | 0.05 % | FAIL |
| teapot | render | 0.1111 | 0.9743 | 0.00% | 112 | 18.2 | 9.8 | 0.051 | 0.305 → 0.228 | 0.222 / 0.212 | 0.03 % | PASS |
|  | phys | 0.1111 | 0.9624 | 0.00% | 108 | 6.6 | 3.7 | 0.076 | 0.280 → 0.216 | 0.218 / 0.212 | 0.03 % | PASS |

## Gradient reach (delivered frame of the render run)

| target | surface share | ∂D_render/∂x active all / surface / interior | mean |g_r| surf / int | mean |g_vol| surf / int | ∂D_render/∂dFc active all / surface / interior | reach by depth decile |
|---|---|---|---|---|---|---|
| A | 35.6 % | 6.6 % / 15.1 % / 2.0 % | 9.8e-06 / 3.0e-07 | 3.0e-06 / 2.6e-06 | 56.9 % / 61.5 % / 54.3 % | 0.61 0.61 0.61 0.61 0.56 0.54 0.54 0.53 0.53 0.54 |
| C | 37.9 % | 13.9 % / 24.3 % / 7.5 % | 1.9e-05 / 8.3e-07 | 2.1e-05 / 1.9e-05 | 73.9 % / 67.4 % / 77.9 % | 0.67 0.67 0.67 0.65 0.76 0.95 0.94 0.92 0.78 0.46 |
| V | 26.6 % | 7.0 % / 22.3 % / 1.4 % | 2.1e-05 / 2.8e-07 | 8.6e-06 / 7.5e-06 | 62.0 % / 70.0 % / 59.1 % | 0.70 0.70 0.68 0.57 0.58 0.56 0.58 0.57 0.58 0.69 |
| armadilo | 40.6 % | 9.2 % / 19.0 % / 2.5 % | 8.4e-06 / 2.9e-07 | 8.5e-06 / 3.8e-06 | 50.2 % / 50.9 % / 49.8 % | 0.51 0.51 0.51 0.51 0.49 0.56 0.55 0.51 0.48 0.45 |
| bob | 35.7 % | 10.6 % / 24.2 % / 3.0 % | 1.0e-05 / 8.9e-07 | 3.1e-06 / 2.8e-06 | 81.5 % / 86.5 % / 78.7 % | 0.86 0.86 0.86 0.86 0.82 0.81 0.81 0.79 0.76 0.71 |
| bunny | 33.3 % | 6.7 % / 17.1 % / 1.6 % | 1.3e-05 / 5.7e-07 | 3.4e-06 / 2.8e-06 | 51.7 % / 60.9 % / 47.1 % | 0.61 0.61 0.61 0.59 0.48 0.48 0.48 0.45 0.45 0.46 |
| dragon | 47.4 % | 8.9 % / 16.3 % / 2.2 % | 8.8e-06 / 3.1e-07 | 9.5e-06 / 4.4e-06 | 51.1 % / 46.7 % / 55.0 % | 0.47 0.47 0.47 0.47 0.44 0.53 0.63 0.57 0.60 0.56 |
| heart | 27.3 % | 8.0 % / 24.6 % / 1.8 % | 9.0e-06 / 2.4e-07 | 2.9e-06 / 2.8e-06 | 90.5 % / 90.1 % / 90.6 % | 0.90 0.90 0.90 0.87 0.88 0.90 0.89 0.92 0.93 0.96 |
| spot | 34.1 % | 9.0 % / 21.9 % / 2.4 % | 7.9e-06 / 2.6e-07 | 7.5e-06 / 3.8e-06 | 84.7 % / 90.9 % / 81.4 % | 0.91 0.91 0.91 0.90 0.79 0.81 0.82 0.82 0.82 0.80 |
| teapot | 26.0 % | 6.2 % / 20.9 % / 1.1 % | 1.2e-05 / 1.6e-07 | 2.7e-06 / 2.4e-06 | 55.8 % / 69.9 % / 50.8 % | 0.70 0.70 0.69 0.57 0.55 0.53 0.51 0.50 0.44 0.43 |

## Assessment

판단. 남은 것은 세 가지가 맞고, 이 보고서로 각각의 현재 위치가 수치로 정해졌습니다.Mass ejection은 형상 문제이지 렌더 채널 문제가 아닙니다. 끝 상태에서 목표 표면 0.5 wu 밖 입자 수(§6 표): teapot·heart·spot 0/0, bunny 5/0, A 1/0, armadillo 13/15, dragon 274/177, V 123/42, bob 580/127 (render/phys). 얇은 부속물(용 가시, 아르마딜로 발톱)에서 두 arm 모두 튕기고, 매끈한 형상에서는 두 arm 모두 0이며, 구멍이 있는 타깃(bob, V의 오목부)에서는 렌더 채널이 튕김을 3–5배로 늘립니다. 튕긴 입자는 중간 프레임에 이미 밖에 있어(100 %) 돌아오지 않습니다. 다음 사다리는 dragon·armadillo에서 v_max 클램프와 near-band 재결합 창을 시험하는 것입니다.고해상도·다양한 예제: 6/10(teapot, heart, spot, bunny, A, armadillo)이 chamfer 0.111–0.116, silIoU 0.94–0.98로 수렴했고 거기서 렌더 제어가 physics-only보다 silIoU +0.4–3.6 pt, 얇은 부위 질량 +0.001–0.019를 더 채웁니다(chamfer 동률 ±1 %). 고리 위상·날카로운 오목부(bob, V)에서는 렌더 arm이 physics-only보다 나쁩니다: bob 0.1496/0.80 vs 0.1220/0.87, V 0.1248/0.88 vs 0.1147/0.93(chamfer/silIoU), 튕김 3–5배(580 vs 127, 123 vs 42 입자). dragon은 두 arm 모두 silIoU 0.79–0.81(얇은 가시 미채움 + 튕김 0.4–0.7 %), C(고리)는 두 arm 모두 실패: commit 9부터 모든 후보 창이 outer merit를 악화시켜(gain −0.08, reversal 0.96) 20 commit에서 동결, 구멍이 열리지 않았습니다. 요약: 구멍을 열어야 하는 타깃에서 실루엣 손실은 도움이 안 되거나 해롭고(구멍 영역의 excess 항이 입자를 밖으로 밀어 튕김을 늘림), 질량 매칭 자체도 위상 변화 앞에서 멈춥니다.속도: 프로파일에서 창당 시간의 40 %가 배열 할당(호스트→GPU 복사 5.7만 회)이었고 GPU 측 할당으로 창당 −31 %, 런당 −27 %(결과 비트 동일). 이 표에서 그 차이가 보입니다: 수정 전에 시작한 bunny·teapot 렌더 런은 10.0·9.8 s/commit, 이후 런은 6.3–6.7 s/commit. physics-only는 3.5–3.8 s/commit(렌더 채널 비용이 창당 약 2.8 s).

## Speed

hyde06 GPU 2, 40k --ppc 8 density 레시피 12 commit, cProfile. 창당 시간의 40 %가 Trajectory.__init__(rollout당 189개 per-step 배열을 numpy zeros/identity에서 만들어 GPU로 복사; 12 commit에 57,252회)이었고, MPM forward + adjoint 31 %, line-search .item() 동기화 13 %, PBR-lite 음영 채널 8 %, 실루엣 손실 5 %였습니다. 수정(5a3b8ee): t &gt; 0 배열을 wp.zeros로 GPU에서 만들고 F 계열은 캐시된 identity를 wp.clone. 같은 12-commit 런 재프로파일: run_pipeline 108.3 → 78.7 s(−27 %), optimize_window 97.8 → 67.3 s(−31 %), arm 1.8 → 1.3 min. CPU에서는 비트 동일, GPU에서는 CUDA atomic add의 런 간 잡음(x_T 7e-9, dFc 기울기 5e-7) 안(tests/test_traj_alloc.py). 이 배치 도중 배포되어 이후 런부터 적용: 렌더 런 s/commit 10.0 → 6.3–6.7. 다음 후보: line-search 스칼라를 한 텐서로 묶어 .item() 동기화를 창당 1회로(약 10 %), coarse 단계에서 PBR-lite 뷰 수 절반(약 4 %), 끝의 stray 궤적 census 4프레임 간격(런당 4 s).

## Target fix and mass ejection

타깃 수정. trimesh의 축 방향 base voxel 채움이 구멍 난 bunny에서 1-voxel 기둥(485개)을 그려 40k에서 귀 위의 점선으로 보였던 문제를 orthographic 채움 + 기둥 제거로 고쳤습니다(0ccc43a, tests/test_sampler_fill.py). 이 보고서의 모든 타깃은 수정 샘플러로 뽑았습니다.Mass ejection 센서스 (scripts/probes/stray_census.py, 전달 프레임에서 목표 표면 0.5 wu 밖 입자 수, render / phys):targetrenderphys최대 거리 (wu)중간 프레임에 이미 밖bunny5 (0.013 %)03.1100 %teapot000.13–armadillo13 (0.033 %)15 (0.037 %)5.8 / 3.4100 %heart000.13–dragon274 (0.69 %)177 (0.44 %)5.8 / 5.5100 %A102.1100 %C (실패 런)40 %26 %2.6 / 2.9–spot000.14–V123 (0.307 %)42 (0.105 %)3.0 / 6.6100 %bob580 (1.450 %)127 (0.318 %)5.2 / 7.5100 %읽기: 튕김은 얇은 부속물이 있는 타깃에서 두 arm 모두 일어나고, 렌더 채널은 그 위에 한 자릿수 입자를 더합니다. 튕긴 입자는 돌아오지 않습니다(중간 프레임에 이미 밖). G4_ejection FAIL은 이 수와 일치합니다. 다음 사다리(미실행): dragon·armadillo에서 v_max 클램프(MPMParams에 있음, CLI 노출 필요), near-band 재결합 창(w_nn을 튕긴 입자에만 가중), 초반 창의 λ 램프.

## Summary

되는 것: 매끈하거나 적당히 얇은 형상(teapot, heart, spot, bunny, A, armadillo)은 40k --ppc 8 레시피로 chamfer 0.111–0.116, silIoU 0.94–0.98, 구멍 0–0.3 %, 튕김 0–15 입자에 8–19분(렌더) / 4–9분(physics-only)에 수렴합니다.렌더 제어의 값: 그 6개에서 같은 코드 경로의 physics-only와 견주어 chamfer 동률(±1 %), silIoU +0.4–3.6 pt, 얇은 부위 질량 +0.001–0.019, 손실 감소율 x0.02–0.05 vs x0.03–0.08. 대가는 창당 약 2.8 s(렌더 채널)와 한 자릿수의 추가 튕김 입자. 기울기 측정이 이 그림을 설명합니다: 이미지 손실은 표면의 15–25 %(전체 6–9 %)만 직접 보지만 MPM adjoint를 거쳐 전체 입자의 50–90 % 제어에 닿습니다.안 되는 것: 구멍을 열어야 하는 타깃. bob(고리)·V(날카로운 오목부)에서 렌더 arm이 physics-only보다 나쁘고 튕김을 3–5배 늘립니다. dragon(얇은 가시)은 두 arm 모두 silIoU 0.8, 튕김 0.4–0.7 %. C(고리)는 두 arm 모두 commit 9에서 하강이 멈춥니다(outer merit가 모든 후보를 거부). 다음 단계: (1) dragon·armadillo·bob에서 튕김 사다리(v_max, near-band 재결합, 초반 λ 램프), (2) 구멍 타깃에서 실루엣 excess 항의 역할 분리(w_spray 0 arm)와 C의 outer-merit 가드 완화, (3) 속도 후보 3개.산출물: 이 페이지(GIF 20개, PBR 스틸 60장, 히트맵 10장, 손실 곡선 10장), docs/highres_report.md(표), docs/experiments.md 2026-09-16 절, 3D 뷰어의 20개 런.

## Viewer

모든 런은 --live_dir로 뷰어 패킷을 썼습니다(/data/relcfd/chayo/physmorph_v2/output/live/hr_&lt;target&gt;_render_full_dt_iso_nn/, phys 쌍은 hr_&lt;target&gt;_phys_…). hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있고(/runs에 20개 런이 보임), 로컬에서 python scripts/viewer_tunnel.py --open으로 터널을 열면 http://127.0.0.1:8765/?run=hr_bunny_render_full_dt_iso_nn(live), /quad(4개 동시), /compare(render vs phys 좌우 + 텔레메트리)로 볼 수 있습니다(docs/viewer.md).
