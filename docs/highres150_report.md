# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0781 | 0.8476 | 0.39% | 219 | 23.0 | 6.3 | 0.072 | 0.670 → 0.302 | 0.118 / 0.172 | 179 | 553 (0.369 %) | 2.06 wu | FAIL |
| C | 0.5711 | 0.7045 | 0.00% | 20 | 2.1 | 6.3 | 0.733 | 1.000 → 0.526 | 0.051 / 0.156 | 3 | 74644 (49.763 %) | 2.67 wu | FAIL |
| V | 0.0766 | 0.7923 | 1.03% | 193 | 20.7 | 6.4 | 0.065 | 0.578 → 0.233 | 0.165 / 0.187 | 260 | 479 (0.319 %) | 1.94 wu | FAIL |
| armadilo | 0.0814 | 0.7940 | 0.35% | 169 | 19.0 | 6.7 | 0.086 | 0.503 → 0.427 | 0.118 / 0.209 | 86 | 892 (0.595 %) | 2.14 wu | FAIL |
| bob | 0.1621 | 0.5766 | 0.78% | 109 | 10.9 | 6.0 | 0.341 | 0.878 → 0.202 | 0.195 / 0.172 | 134 | 6827 (4.551 %) | 4.05 wu | FAIL |
| bunny | 0.0757 | 0.9738 | 0.00% | 110 | 19.9 | 10.9 | 0.088 | 0.395 → 0.337 | 0.131 / 0.192 | 0 | 0 (0.000 %) | 0.12 wu | FAIL |
| dragon | 0.1435 | 0.6421 | 2.37% | 108 | 11.0 | 6.1 | 0.281 | 1.000 → 0.316 | 0.104 / 0.178 | 128 | 4917 (3.278 %) | 4.52 wu | FAIL |
| heart | 0.0732 | 0.9832 | 0.00% | 97 | 11.3 | 7.0 | 0.076 | 0.423 → 0.341 | 0.154 / 0.191 | 0 | 0 (0.000 %) | 0.09 wu | PASS |
| spot | 0.0769 | 0.9514 | 0.00% | 86 | 8.7 | 6.1 | 0.082 | 0.588 → 0.506 | 0.099 / 0.180 | 21 | 28 (0.019 %) | 1.25 wu | FAIL |
| teapot | 0.0721 | 0.9798 | 0.00% | 123 | 16.1 | 7.8 | 0.101 | 0.261 → 0.235 | 0.201 / 0.214 | 0 | 0 (0.000 %) | 0.09 wu | PASS |

## Assessment

Mass ejection: 원인(소스 표면 껍질 입자의 초기 표류 + numerical fracture + 소성 흡수)을 막는 12개 메커니즘은 모두 반증됐고, 대신 이산화가 정의하는 규칙으로 제거했다 — 한 셀 팽창한 occupancy에서 몸체와 연결되지 않은 입자는 어떤 grid node도 공유하지 않으므로 연속체 요소가 아니며, commit마다 가장 가까운 몸체 입자에 병합된다(질량 보존, 임계값·가중치 없음, MPM/PIC의 conservative resampling). 40k 검증: armadillo/bob/dragon 모두 끝 프레임 far 0, silIoU 0.939→0.961 / 0.822→0.980 / 0.901→0.965. 한계: 표류를 되돌리는 것이라 병합 프레임에서 ~2셀 점프가 생기고 재발할 수 있다(횟수 기록).구현 버그 2건 수정: line search·commit 롤아웃의 bond 누락; --domain auto의 밀도 단위 보정 셀 누락(가중치 ~3배). 그 사이의 결과는 전부 폐기·재실행.속도: 150k window당 6.8 → 2.3–2.9 s(무부하; 영구 궤적 + CUDA graph, 테이프 adjoint graph, line-search 소진 시 window 종료). 300 window 12–14 min; 10 min 미만은 P2G/G2P 수동 adjoint가 필요. 배치 실측은 다른 사용자 CPU 부하로 6–7 s/commit.

## Mass ejection

실험 사다리와 각 단계의 수치는 docs/experiments.md 2026-09-16 'mass-ejection ladder' 절에 있다.진단도구: scripts/probes/fragment_trace.py(이탈 입자의 소스 이웃 거리비 시계열), stray_census.py(끝 프레임 far 입자), selfprop_probe.py(단일 입자 자기추진 — 무시할 수준).이탈 입자는 소스 구의 바깥 10 % 껍질 입자(dragon 81 %, armadillo 100 %, bob 45 %)이고, 이웃 거리비가 run의 1/12 시점에 4.6–6, 1/4 시점에 11–14, 이후 평탄 — 초기 팽창 구간의 선두 입자들이 매 window 제어 clip(0.02) 한계만큼 이웃보다 앞서 밀려 떨어진 뒤 다시 돌아오지 않는다(한 셀 이상 떨어지면 MPM은 두 입자를 연결하지 않고, 매 commit의 소성 흡수가 되돌리는 응력을 지운다).메커니즘 사다리 — 고정 도메인/보정된 자동 도메인에서 유효한 판정메커니즘dragon 40k chamfer / silIoUfar &gt;0.5 wu판정per-particle + 재료 재결합 v5 (bonds, no-grad 경로 버그 수정 후) — 기준0.1199 / 0.901 (auto) · 0.1224 / 0.887 (fixed)83 · 132최선, 미해결Sobolev(H1) 하강 방향 (--grad_h1)0.1240 / 0.88297반증이웃 합의 소성 흡수 (--assim_consensus)0.1294 / 0.824219반증제어 basis 24³ (--control_grid 24)0.1324 / 0.931192이탈 악화격리 veto · 탈출 hinge · 속도 cap · 연속성 line search · bond spring · 재결합 v2–v4——이전 절 (모두 반증)보정 버그--domain auto(속도 작업)가 밀도 단위 보정의 기준 셀(0.5 wu)을 cfg에 전달하지 않아 유효 가중치가 ~3배 어긋났고, 그 사이에 나온 basis/consensus/Sobolev의 '재앙적' 결과와 첫 150k 배치는 모두 이 버그의 산물이었다(dragon 40k 비율 4.36e4 → 수정 후 1.41e4, 고정 도메인 1.45e4). 위 표는 수정 후 재실행값이다.구조적으로 남은 방향문헌상 입자 분리를 구성적으로 막는 정식화는 참조 배치에 연결성을 고정하는 total-Lagrangian MPM(de Vaucorbeil 등; 형상함수를 미변형 격자에서 평가해 이웃을 영구히 유지)과 입자 도메인을 셀 크기로 제한하는 CPDI 계열(Simulating Brittle Fracture with Material Points)이다. 대가는 위상 변화(A의 구멍, teapot 손잡이)를 만드는 바로 그 numerical fracture를 금지한다는 것 — 이산화가 정의하는 결합 범위를 가진 혼합 정식화가 열린 설계다.

## Speed

단계150k s/window내용base (profile cpu2)6.8고정 233³ 격자, per-candidate Trajectory 재할당, numpy SVD/det/add.atpass 33.0–4.0영구 no-grad Trajectory + CUDA graph, 공유 격자(7.4 → 0.65 GB), torch 소성 흡수, batched det, --domain autopass 3b2.0–2.8line search 소진 시 window 종료(같은 점·같은 기울기로 이미 기각된 step을 재시험하던 30/37 rollout 제거)pass 42.3–2.9테이프 롤아웃 forward/adjoint CUDA graph(PersistentAdjoint); 남은 바닥 = iteration당 adjoint 3회 × 60–75 ms배치 실측(hyde06, 다른 사용자의 CPU 작업 5+개가 100 %로 상주) ~6 s/commit → 타깃당 ~30 min. 계측은 PHYSMORPH_TIMING=1로 window별 eval/terms/grad(phys·dt·render)/final 분해를 찍는다.

## Summary

150k 갤러리(보정본): 10개 중 7개가 40k보다 낫거나 동등(bunny 0.077/0.92, teapot 0.072/0.97, heart 0.073/0.98 ejection PASS, spot 0.077/0.95, A 0.078/0.85, armadillo 0.081/0.79, V 0.077/0.79). 2개는 150k에서 오히려 나빠짐(dragon 0.144/0.64, bob 0.162/0.58 — 가시·다리·링처럼 얇은 부위에서 재료가 뭉치고 이탈 조각이 100개 이상). C는 세 번(게이트 on 자동 도메인, 게이트 off, 고정 도메인) 모두 15~20 window에서 merit가 5 %씩 나빠져 정지 — 150k에서 실패(0.57).Mass ejection: 원인은 초기 팽창 구간의 표면 선두 입자 + numerical fracture + 소성 흡수의 조합으로 좁혀졌고, 유효한 판정 기준으로 12개 구조적 메커니즘이 반증됐다. 현재 최선은 재료 재결합 v5(dragon 40k far 274 → 83). 150k에서 그리드 연결성 기준 조각 수: heart 0, teapot 1, spot 21, bunny 47, armadillo 86, dragon 128, bob 134, A 179, V 260.속도: 150k window당 6.8 → 2.3–2.9 s(무부하), 300 window 12–14 min. 실제 배치는 다른 사용자의 CPU 부하로 6–7 s/commit(타깃당 9–23 min). 10 min 미만은 P2G/G2P 수동 adjoint가 필요.부수 수정: (a) line search·commit 롤아웃의 bond 누락, (b) 자동 도메인의 보정 셀 누락(가중치 ~3배) — heart의 조기 동결은 (b)의 결과였고, C의 동결은 진짜 회귀(팽창 overshoot)였다.

## Viewer

모든 150k 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root .../output/live --port 8765가 떠 있으므로, 로컬에서 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 run 선택기로 열면 된다(docs/viewer.md).
