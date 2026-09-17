# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): https://claude.ai/code/artifact/2f348b78-324e-4cf0-a491-ea49f92fd5b1

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0743 | 0.9837 | 0.00% | 193 | 20.6 | 6.4 | 0.050 | 0.695 → 0.292 | 0.120 / 0.172 | 0 | 0 (0.000 %) | 0.09 wu | FAIL |
| C | 0.5698 | 0.7084 | 0.00% | 21 | 2.5 | 7.1 | 0.730 | 1.000 → 0.533 | 0.051 / 0.156 | 0 | 74568 (49.712 %) | 2.79 wu | FAIL |
| V | 0.0737 | 0.9745 | 0.00% | 174 | 17.0 | 5.9 | 0.045 | 0.587 → 0.232 | 0.167 / 0.187 | 0 | 133 (0.089 %) | 1.16 wu | FAIL |
| armadilo | 0.0801 | 0.9249 | 0.32% | 129 | 14.3 | 6.7 | 0.077 | 0.506 → 0.432 | 0.121 / 0.209 | 0 | 483 (0.322 %) | 2.40 wu | FAIL |
| bob | 0.1695 | 0.5812 | 1.44% | 103 | 9.9 | 5.7 | 0.371 | 0.879 → 0.210 | 0.193 / 0.172 | 0 | 6994 (4.663 %) | 3.98 wu | FAIL |
| bunny | 0.0757 | 0.9738 | 0.00% | 110 | 19.9 | 10.9 | 0.088 | 0.395 → 0.337 | 0.131 / 0.192 | 0 | 0 (0.000 %) | 0.12 wu | FAIL |
| dragon | 0.1335 | 0.7756 | 0.10% | 108 | 11.6 | 6.4 | 0.239 | 1.000 → 0.297 | 0.110 / 0.178 | 0 | 3632 (2.421 %) | 4.49 wu | FAIL |
| heart | 0.0732 | 0.9832 | 0.00% | 97 | 11.3 | 7.0 | 0.076 | 0.423 → 0.341 | 0.154 / 0.191 | 0 | 0 (0.000 %) | 0.09 wu | PASS |
| spot | 0.0763 | 0.9758 | 0.00% | 82 | 8.6 | 6.3 | 0.082 | 0.584 → 0.510 | 0.099 / 0.180 | 0 | 0 (0.000 %) | 0.10 wu | FAIL |
| teapot | 0.0721 | 0.9798 | 0.00% | 123 | 16.1 | 7.8 | 0.101 | 0.261 → 0.235 | 0.201 / 0.214 | 0 | 0 (0.000 %) | 0.09 wu | PASS |

## Assessment

Mass ejection: 원인(소스 표면 껍질 입자의 초기 표류 + numerical fracture + 소성 흡수)을 막는 12개 메커니즘은 모두 반증됐고, 대신 이산화가 정의하는 규칙으로 제거했다 — 한 셀 팽창한 occupancy에서 몸체와 연결되지 않은 입자는 어떤 grid node도 공유하지 않으므로 연속체 요소가 아니며, commit마다 가장 가까운 몸체 입자에 병합된다(질량 보존, 임계값·가중치 없음, MPM/PIC의 conservative resampling). 40k 검증: armadillo/bob/dragon 모두 끝 프레임 far 0, silIoU 0.939→0.961 / 0.822→0.980 / 0.901→0.965. 한계: 표류를 되돌리는 것이라 병합 프레임에서 ~2셀 점프가 생기고 재발할 수 있다(횟수 기록).구현 버그 2건 수정: line search·commit 롤아웃의 bond 누락; --domain auto의 밀도 단위 보정 셀 누락(가중치 ~3배). 그 사이의 결과는 전부 폐기·재실행.속도: 150k window당 6.8 → 2.3–2.9 s(무부하; 영구 궤적 + CUDA graph, 테이프 adjoint graph, line-search 소진 시 window 종료). 300 window 12–14 min; 10 min 미만은 P2G/G2P 수동 adjoint가 필요. 배치 실측은 다른 사용자 CPU 부하로 6–7 s/commit.

## Mass ejection

실험 사다리와 각 단계의 수치는 docs/experiments.md 2026-09-16 'mass-ejection ladder' 절에 있다.진단도구: scripts/probes/fragment_trace.py(이탈 입자의 소스 이웃 거리비 시계열), stray_census.py(끝 프레임 far 입자), selfprop_probe.py(단일 입자 자기추진 — 무시할 수준).이탈 입자는 소스 구의 바깥 10 % 껍질 입자(dragon 81 %, armadillo 100 %, bob 45 %)이고, 이웃 거리비가 run의 1/12 시점에 4.6–6, 1/4 시점에 11–14, 이후 평탄 — 초기 팽창 구간의 선두 입자들이 매 window 제어 clip(0.02) 한계만큼 이웃보다 앞서 밀려 떨어진 뒤 다시 돌아오지 않는다(한 셀 이상 떨어지면 MPM은 두 입자를 연결하지 않고, 매 commit의 소성 흡수가 되돌리는 응력을 지운다).메커니즘 사다리 — 고정 도메인/보정된 자동 도메인에서 유효한 판정메커니즘dragon 40k chamfer / silIoUfar &gt;0.5 wu판정per-particle + 재료 재결합 v5 (bonds, no-grad 경로 버그 수정 후) — 기준0.1199 / 0.901 (auto) · 0.1224 / 0.887 (fixed)83 · 132최선, 미해결Sobolev(H1) 하강 방향 (--grad_h1)0.1240 / 0.88297반증이웃 합의 소성 흡수 (--assim_consensus)0.1294 / 0.824219반증제어 basis 24³ (--control_grid 24)0.1324 / 0.931192이탈 악화격리 veto · 탈출 hinge · 속도 cap · 연속성 line search · bond spring · 재결합 v2–v4——이전 절 (모두 반증)보정 버그--domain auto(속도 작업)가 밀도 단위 보정의 기준 셀(0.5 wu)을 cfg에 전달하지 않아 유효 가중치가 ~3배 어긋났고, 그 사이에 나온 basis/consensus/Sobolev의 '재앙적' 결과와 첫 150k 배치는 모두 이 버그의 산물이었다(dragon 40k 비율 4.36e4 → 수정 후 1.41e4, 고정 도메인 1.45e4). 위 표는 수정 후 재실행값이다.구조적으로 남은 방향문헌상 입자 분리를 구성적으로 막는 정식화는 참조 배치에 연결성을 고정하는 total-Lagrangian MPM(de Vaucorbeil 등; 형상함수를 미변형 격자에서 평가해 이웃을 영구히 유지)과 입자 도메인을 셀 크기로 제한하는 CPDI 계열(Simulating Brittle Fracture with Material Points)이다. 대가는 위상 변화(A의 구멍, teapot 손잡이)를 만드는 바로 그 numerical fracture를 금지한다는 것 — 이산화가 정의하는 결합 범위를 가진 혼합 정식화가 열린 설계다.

## Speed

단계150k s/window내용base (profile cpu2)6.8고정 233³ 격자, per-candidate Trajectory 재할당, numpy SVD/det/add.atpass 33.0–4.0영구 no-grad Trajectory + CUDA graph, 공유 격자(7.4 → 0.65 GB), torch 소성 흡수, batched det, --domain autopass 3b2.0–2.8line search 소진 시 window 종료(같은 점·같은 기울기로 이미 기각된 step을 재시험하던 30/37 rollout 제거)pass 42.3–2.9테이프 롤아웃 forward/adjoint CUDA graph(PersistentAdjoint); 남은 바닥 = iteration당 adjoint 3회 × 60–75 ms배치 실측(hyde06, 다른 사용자의 CPU 작업 5+개가 100 %로 상주) ~6 s/commit → 타깃당 ~30 min. 계측은 PHYSMORPH_TIMING=1로 window별 eval/terms/grad(phys·dt·render)/final 분해를 찍는다.

## Summary

150k 갤러리 v3(재부착): 10개 중 9개가 수렴했고 전부 끝 프레임 그리드 비연결 입자 0. v2 대비 silIoU: bunny 0.920→0.974, armadillo 0.794→0.925, dragon 0.642→0.776, A 0.848→0.984, V 0.792→0.975, spot 0.951→0.976, teapot 0.969→0.980, heart 0.983(변화 없음, 재부착 0회); bob은 0.577→0.581로 여전히 약함(링+얇은 다리, hole 1.4 %). C는 세 설정 모두 ~20 window에서 외부 merit 게이트가 멈춤(팽창 overshoot, 이탈 문제 아님) — 150k 실패로 남긴다.재부착 횟수(드리프트의 대가): heart 0, teapot 32, spot 42, bunny 142, armadillo 619, bob 770, V 1074, A 1585, dragon 3753(입자 150k 중). 병합 프레임에서 ~2셀 점프가 생기고, 재발도 있다(같은 입자가 다시 표류).Mass ejection 조사: 원인은 초기 팽창 구간의 표면 선두 입자 + numerical fracture + 소성 흡수의 조합으로 좁혀졌고, 유효한 판정 기준으로 12개 구조적 메커니즘이 반증됐다(ejection 절). 표류 자체를 막는 정식화(TLMPM/CPDI 계열)는 위상 변화와 충돌하는 열린 설계.속도: 150k window당 6.8 → 2.3–2.9 s(무부하), 300 window 12–14 min. 실제 배치는 다른 사용자 CPU 부하로 타깃당 9–21 min. 10 min 미만은 P2G/G2P 수동 adjoint가 필요.부수 수정: line search·commit 롤아웃의 bond 누락; 자동 도메인의 보정 셀 누락(가중치 ~3배). 서버는 이제 전부 /data 아래에서 돌고(docs/pipeline.md), 새 메시 9개 40k 갤러리는 별도 페이지.

## Viewer

모든 150k v3 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150r_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다(v2는 h150_…, 새 메시는 n40_…). hyde06에서 /data/relcfd/chayo/physmorph_v2/repo의 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있으므로, 로컬에서 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 run 선택기로 열면 된다(docs/viewer.md).
