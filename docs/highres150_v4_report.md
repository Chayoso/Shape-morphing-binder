# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0894 | 0.9827 | 0.00% | 260 | 53.4 | 11.9 | 0.039 | 0.658 → 0.158 | 0.118 / 0.172 | 0 | 0 (0.000 %) | 0.12 wu | FAIL |
| C | 0.3906 | 0.7484 | 1.26% | 31 | 4.9 | 9.1 | 0.241 | 1.000 → 0.433 | 0.020 / 0.156 | 0 | 18610 (12.407 %) | 2.93 wu | FAIL |
| V | 0.1036 | 0.9715 | 0.96% | 74 | 15.0 | 12.2 | 0.065 | 0.502 → 0.159 | 0.094 / 0.187 | 0 | 1 (0.001 %) | 0.73 wu | FAIL |
| armadilo | 0.0912 | 0.9679 | 0.25% | 223 | 48.9 | 13.7 | 0.032 | 0.443 → 0.100 | 0.189 / 0.209 | 0 | 0 (0.000 %) | 0.29 wu | FAIL |
| bob | 0.1020 | 0.9577 | 3.01% | 62 | 8.5 | 8.3 | 0.035 | 0.921 → 0.210 | 0.083 / 0.172 | 0 | 3 (0.002 %) | 0.95 wu | FAIL |
| bunny | 0.0949 | 0.9739 | 0.18% | 133 | 21.4 | 12.7 | 0.106 | 0.273 → 0.157 | 0.122 / 0.192 | 0 | 0 (0.000 %) | 0.40 wu | FAIL |
| dragon | 0.1206 | 0.9689 | 0.55% | 127 | 18.8 | 8.9 | 0.062 | 1.000 → 0.181 | 0.099 / 0.178 | 0 | 0 (0.000 %) | 0.40 wu | FAIL |
| heart | 0.0825 | 0.9776 | 0.13% | 73 | 13.1 | 10.8 | 0.099 | 0.231 → 0.186 | 0.144 / 0.191 | 0 | 0 (0.000 %) | 0.10 wu | PASS |
| spot | 0.0869 | 0.9813 | 0.00% | 230 | 45.5 | 11.9 | 0.034 | 0.399 → 0.142 | 0.163 / 0.180 | 0 | 0 (0.000 %) | 0.23 wu | FAIL |
| teapot | 0.0826 | 0.9768 | 0.36% | 83 | 16.6 | 12.1 | 0.162 | 0.252 → 0.173 | 0.182 / 0.214 | 0 | 0 (0.000 %) | 0.21 wu | PASS |

## Assessment

원인 확인(19개 mesh, 40k, 재부착 없음): 밀도 손실은 19개 중 11개에서 끝 프레임 그리드 비연결 입자를 남긴다(bob 85, beast 76, dragon 41, V 33, nefertiti 29, ogre 17, armadillo 12, homer 7, bunny 3, cow 2, cheburashka 1, fandisk 1; 합 307). 셀 합 손실이 빈 타깃 셀의 외톨이 입자에 최대 한계 이득을 주는 것(H3)이 원인이다. 수송 손실 단독으로 바꾸면 이탈은 0–5로 사라지지만 입자 규모의 채움을 못 해 구멍이 남는다(entropic map의 상이 표면 안쪽 ~0.9 spacing; 고정 cell-sum을 보는 수렴 추적기가 3–4분에 런을 멈춤).수정(ot_pace): 밀도 손실의 target을 수송 plan의 displacement interpolation(McCann)으로 한 window에 한 blur 반경씩만 앞에 둔다. 상수는 없다 — 보폭은 plan의 해상도(표본 간격), 이웃 수는 그 부피에서, 정지 기준은 표준 Sinkhorn L1 질량 오차. 같은 19개 mesh, 재부착 없이 fragments 합 307 → 44(13개 mesh 0, 16개 ≤ 3; 악화 homer 7→14; C·beast는 개선되나 8·14). 이탈하던 mesh 전부 silIoU +3~+14 pt.150k(재부착 포함) v3 → v4: 전부 fragments 0; 재부착 −90 %; silIoU dragon 0.776 → 0.969, bob 0.581 → 0.958, armadillo 0.925 → 0.968, 나머지 ±0.006; chamfer는 쉬운 7개 타깃에서 0.010–0.030 wu 나빠지고 dragon·bob에서 좋아진다.비용과 원인: PBR 스틸에서 v4는 표면이 입자 규모로 거칠고 얇은 부분(bunny 귀·발, teapot 주둥이·손잡이, V 팔 끝)이 희박하다(등밀도면 비디오에선 그 부분이 빠질 수 있음); v3는 매끈하지만 이탈 덩어리가 떠다닌다. 측정한 원인: 실제 150k 끝 상태에서 plan이 요구하는 이동은 p50 2 blur 반경, p90 7 — 표본 노이즈가 아니라 몸체 내부 밀도 재배치(MPM 구름은 균일하지 않고 타깃 표본은 균일)이며, 셀 합의 log 형태는 이를 용인하지만 plan은 안 한다. 이를 없애려던 4개 변형(on-support snap, 잔차만 수송 ot_resid, 형상 수송 ot_shape, 고정 target hand-off)은 모두 ot_pace보다 못했다(docs/experiments.md 2026-09-17).OT solver 버그 2건 수정: barycentric projection의 row 정규화(수렴 전 상이 hull 밖으로), 고정 20 sweep(질량 오차 83 %) → 기하급수 ε-scaling + L1 오차 정지(115 sweep). 8192 부분표본 dual + out-of-sample entropic map으로 window당 ~1 s(40k) / 3–4 s(150k).

## Mass ejection

실험 사다리와 각 단계의 수치는 docs/experiments.md 2026-09-17 절, 정식화는 docs/method.md §10.8에 있다.같은 날 반증된 대안변형결과(40k, 재부착 없음)판정수송 손실 단독(ot40b)0–5 fragments, hole 2.7–3.5 %, 3–4분에 정지(추적기가 고정 cell-sum을 봄)이탈은 잡지만 채움 실패leash v1(약한 hinge)merit 진동, 26–31 window 정지, hole 2–4 %반증leash v2/v3(투영 anchor + 입자별 힘 parity, 평활화)fragments 5–13, silIoU 0.73–0.78반증(강한 anchor 힘이 몸체를 찢음)수송 손실 + gate off(ot40g)20 window 정지(추적기), cell-sum 상승반증ot_pacebob 85→1, dragon 41→2, armadillo 12→0, ogre 17→3, V 33→0, bunny 3→0, nefertiti 29→0 …채택(비용 있음)ot_pace + on-support snap150k cow: 재부착 104(density 83), silIoU 0.920(0.944)반증(접선 수송이 죽음)ot_resid(잔차만 수송)bunny chamfer 0.1248, 거부 35회반증ot_shape(형상 수송, 밀도 역수 표본)bunny 0.1301, dragon 0.1357반증hand-off(결손 cell 인접 시 고정 target)bunny 0.1279, 전환 직후 거부 14회, 6분 정지반증40k 원인 검증 sweep — 끝 프레임 fragments, density → ot_pacebunny 3→0, teapot 0→0, armadillo 12→0, heart 0→0, A 0→0, dragon 41→2, C 0(frozen)→8, V 33→0, spot 0→0, bob 85→1, cow 2→2, homer 7→14, maxplanck 0→0, nefertiti 29→0, fandisk 1→0, ogre 17→3, beast 76→14, cheburashka 1→0, bimba 0→0.

## Speed

ot_pace의 추가 비용은 window당 plan 계산 ~1 s(40k, GPU 공유) / 3–4 s(150k): 8192×8192 Sinkhorn(cost block 캐시, 대칭 self-plan, L1 정지) + 전체 입자 map pass + KD-tree 투영. 150k 런은 GPU당 2개씩 돌려 9–53 min(bunny 21, teapot 17, heart 13, spot 46, A 53, V 15, armadillo 49, dragon 19, bob 9, C 5).

## Summary

150k 갤러리 v4(ot_pace + 재부착): 10개 전부 끝 프레임 그리드 비연결 입자 0, 재부착 합 8023 → 829. silIoU v3→v4: bunny 0.974→0.974, teapot 0.980→0.977, heart 0.983→0.978, spot 0.976→0.981, A 0.984→0.983, V 0.975→0.972, armadillo 0.925→0.968, dragon 0.776→0.969, bob 0.581→0.958; C는 두 recipe 모두 실패(gate 정지).재부착 횟수(v3 → v4): heart 0→0, teapot 32→2, bunny 142→19, spot 42→34, V 1074→81, A 1585→90, dragon 3753→134, armadillo 619→153, bob 770→253, C 6→63.비용: chamfer가 쉬운 타깃에서 0.010–0.030 wu 나빠지고(v3 0.072–0.080 → v4 0.083–0.104), 스틸에서 표면 거칠기와 얇은 부분의 희박함이 보인다. 열린 항목은 plan의 내부 재배치 요구를 없애는 것.40k 원인 검증 sweep(재부착 없음, 19 mesh): fragments 합 307 → 44; 13개 0, 16개 ≤ 3; 악화 homer(7→14); fragment-free 아님 C/beast/homer.

## Viewer

모든 150k v4 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150p_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다(v3는 h150r_…, 새 메시 v4는 n150p_…). hyde06에서 /data/relcfd/chayo/physmorph_v2/repo의 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있으므로, 로컬에서 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 run 선택기로 열면 된다(docs/viewer.md).
