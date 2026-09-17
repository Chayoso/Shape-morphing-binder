# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0804 | 0.9687 | 0.00% | 265 | 16.9 | 3.8 | 0.021 | 0.509 → 0.159 | 0.149 / 0.172 | 0 | 0 (0.000 %) | 0.21 wu | FAIL |
| C | 0.3832 | 0.6597 | 0.05% | 22 | 1.2 | 3.3 | 0.535 | 1.000 → 0.223 | 0.079 / 0.156 | 0 | 32301 (21.534 %) | 3.13 wu | FAIL |
| V | 0.0783 | 0.9688 | 0.00% | 181 | 11.6 | 3.8 | 0.018 | 0.573 → 0.145 | 0.167 / 0.187 | 0 | 0 (0.000 %) | 0.21 wu | FAIL |
| armadilo | 0.0818 | 0.9192 | 0.03% | 146 | 11.4 | 4.7 | 0.051 | 0.612 → 0.306 | 0.118 / 0.209 | 0 | 0 (0.000 %) | 0.19 wu | FAIL |
| bob | 0.0778 | 0.9678 | 2.66% | 218 | 13.4 | 3.7 | 0.011 | 0.752 → 0.112 | 0.182 / 0.172 | 0 | 1 (0.001 %) | 1.01 wu | FAIL |
| bunny | 0.0790 | 0.9578 | 0.00% | 276 | 24.6 | 5.3 | 0.035 | 0.458 → 0.189 | 0.163 / 0.192 | 0 | 0 (0.000 %) | 0.23 wu | PASS |
| dragon | 0.0831 | 0.9392 | 0.89% | 189 | 13.6 | 4.3 | 0.046 | 0.916 → 0.197 | 0.101 / 0.178 | 0 | 0 (0.000 %) | 0.34 wu | FAIL |
| heart | 0.0758 | 0.9803 | 0.00% | 146 | 8.9 | 3.7 | 0.038 | 0.450 → 0.203 | 0.181 / 0.191 | 0 | 0 (0.000 %) | 0.10 wu | PASS |
| spot | 0.0796 | 0.9642 | 0.01% | 143 | 9.8 | 4.1 | 0.046 | 0.698 → 0.266 | 0.132 / 0.180 | 0 | 0 (0.000 %) | 0.37 wu | FAIL |
| teapot | 0.0762 | 0.9699 | 0.00% | 198 | 16.3 | 4.9 | 0.065 | 0.261 → 0.168 | 0.204 / 0.214 | 0 | 0 (0.000 %) | 0.19 wu | FAIL |

## Assessment

원인(19개 mesh에서 확인): log 셀 합 밀도 손실은 빈 타깃 셀의 외톨이 입자에 최대 한계 이득을 주고(기울기 2r/(m_ref+m)는 빈 셀에서 최대), 표면 선두 입자가 window마다 제어 clip만큼 앞서 나가다 몸체와 한 cell 이상 벌어지면 grid node를 공유하지 못해 연속체에서 떨어진다(numerical fracture); 소성 흡수가 복원 응력을 지운다. 재부착 없는 40k에서 ppc 8 recipe는 19개 중 11개에서 끝 프레임 비연결 입자를 남겼다(합 307).수정(사용자가 제안한 mesh size): cell을 키운다. 40k에서 cell 0.31 wu(ppc 27)로 fragments 307 → 11(15개 mesh 0, 최대 4), target 밖 0.5 wu 이상 입자 dragon 216 → 0. 150k에서는 같은 ppc 27이 cell 0.20이 되어 dragon(1562)·bob(2343개 덩어리 병합)이 여전히 이탈했고, cell을 0.31로 되돌리자(ppc 91) dragon 755·bob 486(덩어리 없음), 나머지 0–29. 그래서 계약은 dx = 형상 대각선/26, ppc = N·dx³/V이다(--cell_diag 26). 26은 사다리가 보인 '갈라지지 않는 가장 고운 cell'(측정값)이고, ppc 곡선(cell 0.20 갈라짐 / 0.31 버팀 / 0.41 버티지만 세부 손실)이 근거다. C++ 오라클의 계약(형상 8 cell의 grid_dx 1.0, ppc는 샘플링에서)과 같은 구조를 더 고운 grid에서 쓰는 것이다.남은 것: dragon(755)·bob(486)·nefertiti·beast의 잔여 재부착은 형상의 얇은 부분이 cell 0.31에서도 leader를 만드는 경우다 — 이탈률은 cell 크기와 별개의 변수를 좇는다(열린 항목). C는 어느 recipe로도 morph되지 않는다(gate 정지).같은 날 반증·보류된 대안: 수송 손실 단독(구멍), 수송 leash 3종, 수송 pacing ot_pace(이탈 −86 %지만 표면 거칠음 — 비교 페이지 6b144784), 잔차/형상 수송·hand-off, loss grid 확대(dragon만 절반), 선형 잔차(log 보정과 불일치로 정지), C++식 shell-biased 샘플링(이탈 증가), window당 1 iteration(악화). OT solver 버그 2건 수정.

## Mass ejection

실험 사다리와 수치는 docs/experiments.md 2026-09-17 절, 정식화는 docs/method.md §10.8–10.9.cell 크기 곡선(재부착 없음 40k / 재부착 150k)dx (wu)40k: ppc, dragon frag / silIoU150k: ppc, dragon 재부착 / silIoU0.208: 41 / 0.85427: 1562 / 0.8330.3127: 0 / 0.95591: 755 / 0.9390.4164: 0 / 0.930—40k sweep, 19 mesh, 재부착 없음 — 끝 프레임 fragments, dx 0.20 → 0.31bunny 3→0, teapot 0→0, armadillo 12→0, heart 0→0, A 0→0, dragon 41→0, C gate 정지(양쪽), V 33→0, spot 0→0, bob 85→2, cow 2→0, homer 7→0, maxplanck 0→0, nefertiti 29→0, fandisk 1→0, ogre 17→2, beast 76→4, cheburashka 1→3, bimba 0→0. 합 307 → 11.150k 재부착 v3 → v5(dx 0.20) → v6(dx 0.31)bunny 142→27→0, teapot 32→0→0, heart 0→0→0, spot 42→1→1, A 1585→386→16, V 1074→441→29, armadillo 619→396→21, dragon 3753→1562→755, bob 770→3267→486, C 6→63→0. 합 8023 → 6143 → 1308. v6에서 덩어리 병합(한 commit 100개 이상)은 bob 188·dragon 수 건뿐이며 안전망은 target 위에 놓인 조각을 병합하지 않도록 고쳤다.

## Speed

cell 0.31은 150k에서 grid 37–43³이라 window당 시간이 짧다: 10개 런 9–25 min(heart 9, spot 10, armadillo 11, V 12, bob 13, dragon 14, teapot 16, A 17, bunny 25, C 1). GPU당 2개씩 돌린 실측.

## Summary

150k 갤러리 v6(dx 0.31 + 재부착 안전망): 10개 전부 끝 프레임 비연결 입자 0. silIoU v3→v6: bunny 0.974→0.958, teapot 0.980→0.970, heart 0.983→0.980, spot 0.976→0.964, A 0.984→0.969, V 0.975→0.969, armadillo 0.925→0.919, dragon 0.776→0.939, bob 0.581→0.968; C는 실패(gate). chamfer 0.076–0.083(v3 0.072–0.080; dragon 0.134→0.083, bob 0.170→0.078).재부착 횟수(v3 → v6): heart 0→0, teapot 32→0, bunny 142→0, spot 42→1, A 1585→16, armadillo 619→21, V 1074→29, bob 770→486, dragon 3753→755, C 6→0.40k 원인 검증 sweep(재부착 없음, 19 mesh, cell 0.31): fragments 합 307 → 11; 15개 0, 전부 ≤ 4.정의: dx = 형상 대각선/26, ppc = N·dx³/V (--cell_diag 26); 분리에는 한 cell의 간격이 필요하고 그 간격이 0.31 wu일 때 탄성 이웃이 버틴다(측정 경계). 열린 항목: cell 0.31에서도 남는 dragon·bob의 재부착.

## Viewer

모든 150k v6 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150y_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다(v5 h150q_…, v4 h150p_…, v3 h150r_…, 새 메시 v6 n150y_…). hyde06에서 /data/relcfd/chayo/physmorph_v2/repo의 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있으므로, 로컬에서 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 run 선택기로 열면 된다(docs/viewer.md).
