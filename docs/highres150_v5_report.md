# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0768 | 0.9755 | 0.00% | 172 | 27.0 | 9.4 | 0.029 | 0.691 → 0.200 | 0.138 / 0.172 | 0 | 0 (0.000 %) | 0.11 wu | FAIL |
| C | 0.4609 | 0.6428 | 0.00% | 23 | 2.6 | 6.7 | 0.663 | 1.000 → 0.363 | 0.057 / 0.156 | 0 | 35341 (23.561 %) | 3.15 wu | FAIL |
| V | 0.0760 | 0.9827 | 0.00% | 204 | 30.8 | 9.1 | 0.021 | 0.570 → 0.168 | 0.173 / 0.187 | 0 | 0 (0.000 %) | 0.13 wu | FAIL |
| armadilo | 0.0795 | 0.9313 | 0.03% | 129 | 18.0 | 8.4 | 0.052 | 0.515 → 0.390 | 0.113 / 0.209 | 0 | 33 (0.022 %) | 1.84 wu | FAIL |
| bob | 0.0773 | 0.9749 | 2.53% | 144 | 18.4 | 7.7 | 0.018 | 0.872 → 0.134 | 0.216 / 0.172 | 0 | 7 (0.005 %) | 0.84 wu | FAIL |
| bunny | 0.0770 | 0.9740 | 0.00% | 177 | 20.2 | 6.8 | 0.055 | 0.418 → 0.254 | 0.152 / 0.192 | 0 | 0 (0.000 %) | 0.16 wu | FAIL |
| dragon | 0.1109 | 0.8325 | 0.12% | 137 | 12.8 | 5.6 | 0.131 | 1.000 → 0.268 | 0.120 / 0.178 | 0 | 1561 (1.041 %) | 5.74 wu | FAIL |
| heart | 0.0746 | 0.9821 | 0.00% | 141 | 20.8 | 8.8 | 0.046 | 0.454 → 0.247 | 0.174 / 0.191 | 0 | 0 (0.000 %) | 0.10 wu | PASS |
| spot | 0.0799 | 0.9653 | 0.00% | 98 | 12.5 | 7.6 | 0.069 | 0.615 → 0.368 | 0.110 / 0.180 | 0 | 0 (0.000 %) | 0.10 wu | FAIL |
| teapot | 0.0749 | 0.9768 | 0.00% | 116 | 14.7 | 7.6 | 0.079 | 0.271 → 0.200 | 0.202 / 0.214 | 0 | 0 (0.000 %) | 0.12 wu | PASS |

## Assessment

원인(19개 mesh에서 확인): log 셀 합 밀도 손실은 빈 타깃 셀의 외톨이 입자에 최대 한계 이득을 준다(기울기 2r/(m_ref+m)는 빈 셀에서 최대). 표면 선두 입자가 window마다 제어 clip만큼 앞서 나가고, 몸체와 한 cell 이상 벌어지면 grid node를 공유하지 못해 연속체에서 분리된다(numerical fracture); 소성 흡수가 복원 응력을 지운다. 재부착 없는 40k에서 밀도 recipe(ppc 8)는 19개 중 11개에서 끝 프레임 비연결 입자를 남겼다(합 307).수정(사용자가 제안한 mesh size): ppc 27. dx = h·ppc^{1/3}이므로 cell이 3h가 되고 분리에 필요한 간격이 1.5배 — 같은 손실, 같은 제어에서 40k fragments 307 → 11, 이탈하던 mesh의 silIoU +2~+11 pt, grid 59³ → 41³라 더 빠르다. 27 = 3³은 MPM 표준 배치이며 형상별 튜닝이 아니다. ppc 64(4h)도 분리는 없지만 grid가 너무 거칠어 품질이 내려간다(dragon 0.930).150k에서 남은 것: cell이 0.20 wu가 되는 150k에서는 dragon(재부착 1562)·bob(링이 실제로 부러져 2343개 덩어리가 병합됨)·armadillo(396)·V(441)가 여전히 이탈한다. 같은 recipe가 40k(cell 0.31)에서는 0이므로 절대 cell 크기를 시험했다: 150k dx 0.31(ppc 91)은 dragon silIoU 0.833 → 0.937, chamfer 0.111 → 0.084로 품질을 크게 올리지만 재부착은 1048로 남는다 — cell 크기는 품질을, 이탈률은 다른 변수를 좇는다(열린 항목). 재부착 안전망은 target 위에 놓인 조각을 병합하지 않도록 고쳤다(다음 배치부터).같은 날 반증·보류된 대안: 수송 손실 단독(이탈 0–5, 구멍), 수송 leash 3종, 수송 pacing ot_pace(이탈 −86 %지만 표면 거칠음·얇은 부분 희박 — 비교 페이지 6b144784), 잔차/형상 수송·hand-off, loss grid 확대(dragon만 절반), 선형 잔차(log 보정과 불일치로 정지), C++식 shell-biased 샘플링(이탈 오히려 증가: C++의 면역은 형상당 8 cell짜리 거친 grid), window당 1 iteration(훨씬 악화).OT solver 버그 2건 수정(row 정규화, ε-scaling + L1 정지)과 부분표본 dual/out-of-sample map은 코드에 남아 있다(--phys_loss ot*).

## Mass ejection

실험 사다리와 수치는 docs/experiments.md 2026-09-17 절, 정식화는 docs/method.md §10.8–10.9.ppc 곡선(40k, 재부착 없음, log density loss)ppc (cell/h)dragon frag / chamfer / silIoUbobarmadillo8 (2h, dx 0.21)41 / 0.1277 / 0.85485 / 0.1239 / 0.84912 / 0.1144 / 0.93527 (3h, dx 0.31)0 / 0.1264 / 0.9552 / 0.1176 / 0.9580 / 0.1198 / 0.95564 (4h, dx 0.41)0 / 0.1361 / 0.9301 / 0.1200 / 0.9570 / 0.1220 / 0.94140k sweep, 19 mesh, 재부착 없음 — 끝 프레임 fragments, ppc 8 → 27bunny 3→0, teapot 0→0, armadillo 12→0, heart 0→0, A 0→0, dragon 41→0, C gate 정지(양쪽), V 33→0, spot 0→0, bob 85→2, cow 2→0, homer 7→0, maxplanck 0→0, nefertiti 29→0, fandisk 1→0, ogre 17→2, beast 76→4, cheburashka 1→3, bimba 0→0. 합 307 → 11.target 밖 입자 census(40k 끝 프레임, 가장 가까운 target 점에서 0.5 wu 이상)dragon 216 → 0 (최대 3.92 → 0.24 wu), bob 254 → 7, armadillo 13 → 0, bunny 0, homer 0.다른 지렛대(같은 trio)loss_res 64 → 32: 23 / 87 / 5(dragon만 절반). 선형 잔차: 파이프라인 정지. ot_pace: 2 / 1 / 0이지만 표면 품질 손실. shell-biased 샘플링(C++): dragon target 밖 552개(ppc 8 균일 216). window당 1 iteration: 408 / 363.

## Speed

ppc 27은 MPM grid를 줄여(150k: 47–68³, ppc 8 auto 도메인의 ~157³ 대비) window당 시간이 줄어든다: 150k 런 12–31 min(bunny 20, teapot 15, heart 21, spot 13, A 27, V 31, armadillo 18, dragon 13, bob 18, C 3). OT 계열의 plan 비용은 없다.

## Summary

150k 갤러리 v5(ppc 27 + 재부착 안전망): 10개 전부 끝 프레임 비연결 입자 0. silIoU v3→v5: bunny 0.974→0.974, teapot 0.980→0.977, heart 0.983→0.982, spot 0.976→0.965, A 0.984→0.976, V 0.975→0.983, armadillo 0.925→0.931, dragon 0.776→0.833, bob 0.581→0.975; C는 두 recipe 모두 실패(gate 정지). chamfer는 v3와 ±0.005 이내(dragon 0.134→0.111, bob 0.170→0.077 개선).재부착 횟수(v3 → v5): heart 0→0, teapot 32→0, spot 42→1, bunny 142→27, A 1585→386, V 1074→441, armadillo 619→396, dragon 3753→1562, bob 770→3267(한 commit에 2343개 덩어리), C 6→63.40k 원인 검증 sweep(재부착 없음, 19 mesh, ppc 27): fragments 합 307 → 11; 15개 0, 전부 ≤ 4.정의: dx = (V·ppc/N)^{1/3} = h·ppc^{1/3}; 분리 간격 = 한 cell = ppc^{1/3} h; ppc는 이산화 상수, dx는 N에서 유도. 150k에서 절대 cell 0.31 wu(ppc 91)는 dragon 품질을 0.937로 올리지만 재부착 1048은 남는다 — 열린 항목.

## Viewer

모든 150k v5 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150q_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다(v3 h150r_…, v4 h150p_…, 새 메시 v5 n150q_…). hyde06에서 /data/relcfd/chayo/physmorph_v2/repo의 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있으므로, 로컬에서 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 run 선택기로 열면 된다(docs/viewer.md).
