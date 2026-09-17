# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 0.0910 | 0.8734 | 3.18% | 159 | 11.6 | 4.4 | 0.067 | 1.000 → 0.351 | 0.059 / 0.209 | 0 | 1 (0.001 %) | 0.51 wu | FAIL |
| bimba | 0.0785 | 0.9707 | 0.00% | 180 | 12.1 | 4.0 | 0.029 | 0.763 → 0.243 | 0.127 / 0.157 | 0 | 0 (0.000 %) | 0.18 wu | FAIL |
| cheburashka | 0.0775 | 0.9544 | 0.00% | 247 | 16.2 | 3.9 | 0.020 | 0.854 → 0.227 | 0.136 / 0.186 | 0 | 1 (0.001 %) | 1.32 wu | FAIL |
| cow | 0.0767 | 0.9247 | 0.00% | 254 | 15.7 | 3.7 | 0.020 | 0.702 → 0.181 | 0.141 / 0.175 | 0 | 0 (0.000 %) | 0.17 wu | FAIL |
| fandisk | 0.0770 | 0.9760 | 0.00% | 230 | 15.2 | 4.0 | 0.021 | 0.484 → 0.190 | 0.181 / 0.205 | 0 | 0 (0.000 %) | 0.16 wu | FAIL |
| homer | 0.0813 | 0.9382 | 0.00% | 152 | 11.0 | 4.3 | 0.054 | 0.496 → 0.279 | 0.101 / 0.171 | 0 | 0 (0.000 %) | 0.19 wu | FAIL |
| maxplanck | 0.0776 | 0.9686 | 0.02% | 208 | 13.3 | 3.8 | 0.037 | 0.496 → 0.173 | 0.187 / 0.209 | 0 | 0 (0.000 %) | 0.14 wu | FAIL |
| nefertiti | 0.0850 | 0.9489 | 0.00% | 105 | 8.1 | 4.7 | 0.082 | 0.878 → 0.419 | 0.078 / 0.169 | 0 | 1 (0.001 %) | 1.12 wu | FAIL |
| ogre | 0.0790 | 0.9287 | 0.17% | 247 | 17.2 | 4.2 | 0.031 | 0.954 → 0.255 | 0.132 / 0.207 | 0 | 0 (0.000 %) | 0.28 wu | FAIL |

## Assessment

이탈: 재부착 v3 → v5 → v6: cow 83 → 10 → 28, homer 271 → 19 → 5, maxplanck 14 → 0 → 0, nefertiti 559 → 551 → 152, fandisk 600 → 142 → 13, ogre 981 → 194 → 15, beast 3443 → 894 → 493, cheburashka 3206 → 45 → 3, bimba 1491 → 127 → 1. 40k 재부착 없는 sweep(cell 0.31)에서 이 9개의 끝 프레임 fragments는 cow 0, homer 0, maxplanck 0, nefertiti 0, fandisk 0, ogre 2, beast 4, cheburashka 3, bimba 0.품질: silIoU v3 → v6: cow 0.944 → 0.925, homer 0.969 → 0.938, maxplanck 0.981 → 0.969, nefertiti 0.868 → 0.949, fandisk 0.804 → 0.976, ogre 0.879 → 0.929, beast 0.829 → 0.873, cheburashka 0.972 → 0.954, bimba 0.963 → 0.971. 거친 cell의 비용은 매끈한 메시에서 −1~−2.4 pt(homer·cow·cheburashka), 이탈하던 메시에서는 +4~+17 pt. chamfer는 0.077–0.091 wu.남은 것: beast(얇은 다리·뿔)와 nefertiti는 cell 0.31에서도 각각 493·152회 재부착 — 10개 타깃의 dragon·bob과 같은 열린 항목(이탈률은 cell 크기와 별개의 변수를 좇는다).정의: dx = 형상 대각선/26, ppc = N·dx³/V (--cell_diag 26); 근거와 사다리는 10개 타깃 갤러리와 docs/experiments.md 2026-09-17 절, docs/method.md §10.9.

## Mass ejection

원인·사다리·정의는 10개 타깃 갤러리(2f348b78)와 docs/experiments.md 2026-09-17 절, docs/method.md §10.9에 있다. 이 페이지는 같은 recipe를 새 메시 9개에 적용한 결과다.재부착 v3 → v5 → v6 (150k)meshv3 (ppc 8, cell 0.20)v5 (ppc 27, cell 0.20)v6 (cell 0.31)silIoU v6cow8310280.925homer2711950.938maxplanck14000.969nefertiti5595511520.949fandisk600142130.976ogre981194150.929beast34438944930.873cheburashka32064530.954bimba149112710.971합106481982710

## Speed

cell 0.31에서 150k grid는 35–45³; 런당 8–17 min(nefertiti 8, homer 11, beast 12, bimba 12, maxplanck 13, fandisk 15, cow 16, cheburashka 16, ogre 17), GPU당 2개씩 돌린 실측.

## Summary

새 메시 150k v6: 9개 전부 끝 프레임 비연결 입자 0; 재부착 합 710회(v3 10648, v5 1982); 6개 메시는 0–15회.silIoU v3 → v6: cow 0.944→0.925, homer 0.969→0.938, maxplanck 0.981→0.969, nefertiti 0.868→0.949, fandisk 0.804→0.976, ogre 0.879→0.929, beast 0.829→0.873, cheburashka 0.972→0.954, bimba 0.963→0.971.열린 항목: beast 493·nefertiti 152회의 잔여 재부착.

## Viewer

런의 live packet은 /data/relcfd/chayo/physmorph_v2/output/live/n150y_&lt;mesh&gt;_render_full_dt_iso_nn/(v5 n150q_…). hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765가 떠 있으므로 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io 후 http://127.0.0.1:8765/에서 연다(docs/viewer.md).
