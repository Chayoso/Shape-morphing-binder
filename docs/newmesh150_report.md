# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beast | 0.1217 | 0.8286 | 1.69% | 110 | 13.1 | 7.1 | 0.158 | 1.000 → 0.698 | 0.047 / 0.209 | 0 | 4147 (2.765 %) | 3.42 wu | FAIL |
| bimba | 0.0798 | 0.9634 | 0.01% | 196 | 19.3 | 5.9 | 0.069 | 0.711 → 0.354 | 0.120 / 0.157 | 0 | 554 (0.369 %) | 2.06 wu | FAIL |
| cheburashka | 0.0773 | 0.9715 | 0.00% | 110 | 11.7 | 6.4 | 0.069 | 0.788 → 0.541 | 0.085 / 0.186 | 0 | 0 (0.000 %) | 0.11 wu | FAIL |
| cow | 0.0755 | 0.9437 | 0.00% | 104 | 11.2 | 6.5 | 0.080 | 0.561 → 0.502 | 0.086 / 0.175 | 0 | 0 (0.000 %) | 0.10 wu | FAIL |
| fandisk | 0.1106 | 0.8044 | 0.02% | 142 | 18.5 | 7.8 | 0.292 | 0.400 → 0.273 | 0.164 / 0.205 | 0 | 3544 (2.363 %) | 3.25 wu | FAIL |
| homer | 0.0782 | 0.9687 | 0.01% | 180 | 26.5 | 8.8 | 0.088 | 0.710 → 0.315 | 0.111 / 0.171 | 0 | 0 (0.000 %) | 0.12 wu | FAIL |
| maxplanck | 0.0734 | 0.9812 | 0.13% | 112 | 10.9 | 5.8 | 0.080 | 0.447 → 0.324 | 0.167 / 0.209 | 0 | 0 (0.000 %) | 0.09 wu | FAIL |
| nefertiti | 0.0896 | 0.8680 | 0.95% | 84 | 12.4 | 8.9 | 0.136 | 0.748 → 0.593 | 0.072 / 0.169 | 0 | 1090 (0.727 %) | 2.98 wu | FAIL |
| ogre | 0.0840 | 0.8794 | 1.28% | 123 | 13.2 | 6.4 | 0.108 | 0.770 → 0.589 | 0.083 / 0.207 | 0 | 818 (0.545 %) | 2.40 wu | FAIL |

## Assessment

chamfer 0.073–0.090, silIoU 0.87–0.98이 7개(max-planck 0.073/0.981, cheburashka 0.077/0.972, homer 0.078/0.969, bimba 0.080/0.963, cow 0.076/0.944, ogre 0.084/0.879, nefertiti 0.090/0.868); 약한 둘은 fandisk(0.111/0.804 — 날카로운 모서리)와 beast(0.122/0.829, hole 1.7 % — 얇은 팔다리).재부착 횟수(표류의 대가): max-planck 14, cow 83, homer 271, nefertiti 559, fandisk 600, ogre 981, bimba 1491, cheburashka 3206, beast 3443. 최종 상태의 이탈 입자는 모두 0.런 길이가 26–94 commit로 짧다(외부 merit 게이트의 정지); 40k보다 chamfer는 좋고 silIoU는 비슷하다.

## Mass ejection



## Speed



## Summary



## Viewer

live packet: /data/relcfd/chayo/physmorph_v2/output/live/n150_&lt;target&gt;_render_full_dt_iso_nn/; 뷰어는 같은 서버(viewer_serve.py --port 8765)에서 run 선택기로 연다.
