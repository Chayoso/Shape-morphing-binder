# PhysMorph 150k v7 — 다섯 목표

Artifact (surface videos, PBR stills, loss curves): 

| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.0813 | 0.9735 | 0.00% | 215 | 72.9 | 20.3 | 0.032 | 0.543 → 0.173 | 0.151 / 0.172 | 0 | 0 | 0 (0.000 %) | 0.24 wu | FAIL |
| C | 0.1548 | 0.9395 | 7.16% | 163 | 62.6 | 22.3 | 0.069 | 1.000 → 0.042 | 0.161 / 0.156 | 0 | 530 | 23 (0.015 %) | 1.48 wu | FAIL |
| V | 0.0796 | 0.9692 | 0.00% | 154 | 47.3 | 18.4 | 0.028 | 0.499 → 0.161 | 0.163 / 0.187 | 0 | 1 | 0 (0.000 %) | 0.28 wu | FAIL |
| armadilo | 0.0782 | 0.9486 | 0.28% | 225 | 88.5 | 23.6 | 0.025 | 0.561 → 0.211 | 0.165 / 0.209 | 0 | 28 | 0 (0.000 %) | 0.19 wu | FAIL |
| bob | 0.0773 | 0.9690 | 2.57% | 125 | 41.5 | 26.7 | 0.015 | 0.861 → 0.135 | 0.181 / 0.172 | 0 | 126 | 0 (0.000 %) | 0.13 wu | FAIL |
| bunny | 0.0801 | 0.9658 | 0.01% | 200 | 73.9 | 22.2 | 0.056 | 0.401 → 0.199 | 0.167 / 0.192 | 0 | 2 | 0 (0.000 %) | 0.27 wu | PASS |
| dragon | 0.0852 | 0.9597 | 0.00% | 180 | 82.2 | 27.4 | 0.043 | 0.906 → 0.208 | 0.124 / 0.178 | 0 | 7 | 1 (0.001 %) | 0.97 wu | FAIL |
| heart | 0.0767 | 0.9788 | 0.00% | 105 | 26.3 | 15.0 | 0.054 | 0.412 → 0.207 | 0.181 / 0.191 | 0 | 0 | 0 (0.000 %) | 0.10 wu | PASS |
| spot | 0.0777 | 0.9719 | 0.00% | 173 | 68.1 | 23.6 | 0.031 | 0.615 → 0.219 | 0.154 / 0.180 | 0 | 3 | 0 (0.000 %) | 0.18 wu | FAIL |
| teapot | 0.0773 | 0.9695 | 0.00% | 97 | 24.5 | 15.1 | 0.100 | 0.259 → 0.187 | 0.203 / 0.214 | 0 | 0 | 0 (0.000 %) | 0.29 wu | PASS |

## Frame QA (photoreal videos)

Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.

| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 284 | 0 (0.00 cells) | 8 (5) | 0 (1) | 0 | 0 | 8 (11) | 966 (33) | 11 | 0 | 0 |
| C | 203 | 6 (1.24 cells) | 124 (17) | 34 (2) | 5 | 29 | 124 (506) | 22297 (54) | 3783 | 530 | 0 |
| V | 204 | 0 (0.01 cells) | 4 (2) | 0 (1) | 0 | 0 | 3 (3) | 1133 (39) | 7 | 1 | 0 |
| armadilo | 298 | 0 (0.05 cells) | 0 (1) | 0 (1) | 0 | 0 | 0 (0) | 702 (39) | 16 | 28 | 0 |
| bob | 166 | 0 (0.38 cells) | 130 (4) | 0 (1) | 0 | 0 | 130 (143) | 3526 (48) | 1 | 126 | 0 |
| bunny | 268 | 0 (0.00 cells) | 94 (4) | 0 (1) | 0 | 0 | 3 (3) | 250 (33) | 16 | 2 | 0 |
| dragon | 241 | 0 (0.02 cells) | 5 (3) | 0 (1) | 0 | 0 | 5 (6) | 1457 (54) | 94 | 7 | 0 |
| heart | 141 | 0 (0.00 cells) | 0 (1) | 0 (1) | 0 | 0 | 0 (0) | 37 (24) | 0 | 0 | 0 |
| spot | 229 | 0 (0.01 cells) | 5 (3) | 0 (1) | 0 | 0 | 5 (6) | 914 (30) | 1 | 3 | 0 |
| teapot | 130 | 0 (0.00 cells) | 1 (2) | 0 (1) | 0 | 0 | 1 (1) | 54 (39) | 1 | 0 | 0 |

## Assessment

1. 150k 이탈: 안전망 없이(nn150) bunny 0 / dragon 3 / bob 54 / C 74 fragments(0 / 0.002 / 0.04 / 0.05 %) — bunny는 완전히 0, dragon은 낱개 입자 3개(cell 하나보다 작아 그려지지 않음), bob과 C는 아직 안전망이 필요하다(bob 고리 끝 ~50개 덩어리, C 팔 끝 ~100개 덩어리). 안전망 포함 v7: 18개 합 284(v6 2018, 7배 감소), dragon 755 → 7, nefertiti 152 → 5, beast 493 → 68, A 16 → 0, V 29 → 1. 두 구조적 수정이 이를 만들었다: 수송 pacing 셀 합 + cell 단위 hand-off(팽창기 선두 입자 제거, §10.10)와 domain 벽(§10.11). 반증된 것: grad_h1, 수송 pacing 단독, 전역 hand-off, 선형 잔차, shell 샘플링, 입자별 paced 목표(150k에서 line search를 굶김), 표시 필드의 grid 해상, 구멍 regime의 kNN 평활.1. 렌더 기울기 → 물리 (인과 증명): 같은 seed·같은 입자의 네 런(dragon 150k): render on / 동일 설정 재실행(노이즈 바닥) / physics-only / window 40에서 render 차단. 궤적 발산은 증거가 아니다: 동일 설정 재실행이 render 런과 끝에서 4.7 spacing 갈라지고(150k에서 GPU atomics 비결정성이 접촉 동역학으로 증폭되는 카오스), 차단 쌍둥이는 그 바닥과 모든 프레임에서 구별되지 않는다(0.58 | 3.38 | 4.72 vs 0.58 | 3.33 | 4.66 spacing); physics-only만 조금 넘는다(1.70 | 4.56 | 5.96). 증명은 두 가지에 선다. (i) window마다 기록되는 제어 갱신의 구성: render 채널이 채택된 갱신의 35 %(g_share, 모든 render-on 런에서 0.34–0.35)이고 물리 기울기와의 cosine이 0.02–0.05 — 갱신의 1/3이 셀 합에 없는 방향이며, 이는 카오스와 무관한 결정론적 측정이다. (ii) 결과 대 재실행 편차: silIoU render-on 0.939 / 0.918(같은 설정 두 표본, 편차 0.021) vs physics-only 0.840 / 차단 0.770 — 편차의 4–8배 아래; 재부착 755 / 773 vs 259 / 68 (render 채널이 얇은 부위를 당긴다). bunny와 bob(카오스가 한 자릿수 작은 형상)에서는 궤적에서도 개입 신호가 보인다 — 5절.2. C: 원인은 압축이 아니라(poisson 0.0과 0.45가 똑같이 정지) 구멍 안의 소스에 셀 합이 주는 바깥 방향 밀기와 그 관성 overshoot. 입자별 수송 손실로 40k silIoU 0.72 → 0.96; 150k에서는 처음 1914–2675회 재부착(상자 덫)이었다가 벽을 넣어 silIoU 0.940 / 재부착 530 / 구멍 7.2 %(c150r_C)로 정착 — C는 morph되지만 팔 앞면에서 cell 크기 덩어리를 떨구는 유일한 target으로 남아 안전망이 아직 필요하다(안전망 없이 0.889, 끝 fragments 74).3. 떠다니는 입자 없음: 물리에서 팽창기 고립 입자를 17배 줄였고(위), 전달 규칙으로 MPM cell 하나(dx³)보다 작은 등밀도면 조각은 grid가 해상하지 못하는 물질이므로 그리지 않는다(원시/그린/제외 개수는 프레임별로 기록). 전달 규칙은 밤사이 세 번 다듬었다: (a) 등밀도면 문턱을 bulk의 절반이 아니라 '2입자 굵기 필라멘트가 그려지는 준위'(2 s²/πσ², 0.28)로 유도했고, (b) 등밀도면이 감싸지 못했지만 몸체와 조각을 잇는 입자들은 한 입자 간격 굵기의 실로 그리며(렌더의 위상이 입자 연결을 따른다), (c) '그리드가 해상하지 못하는 재료'는 부피가 아니라 질량으로 잰다 — 조각 안의 입자가 ppc(= N dx³/V, 85개) 미만이면 그리지 않고, 부호가 몸체와 반대인 닫힌 면(bunny 귀 속 빈 공간, 92프레임)은 조각이 아니라 공동으로 센다. 19개 비디오의 최종 QA: 몸체와 떨어진 조각이 그려진 프레임 — 17개 target 0, bunny 0, C 29/203(팔 앞에서 떨어지는 cell 크기 덩어리; 렌더러 없이 격자 기준으로 센 물리 조각 ≥ 1 cell은 C 6프레임, 나머지 18개 0). cow의 '떠 있는 공'은 젖꼭지 자리에 놓인 72개 입자 덩어리가 1입자 실로 이어진 것이었고(입자 척도로는 한 몸), 질량 규칙으로 그리지 않는다. 표는 2b절.4. Photoreal: Open3D/Filament(PBR 도자기, IBL + 태양광, 소프트 섀도, 바닥), 같은 밀도장의 marching-cubes 등밀도면, 두 방위 병렬, 720 px.5. 물성 → 궤적: bunny/dragon 40k, young 3e4 / 1.4e5 / 6e5, poisson 0.2 / 0.45, 소성 0.5 / 0.1. 같은 시각 프레임의 입자별 평균 발산: 초반(런의 10 %) 물성 효과는 동일 설정 재실행의 노이즈(0.013 wu)의 8–20배(soft 0.26, stiff 0.11, ν 0.45 0.11 wu), 끝에서 1.3–2.5배; 끝 chamfer는 모두 0.076–0.078. 중간 형상 스틸: soft는 아직 구, stiff는 귀가 다 뻗음. dragon: 발산 3–8 spacing(끝에서 노이즈의 7–13배), soft 재부착 343 / base 14 / stiff 0, 끝 silIoU 0.846–0.963 — 얇은 부위가 있는 형상에서는 물성이 이탈 거동과 끝 상태까지 바꾼다. 그림·표는 6절.

## Mass ejection

사다리와 수치는 docs/experiments.md 2026-09-17 evening 절, 정식화는 docs/method.md §10.8–10.10.40k, cell 0.31, 재부착 없음 — 끝 fragments / silIoU / 런 전체 고립 입자 피크meshdensity (v6)paced + hand-off (v7)bunny0 / 0.960 / 0.11 %0 / 0.961 / 0.09 %dragon0 / 0.955 / 2.3 %0 / 0.965 / 0.14 %bob2 / 0.958 / 1.1 %0 / 0.974 / 0.29 %150k, 안전망 재부착 횟수 / silIoU — v6 → v7 (v7은 22:05 이후 domain 벽 포함)targetv6v7targetv6v7bunny0 / 0.9582 / 0.966cow28 / 0.9259 / 0.926teapot0 / 0.9700 / 0.970homer5 / 0.9384 / 0.958heart0 / 0.9800 / 0.979maxplanck0 / 0.9690 / 0.968spot1 / 0.9643 / 0.972nefertiti152 / 0.9495 / 0.972A16 / 0.9690 / 0.974fandisk13 / 0.9762 / 0.973V29 / 0.9691 / 0.969ogre15 / 0.92923 / 0.939armadilo21 / 0.91928 / 0.949beast493 / 0.87368 / 0.930dragon755 / 0.9397 / 0.960cheburashka3 / 0.9546 / 0.964bob486 / 0.968126 / 0.969bimba1 / 0.9710 / 0.975C— (정지, 0.660)530 / 0.94018개 합 2018 → 284; 평균 silIoU 0.951 → 0.962domain 상자 덫 — 상자 접촉(commit 시 clip된 입자 수)이 재부착을 따라간다run상자 접촉재부착v6 dragon10706755v6 bob10934486v6 nefertiti702152150k C (벽 이전)27140–389831528–267540k C / teapot / heart / A / nn150 bunny·dragon0–70–7벽 이후 v7 런 전부0—덩어리들은 상자 모서리에 정지해 있었고(프로브 chunk_origin.py: 속도 0.002–0.02 wu/frame, target에서 1.5–3.3 wu), 원인은 스텐실 반폭(2 cell) 띠에서 잘린 스텐실이 운동량을 매 step 잃는 것. 분리형 벽(바깥 법선 속도만 0)이 수정이며 상수는 스텐실 반폭이다.같은 날 반증된 것grad_h1(40k·150k), 수송 pacing 단독(150k: 정체), 전역 hand-off(팔 끝 미도달), 모든 근접 cell hand-off(C runaway 재현), pace = blur 반경(정체), 선형 잔차, shell-biased 샘플링, window당 1 iteration; C의 압축성 가설.

## Render gradient -> physics

같은 seed·같은 입자의 dragon 150k 네 런: render on(v6 갤러리 런) / 동일 설정 재실행(rp_ctrl, 노이즈 바닥) / physics-only(rp_phys, λ = 0) / window 40에서 render 차단(rp_cut; 로그에 'render channel OFF from here' 확인). 프로브 scripts/probes/render_effect.py.render 런으로부터의 입자별 평균 거리(spacing 단위). 동일 설정 재실행(회색)이 차단 쌍둥이(주황)와 겹친다 — 150k에서 궤적 발산은 카오스 바닥이지 인과 신호가 아니다. physics-only(파랑)만 바닥을 넘는다.runwindow 40 이전 mean / max이후 mean끝silIoUchamfer재부착동일 설정 재실행0.58 / 1.413.334.660.9180.0876773render off @400.58 / 1.413.384.720.7700.100768physics-only1.70 / 2.714.565.960.8400.0899259render on (v6)———0.9390.0831755증명은 두 측정에 선다. (i) window별 제어 갱신의 구성(결정론적, 카오스와 무관): optimizer가 매 window 기록하는 g_share = λ‖g_render‖ / (‖g_phys‖ + λ‖g_render‖)가 모든 render-on 런에서 0.34–0.35(window 1–40에서 0.39–0.40), 물리 기울기와 (PCGrad 투영된) render 기울기의 cosine이 0.02(dragon, 재실행, bunny) / 0.05(bob) — 채택된 갱신의 1/3이 셀 합에 없는 방향이다. (ii) 결과 대 재실행 편차: silIoU render-on 0.939 / 0.918(같은 설정 두 표본, 편차 0.021) vs physics-only 0.840 / 차단 0.770 — 편차의 4–8배 아래; 재부착 755 / 773 vs 259 / 68 — render 채널이 얇은 부위(뿔·수염)를 당겨 silIoU를 10점 올리고, 그 대가로 재부착이 3배다.bunny·bob(v7 recipe, 같은 seed): dragon보다 카오스가 한 자릿수 작은 형상에서는 궤적에서도 개입 신호가 보인다. bunny: 차단 쌍둥이의 발산이 window 40 이전 0.14 spacing(최대 0.43)에서 이후 0.77(끝 1.05)로, bob: 0.054(최대 0.15)에서 0.57(끝 0.75)로 — 개입 시점에 5–10배 계단(bob의 동일 설정 재실행 바닥은 0.11 → 0.34, 끝 0.50: 차단 쌍둥이가 바닥의 1.5–1.7배); physics-only는 끝에서 bunny 1.8 / bob 1.8. 결과: bunny render 0.966 / physics-only 0.932 / 차단 0.936(render 없는 두 런은 283–300 window에도 수렴하지 않음, render 런은 211), bob render 0.970 / 동일 설정 재실행 0.968 / physics-only 0.952 / 차단 0.956 — bob의 재실행 편차는 0.002라 render 없는 두 런은 편차의 7–9배 아래다. window별 render 몫 0.35–0.38, cosine 0.03–0.08.bunny 150k — physics-only(파랑), window 40 차단(주황); 차단 쌍둥이는 개입 전 평탄, 개입 후 상승bob 150k — 같은 그림

## Material -> trajectory

같은 seed·같은 입자, 40k, bunny와 dragon: young 3e4(soft) / 1.4e5(base) / 6e5(stiff), poisson 0.45, 소성 흡수 0.1(더 탄성). 프로브 scripts/probes/material_trajectories.py: 같은 시각 아카이브 프레임의 입자별 평균 발산(base 런 기준), 경로 길이, chamfer 곡선; 동일 설정 재실행(mat_ctrl)이 노이즈 바닥이다.dragon, frame 250 — 같은 시각, 물성별로 다른 중간 몸(soft / base / stiff / ν 0.45 / 탄성)bunny, frame 250 — soft는 아직 구에 가깝고 stiff는 귀가 다 뻗어 있다dragon — base로부터의 발산(spacing), chamfer 곡선bunny — 같은 그림run (dragon)발산 wu @10 / 25 / 50 / 100 %끝 (spacing)경로 길이 wu재부착silIoU동일 설정 재실행 (노이즈)0.003 / 0.028 / 0.048 / 0.0601.0———soft (young 3e4)0.40 / 0.54 / 0.51 / 0.498.12.353430.846stiff (6e5)0.44 / 0.36 / 0.37 / 0.427.01.8100.963poisson 0.450.26 / 0.39 / 0.42 / 0.437.0—950.941탄성 (assim 0.1)0.09 / 0.19 / 0.18 / 0.182.9——0.958base00—140.957bunny(spacing 0.060 wu): soft 0.26 / 0.17 / 0.12 / 0.11 wu(끝 1.9 spacing, 재부착 12), stiff 0.11 / 0.09 / 0.10 / 0.11(1.8), ν 0.45 0.11 / 0.12 / 0.13 / 0.14(2.3), 탄성 0.03 / 0.05 / 0.06 / 0.07(1.2); 경로 길이 1.08(base) / 1.12 / 1.26 / 1.09 / 1.17 wu; 끝 chamfer 모두 0.076–0.078. 결론: 물성은 궤적을 초반부터 강하게 바꾸고(초반 발산은 노이즈의 8–130배), 매끈한 형상에서는 render + 밀도 목적함수가 모든 물성을 같은 끝 형상으로 데려가며, 얇은 부위가 있는 형상(dragon)에서는 이탈 거동(soft가 base의 25배)과 끝 품질(0.846–0.963)까지 물성이 정한다.

## Speed

이번 라운드의 벽시계는 비교 대상이 못 된다: 19개 런을 GPU 두 장에 3–4개씩 겹쳐 돌려 target당 18–89분(teapot 24, heart 26, maxplanck 18, bunny 74, dragon 82, armadilo 89, cheburashka 80; v6 단독 실행 8–25분). window당 비용은 수송 plan 풀이 2–4 s(8192 표본, ε-scaling, L1 정지)가 더해진 것 외에 v6과 같다. 단독 실행 속도는 다음 라운드에서 잰다.

## Summary

다섯 목표: (1) 150k 이탈 — 18/19 target에서 안전망 합 284(v6 2018), 안전망 없이 bunny 0·dragon 3; 원인 둘(팽창기 선두 입자, domain 상자 덫) 모두 구조적으로 제거. (2) C — 150k에서 silIoU 0.940으로 morph(v6 정지), 재부착 530. (3) 비디오 청결 — 등밀도면 조각 규칙(cell 미만 제외 + 2입자 필라멘트 문턱)과 프레임별 QA 표; cow의 '떠 있는 공'은 이어진 재료였고 문턱 수정으로 사라짐. (4) photoreal — 19개 전부 Filament PBR, 두 방위. (5) 물성 → 궤적 — bunny·dragon 5물성 + 동일 설정 재실행: 초반 발산 노이즈의 8–130배, dragon은 끝 상태까지 바뀜.렌더 기울기 → 물리: 세 형상에서 (i) window별 갱신의 35 %가 render 채널이고 물리 기울기와 직교(cos 0.02–0.08), (ii) 결과가 재실행 편차의 4–10배 차이(dragon +10점, bunny +3.4, bob +1.4–1.8); 150k 궤적 발산은 dragon에서 카오스 바닥과 구별되지 않고 bunny·bob에서 바닥의 1.4–1.7배 — 궤적은 보조 증거다.남은 것: C의 팔 앞면 덩어리(안전망 없이 74)와 bob 고리의 ~50개 덩어리 — 안전망을 뗀 상태로 0을 만드는 것; 단독 실행 속도; v10을 넘기면 서버·로컬의 옛 결과 정리(사용자 지시).

## Viewer

모든 150k v7 런은 /data/relcfd/chayo/physmorph_v2/output/live/h150v7_&lt;target&gt;_render_full_dt_iso_nn/에 commit별 packet을 남긴다. hyde06에서 scripts/viewer_serve.py --root …/output/live --port 8765 후 ssh -J chayo@hyde01.dabh.io -L 8765:127.0.0.1:8765 chayo@hyde06.dabh.io, http://127.0.0.1:8765/.
