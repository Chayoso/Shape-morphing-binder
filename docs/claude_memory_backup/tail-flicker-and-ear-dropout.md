---
name: tail-flicker-and-ear-dropout
description: "User directive 2026-09-24 morning: (1) decide whether the visible tail 'oscillation' is normal breathing, render-driven back-and-forth, or tangential particle flow along the surface (reconstruction shimmer) and fix accordingly; (2) the ear disappears and re-forms at 1-2 s in the z300b video — diagnose (physics vs reconstruction) and fix; survey both again"
metadata:
  type: feedback
---

The user (2026-09-24, after the night's report): "여전히 진동이 심한데 이게 진동인지 아니면, 표면에서
자꾸 랜더링이 최적화를 진행하면서 왔다갔다 거려서 그러는 건지 or 입자들이 표면을 따라 흘러서
그런건지 모르겠네. 이 부분 확인하고 좀 적용해 봐야 할 거 같은데? 그리고 1초에서 2초로 넘어갈 때(z300)
보면 지금 귀가 살짝 사라졌다가 다시 생성되어 나오는 artifact가 있어. 이것도 어떻게 해야 할지
모르겠어. 이 두 부분 지금 다시 survey해 봐."

**Why:** the night's carrier tests classified the tail motion by intervention, not by KIND; the eye
cannot tell (a) normal breathing, (b) render-driven surface back-and-forth, (c) tangential flow
under a stationary surface that makes the per-frame Poisson refit shimmer. And the ear's
disappear/re-form at 1–2 s is a new visible artefact of the tongue's sparse leading edge.

**How to apply:** first MEASURE the kind (osc_normal.py: normal vs tangential layer displacement
per window; the video sidecar's refit jitter; a stationary-surface test = Poisson of frame k vs
k+1 with the same point set frozen) and only then pick the fix from the survey (temporally coherent
reconstruction / tracked surface for (c), the objective for (a)/(b)). For the ear dropout: check
the sidecar's dropped/bridged components and the surfel count in the ear region per frame; if the
particles never retract it is reconstruction-side (component persistence / thin-feature kernels),
else physics-side (a carrier for the leading edge). Related: [[until-zero-oscillation]],
[[mid-morph-lumps-u-channel]].

**Directive (2026-09-24 17:50):** "이제부터 document도 진동이랑 thin한 쪽 물방울처럼 움직이는 것에
해당하는 부분만으로 정리해 줘" — from now on the deliverable documents (the page and the dossier)
are organised ONLY around the two defects: the oscillation and the thin-feature droplet motion.
Everything else (baselines, surveys not bearing on these, the 40k gallery history) is archived
out of the main page (output/r300_page/index_all_20260924.html keeps the full page) and the
dossier is docs/dossier_oscillation_thin.md; experiments.md stays the append-only lab log.
