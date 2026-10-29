---
name: formulation-stop-rule
description: "User rule 2026-10-01 ~15:55 CDT: after R14 the surviving formulation gets one final validation and is then frozen; a ratio that looks odd is recorded, not turned into a new experiment; the near band keeps its definition and scale (its early conflict with the transport is a known interaction)."
metadata:
  type: feedback
---

After R14 (leave-one-out of w_ctrl, w_creg, w_jvol and the spray cleanup) the terms that survive get ONE final
validation — the 40k gallery, the 300k bunny and dragon, the PBR renders and video, the gradient analysis — and unless
a new catastrophic failure appears there the formulation is frozen. Do not open R15/R16 because some ratio is 1.7×.

**Why:** the user said it plainly ("새 R15/R16 만들면 진짜 논문 못 씁니다"): a day of removing heuristics must end in a
paper; an uneasy number is not the same as something that must be fixed. Example decided the same day: D9b showed the
near band opposes the transport early in a run and more at 300k (not resolution-invariant), but the converged
geometry is not degraded at 40k/100k/300k → keep the definition and the scale, record it as a known interaction, do
not claim resolution invariance of the local correction in the paper, describe the competition honestly in the
analysis/supplement.

**How to apply:** when a diagnostic shows an asymmetry or an N-dependence, first ask whether the final result is
harmed; if not, write it down in docs/experiments.md and move on. New definitions that add a threshold, a gate or a
normalisation need evidence of harm to the outcome, not of inelegance. Prefer experiments that remove terms.

See [[no-parameter-only-fixes]], [[near-band-role-and-next-steps]], [[resume-2026-09-30-r11d]].
