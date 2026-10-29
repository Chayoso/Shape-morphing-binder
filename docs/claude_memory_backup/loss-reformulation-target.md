---
name: loss-reformulation-target
description: "User goal set 2026-09-30 ~18:10 CDT: the objective goes from MJ's 13 terms / 14 tuned constants to 6-7 terms / 5 constants (transport, surface proximity, stability, W1 cleanup, silhouette, shading, maybe one driven-phase regularity term; constants = lambda's calibration plus the render's internal ones). Where it stands (R11d = stability settled as the released motion outside ot_scale) and the order to get there."
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-30T23:33:33.784Z
---

Target (user, 2026-09-30): "6~7/5로 줄이는 걸 목표로 하자" — six or seven loss terms and five constants.

Where it stands (repo_r30, 2026-10-01 19:40 CDT): EIGHT terms — Sinkhorn transport, surface proximity, end drift
(geometry); released motion (settling); near band berth…one loss cell, spray cleanup (local); silhouette, shading
(render). Removed since MJ's code: the density support (weight 8, bound, floor), w_kin 5, w_kin_var 200, the box leash
10, the near band's far bound 1000, the dense distance in the selection merit, w_ctrl 1e-3, w_creg 100 (creg_k 8),
w_jvol 50. Constants left: w_nn 0.2, w_dt 0.2 with the isolation ramp 1.2/1.8, lambda_auto 0.5, sil_k, w_hole,
w_spray, w_pbr, pbr_ambient (the target was 6–7 terms / 5 constants; the two local weights and the render's four
are what remains above it). The final validation passed on 2026-10-01 20:51 CDT (no catastrophic flag); the formulation is frozen by the stop
rule ([[formulation-stop-rule]]), the user to confirm. See
[[resume-2026-09-30-r11d]] for the state and the runtime / parked items.

Final picture aimed at: L = [S_ε + L_surf] + λ L_render + L_stab + L_cleanup, constants λ (calibrated), the render's
sil_k / w_hole / w_spray / w_pbr; the transport grid follows N (the coarse grid lost at 300k, R10).

See [[no-parameter-only-fixes]], [[thin-root-cause-2026-09-30]], [[parked-defects-2026-09-30]].
