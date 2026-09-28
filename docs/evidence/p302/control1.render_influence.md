# Render influence

Observational optimizer telemetry, not causal attribution. Norm shares exclude post-combination transforms and transport addends. Compare matched raw trajectories with render disabled for causal evidence. Image losses do not certify physical holes/rest.

Discretization: `{"N": 300000, "T": 20, "dt": 0.004166666666666667, "dx_wu": 0.3062907543956724, "loss_res": 36, "render_res": 64, "iters": 2}`.

Committed windows: 1/1; inner accepted steps in committed windows: 2 (2 with detailed telemetry).

| Observation (median [min, max]) | Value |
|---|---|
| Adaptive render lambda | 0.327159 [0.327159, 0.327159], n=1 |
| Nominal render direction share | 0.338261 [0.333333, 0.343189], n=2 |
| First-iteration share (legacy fallback) | 0.333333 [0.333333, 0.333333], n=1 |
| Observed render loss change per accepted update | -0.0119187 [-0.0220608, -0.00177649], n=2 |
| Optimizer endpoint update RMS (wu; not velocity) | 0.128031 [0.0865575, 0.169505], n=2 |

Matched render-off causal ablation: not measured by this report.

- body: nominal share 0.326051 [0.320639, 0.331464], n=2; control delta norm 43.8286 [35.9368, 51.7203], n=2.
- stress: nominal share 0.422023 [0.409712, 0.434335], n=2; control delta norm 33.2792 [19.9063, 46.6521], n=2.
- surface_u: nominal share 0.787882 [0.778518, 0.797245], n=2; control delta norm 0.509774 [0.440703, 0.578845], n=2.

Last committed window, last inner accepted image observations:

| Component | Value |
|---|---|
| coverage | 0.01725846 |
| detail_coverage | 0.031657048 |
| detail_edge | 0.00033312591 |
| total | 0.049248632 |
| weight | 0 |
| coarse_render | 0.027277408 |

Component scales/targets differ across policies; image errors do not certify physical coverage/rest.
