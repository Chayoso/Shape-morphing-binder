# Render influence

Observational optimizer telemetry, not causal attribution. Norm shares exclude post-combination transforms and transport addends. Compare matched raw trajectories with render disabled for causal evidence. Image losses do not certify physical holes/rest.

Discretization: `{"N": 300000, "T": 20, "dt": 0.004166666666666667, "dx_wu": 0.3062907543956724, "loss_res": 36, "render_res": 64, "iters": 2}`.

Committed windows: 1/1; inner accepted steps in committed windows: 2 (2 with detailed telemetry).

| Observation (median [min, max]) | Value |
|---|---|
| Adaptive render lambda | 0.0148821 [0.0148821, 0.0148821], n=1 |
| Nominal render direction share | 0.277344 [0.221355, 0.333333], n=2 |
| First-iteration share (legacy fallback) | 0.333333 [0.333333, 0.333333], n=1 |
| Observed render loss change per accepted update | -0.257479 [-0.499867, -0.0150916], n=2 |
| Optimizer endpoint update RMS (wu; not velocity) | 0.134567 [0.0986747, 0.17046], n=2 |

Matched render-off causal ablation: not measured by this report.

- body: nominal share 0.264674 [0.206961, 0.322387], n=2; control delta norm 45.2028 [38.7324, 51.6733], n=2.
- stress: nominal share 0.379426 [0.326446, 0.432406], n=2; control delta norm 33.4984 [20.7388, 46.2581], n=2.
- surface_u: nominal share 0.566356 [0.50408, 0.628632], n=2; control delta norm 0.528828 [0.481345, 0.576311], n=2.

Last committed window, last inner accepted image observations:

| Component | Value |
|---|---|
| coverage | 0.020636192 |
| detail_coverage | 0.029709049 |
| detail_edge | 0.00031850589 |
| total | 0.050663747 |
| weight | 1 |
| coarse_render | 0.032958809 |

Component scales/targets differ across policies; image errors do not certify physical coverage/rest.
