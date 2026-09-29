# Render influence

Observational optimizer telemetry, not causal attribution. Norm shares exclude post-combination transforms and transport addends. Compare matched raw trajectories with render disabled for causal evidence. Image losses do not certify physical holes/rest.

Discretization: `{"N": 300000, "T": 20, "dt": 0.004166666666666667, "dx_wu": 0.3062907543956724, "loss_res": 36, "render_res": 64, "iters": 8}`.

Committed windows: 20/20; inner accepted steps in committed windows: 160 (160 with detailed telemetry).

| Observation (median [min, max]) | Value |
|---|---|
| Adaptive render lambda | 0.0612091 [0.027195, 0.308639], n=20 |
| Nominal render direction share | 0.495038 [0.333333, 0.660366], n=160 |
| First-iteration share (legacy fallback) | 0.416185 [0.333333, 0.574031], n=20 |
| Observed render loss change per accepted update | -0.000138977 [-0.0231346, 0.00249722], n=160 |
| Optimizer endpoint update RMS (wu; not velocity) | 0.00561609 [0.0010372, 0.16976], n=160 |

Matched render-off causal ablation: not measured by this report.

- body: nominal share 0.350483 [0.193483, 0.509972], n=160; control delta norm 4.913 [1.26541, 51.7142], n=160.
- stress: nominal share 0.542391 [0.279687, 0.788544], n=160; control delta norm 1.44713 [0.224059, 46.5899], n=160.
- surface_u: nominal share 0.734622 [0.537108, 0.949988], n=160; control delta norm 0.153793 [0.0682726, 0.716501], n=160.
