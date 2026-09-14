# Sphere/bunny experiment monitor

The viewer shows the latest accepted endpoint from each independent optimization
and the common target at native 1024x1024. It is a development monitor, not a
reviewed morph animation or a 20% quality claim. Panels can be at different
optimization iterations. Image losses and validation metrics are not used to
choose which iterate to display.

The server renders the same passive material surface as the experiment. The
viewer checks paired fixture and implementation provenance, then checks its own
render dependency files and loaded native raster binary against the experiment.
It has no control or deformation variables. HTTP binds only to 127.0.0.1; the
local client connects through SSH. The browser polls once per second, and the
server checks for updated accepted checkpoints every two seconds. Optimization
itself is not real time: the first v9 iteration took about 145-150 seconds per
arm including 292 rollout evaluations (N=12000, dx=.25, dt=1/120, T=64,
144 control modes, eight 1024px training views).

Checkpoint replacement is atomic in the current benchmark writer. Older v9
writers are immutable; the viewer safely retries incomplete ZIP reads. A final
report without an iteration checkpoint falls back to the final saved trajectory,
so a run ending before its first callback does not stay labeled as waiting.
Decoded PNGs and status are published together. A completed development score is
labeled as such; it does not assert passage of sealed-test or visual gates.

The 2026-09-14 active setup is:

- Local URL: http://127.0.0.1:8776
- Server experiment: `~/physmorph_v2/output/bunny_surface_v9_20260914/`
- Server viewer: `~/physmorph_v2/output/bunny_live_v1_20260914/`
- Local tunnel PID file: `output/bunny_live_tunnel_20260914.pid`

The initial sphere and target PNGs and first accepted endpoints were inspected.
They were connected silhouettes without an isolated visible splat in this camera;
the first endpoints were not recognizable bunny morphs. All-frame review is
still required before delivering a finished animation. The images are fixed
gray, so texture transport has not been evaluated.

To reconnect from a local terminal:

```bash
ssh -N -L 127.0.0.1:8776:127.0.0.1:8776 -o ServerAliveInterval=20 chayo@hyde06.dabh.io
```

If that local port is already occupied by the existing tunnel, use the URL
directly. The earlier ellipsoid diagnostic remains separately available on 8774.
