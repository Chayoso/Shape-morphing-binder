# Measurement probes for settled-base

Server-side tools (run on hyde06 with `PYTHONPATH=$REPO`). They read a run's archive
`<tag>_render_full_dt_iso_nn.npz` and, where noted, its run JSON. Tags given to `hollow_frames.py` and
`strip_probe.py` are paths relative to `/data/relcfd/chayo/physmorph_v2/output/`, e.g. `settled/s1_bunny`.

| Probe | What it measures |
|---|---|
| `end_probe.py NPZ` | End frame: pieces off the body, the ear tip's particle count in reference particles, target under-fill. |
| `region_error.py NPZ...` | End-state distance to the target by region (ears, head and upper back, lower body): target surface to the nearest particle, share beyond 1.5 target spacings. |
| `hollow_frames.py TAG F1,F2,...` | Density of the top region (y > 2.3) over the target's at fixed trajectory frames: the transit "vapour". |
| `strip_probe.py TAG` | Progress toward the end position by source depth (skin versus bulk) and the late surface motion, pinned and unpinned. |
| `ear_slab.py NPZ t1,t2,...` | The ear's fill and thickness per height slab over time: how the ear grows. |
| `layer_probe.py LABEL NPZ REF PLAN` | Delivered motion by particle layer for replayed windows. |
| `window_reversal.py NPZ...` | Window-to-window reversal of the outer layer's displacement (breathing), from the accepted history. |
| `com_probe.py NPZ...` | Centre-of-mass path over the whole morph: momentum conservation, since source and target are centred. |
| `video_flicker.py MP4 THR N` | Frame-to-frame change of a rendered video; the tail value is the visible late flicker. |
| `gradcheck_patch.py OPTIMIZER` | Patches a COPY of the optimizer with a first-iteration hook: finite differences against autograd for the physics, render and cleanup terms, the render share per control leaf, and a momentum and position-edit budget of the rollout. Read the JSON with `gc_read.py` and `gc_mom.py`. |
