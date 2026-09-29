# Experiments

Append-only log. Stamps are the GPU server's clock (CDT). Each entry states its prediction and gates before the
result. The full record of the earlier line (2026-09-03 to 09-29, thousands of entries) is `docs/experiments.md` on the
`v3-grid-gs` branch; this file starts with its summary.

## Summary of the earlier line (v3-grid-gs)

The earlier line kept the release's windowed method (score the driven end state against a paced target that
advances one loss cell per window) and added rules around it.

- **Thin features and transit haze.** At 300k particles the ears grew as a thin, sparse haze before filling. The cause
  was measured end to end: the density cell-sum loss cannot see a filled region translating (its gradient lives on the
  density jump layer), the paced target led the body by a whole cell while a window delivered about a tenth of it,
  and the loss filled that lead with the outermost one or two particle layers, which the MPM moves 2–6× faster than
  the bulk under a stress control. Refuted levers: a finer loss grid (it observes less, its floor is the sampling
  noise), a fixed small lead (fills the haze on the bunny, stalls long transports), a tested-candidate lead governor,
  a surface-tapered control, neighbourhood correspondence terms.
- **Surface oscillation.** Window-to-window reversal of the outer layer (the "breathing") was stopped with per-particle
  step control (Rprop) and pins that hold settled particles. Pins and the commit-time position edits keep the surface
  still but move the body outside momentum (centre-of-mass drift 0.6–0.8 spacing over a morph).
- **Rendering.** Mesh reconstructions (Poisson, marching cubes) shimmered frame to frame; the deliverable became a
  Gaussian-splat renderer (quick two-view and 4K PBR). Visible haze in 4K frames comes from sparse material and the
  splat radius growing up to 4× to cover it, not from `F` in the covariance, which the renderers do not use.
- **Render influence.** The per-window balancer held the render share of the control update at about one third.

## 2026-09-29 — B1: settled transport against the earlier line (300k bunny, seed 97, paired)

| | settled | earlier line | release legacy |
|---|---|---|---|
| silhouette IoU / chamfer | 0.9851 / 0.0581 | 0.9769 / 0.0556 | 0.9646 / 0.0594 |
| wall / windows | 7 min / 31 | 12 min / 47 | 28 min / 183 |
| det F min / stray particles | 0.934 / 0 | 0.608 / 30 | 0.581 / 40 |
| progress by depth at t = 0.10 (skin / 2–5 sp / 5–10 sp) | 0.16 / 0.13 / 0.13 | 0.03 / −0.04 / −0.03 | 0.30 / 0.20 / 0.13 |
| transit density at frames 76 / 114 | 0.36 / 0.23 | 0.16 / 0.09 | 0.19 / 0.18 |
| ear tip (reference particles; gate 13) | 11.1 | 8.8 | 4.3 |
| target surface beyond 1.5 spacings (ears / head / lower body) | 11.7 % (13.3 / 10.0 / 11.7) | 7.8 % (14.3 / 7.2 / 6.7) | 23.5 % |
| late surface motion (spacings per frame) | 0.007 / 0.009 | 0.128 / 0.113 | 0.049 / 0.080 |
| window-to-window reversals (whole / second half) | 12 % / 23 % | 0 % / 0 % | 70 % / 100 % |
| splat video tail (D1) | 0.0003 | 0.0008 | — |

With a longer budget (`--reject_stop 20 --patience 20`, 68 windows, 13 min) the settled run reaches ear tip 12.1 and a
10.4 % surface gap; after convergence the same rejected candidate repeated each window until the plateau stop. The
release's legacy method oscillates for its whole second half, which is why its surface is lumpy.

## 2026-09-29 — B2: gradients, render influence, momentum (settled, 300k)

- The branch's tests pass. The gradient path and the line-search path agree to seven digits; finite differences
  agree with autograd within 3 % for the physics, render and cleanup terms at windows 1 and 20.
- The render share of the control update grows from 0.33 (window 1) to 0.90 (window 20); on `u` it is 0.96–0.97
  because the physics gradient on `u` is nearly zero. The render and physics gradients do not conflict (cos +0.14
  to +0.26).
- Render-off twins against a second seed:

| | render on | render off | seed 98 |
|---|---|---|---|
| settled: silhouette IoU | 0.9851 | 0.9672 | 0.9828 |
| settled: surface beyond 1.5 spacings | 11.7 % | 23.5 % | 11.7 % |
| earlier line: silhouette IoU | 0.9769 | 0.9716 | 0.9777 |

  The render's effect on the settled outcome is eight times the seed spread.
- Momentum: see docs/method.md, section 6. Over the whole morph the centre of mass moves ≤ 0.02 spacing in every
  settled run.

## 2026-09-29 11:40 CDT — S1: this branch

`settled-base` = `michael/settled-transport` (835af64, on the release 680622e) plus, from `v3-grid-gs`, the 4K PBR and
quick splat renderers with their modules and tests, the measurement probes and the server environment. Changes to
the method's own files: the renderers accept archives without pins; `--render_weight_scale` (the render-off twin) and
`--reject_stop` are exposed. With the defaults the pipeline behaves as Michael's commit. Tests: 207 passed, 2
skipped.

## 2026-09-29 11:45 CDT — S2 pre-registered: adoption gates

Seed 97. (a) The 19-mesh gallery at 40k (A, armadilo, beast, bimba, bob, bunny, cheburashka, C, cow, dragon,
fandisk, heart, homer, maxplanck, nefertiti, ogre, spot, teapot, V): settled against the earlier line's adopted 40k
form. (b) The 300k dragon: settled against the earlier line's 300k form. Gates, per target: silhouette IoU ≥ the
earlier line − 0.005; no stall; second-half reversal share ≤ 25 %; centre-of-mass drift ≤ 0.1 spacing; the dragon's
silhouette IoU ≥ the earlier line's − 0.003. Table: `scripts/probes/settled/gate_table.py`. Result pending.
