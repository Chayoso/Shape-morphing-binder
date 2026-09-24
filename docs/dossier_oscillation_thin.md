# Dossier — the two defects: the tail's oscillation and the thin feature that moves like droplets

Status document (2026-09-24 evening). By the user's directive the deliverable documents are
organised around these two defects only; `docs/experiments.md` remains the append-only lab log
(dated entries with the pre-registrations P1–P144), `docs/method.md` §10.17a–10.24 the
formulation. Everything measured here is on the 300k bunny unless a target is named; the 40k
gallery (19 targets) is the generalisation check the user requires for any fix.

## 1. The defects

**D1 — the oscillation.** After the body has arrived the morph "breathes": the outer layer's
per-window normal step is 0.002–0.003 wu (3–4 % of a spacing, 0.8 % of a cell) and flips sign at
~70 % of the layer every window, at 40k and at 300k alike; the bulk's consecutive window
displacements are anti-correlated (cos −0.3 … −0.6). Played at 20–30 fps a sign flip every window
is a 10–15 Hz low-spatial-frequency modulation, inside the eye's flicker peak (Kelly 1979):
invisible as geometry, visible as flicker.

**D2 — the thin feature that moves like droplets.** The ear grows from the sphere's crown as a
pile at its base (1.5–1.7× the target's mass) feeding a filament at 0.5–0.7 of the target
thickness that breaks into 50–170 native-spacing pieces (2–4 of them ≥ 20 particles: the
"droplets") which merge after t ≈ 0.4; and in the delivered video the tongue's leading edge is
reconstructed in one frame and not the next (1–2 s), so the tip seems to vanish and re-form.

## 2. What is established (measurements)

### D1
| Measurement | Reading | Meaning |
|---|---|---|
| M2 — the reconstructed surface's own normal velocity between commits vs the layer's normal step (Stam–Schmidt 2011) | v_n rms / d·n rms = 0.91–0.98, six tail windows | the surface moves exactly as the layer does: the breathing is genuine (a), not a re-sampling of a stationary surface (c) |
| M1 — tail-only twins whose layer moves by the normal or the tangential part only (stride 4) | full 0.0013, normal-only 0.0013, tangential-only 0.0009 per frame | the normal part reproduces the visible change; a refit-response floor of the same size coexists |
| M3 — the Poisson leaf widened to the MPM cell | tail change 0.0022 vs 0.0024, bump 1.24° → 1.06°, more pinch-offs | band-limiting the reconstruction changes nothing: not a sub-leaf re-sampling |
| codec floor (20 identical hold frames) | 0.0000 | the encoder contributes nothing |
| alternating vs drift split of the per-frame change (stride 12 / 19) | ALT/DRIFT 1.2–2.2, white noise = 1.73 | frame-to-frame change is spatially incoherent motion; the alternating part is the number to track (stride 19: l300 0.0011, z300b 0.0021) |
| the rebound probe (zero-control rollout from each commit) | +0.3 … +0.9 of a window carried, elastic part a tenth | carried momentum travels on; refuted as the carrier by o300 (windows from rest reverse just the same) |
| the reversal cosine (consecutive accepted windows) | +0.9 in transport, negative from arrival to the end (40k from window 20, 300k from ~50) | the merit alternation begins at arrival; the layer breathing exists in every phase |
| the optimiser's global step in the tail | α at its floor 0.001 (anneal 0.05) in l300 / g41 | the alternation lives at the floor step |

Refuted as the carrier of D1 (one intervention twin each): the carried momentum (o300, m300b),
the render channel (p300: reversal unchanged with the render off), the layer relaxation (w300:
flips 0.94 without it — it damps), the step size (at its floor already), the balancer (λ, g_share
flat), the u bound (j300), the XPIC commit projection and the shifting (the 40k reference has
neither), the moving paced target (v300: the bulk's merit reversal decays, the layer's does not).

### D2
| Measurement | Reading | Meaning |
|---|---|---|
| ear_slab (0.3-wu slabs from the ear base; fill and xz thickness vs the target) | base slab 1.49–1.74× mid-growth; mid-ear thickness 0.40–0.86; pieces 43–177 (300k); 40k: 0.99–1.02×, 0.80–1.01, ≤ 10 pieces | the pile-and-filament growth is the 300k defect; the 40k tongue does not form a sub-cell filament |
| c300 (native constants) vs d300 (`--disc_ref`) | c300: no pile, the ear never fills (0.67); d300: the pile appears, the ear fills (0.93) | the reference discretisation brought both the filled ear and the pile |
| the plan blur under `--disc_ref` | 0.227 wu instead of the sample's 0.116 (a double count against rule 38) | corrected (`--plan_native`, u300); not the pile's cause |
| the ear-tip census (frames 192–324, one Poisson mesh each) | tip particles 47 → 108 monotone, all surfels; mesh one component; the mesh captures 3–27 of them, jumping frame to frame | the particles never retract; the fit passes below a sparse tip by a varying amount |
| Kazhdan's PoissonRecon on the frame's surfels (samplesPerNode 1 / 1.5 / 5 × pointWeight 0 / 4) | tip capture 0.00–0.10 at every setting | a filament 2–3 native spacings across (0.13–0.19 wu) is narrower than the finest node (0.10) and the B-spline support (0.31): unrepresentable at the reference resolution |
| render fallbacks (`--thin_fallback spheres` / `level`) | spheres draw the tip as a bead chain (11–25 spheres, body untouched); the level set finds nothing (the tip's density is below iso) | the edge IS a bead chain at this resolution; the render can hide it or show it, not smooth it |

## 3. Mechanisms

### D2 — adopted for N > 40k (opt-in), the ear item answered at the target level
- **§10.23 the particle-scale density term** (`--w_kde 1`, at equal gradient norm with the cell
  sum): the cell sum cannot tell a sparse cell from a dense filament in part of a cell; the KDE
  match of particles against the target's own points sees a filament surrounded by empty target
  volume as a deficit. y300: base 1.49× → 1.11×, the tongue at 0.90–1.05 of the target thickness
  where filled, tip 16.1 reference particles, 4 strays, silIoU 0.978; price: the ear fills later
  (0.31 at t = 0.2), det F 0.59, the leading edge still fragments.
- **`--plan_native`** (the plan blur at the sample spacing): a correctness fix of rule (38).
- Refuted: the support-preserving (projected) paced target (§10.22, n300 — the cell sum is
  blind to incompressible interior flow; kept opt-in for its end state), the linear cell sum
  (t300), the sampling/screening of the fit (the PoissonRecon probe), the component filter
  (`--keep_attached`, R-1: not the tip's mechanism).
- **Open — the leading edge's coherence (the droplets at the tip)**: physics-side, the sheet-aware
  splitting at the commit (Ando, Thürey, Tsuruno 2012; particles inserted in the sheet plane where
  the in-plane gap exceeds two spacings) or a codimensional carrier (Jiang 2017 / Wang 2020);
  render-side only the faithful bead chain (opt-in) exists.

### D1 — in test: the per-particle Rprop on the control step (§10.24)
The amplitude of the alternation is the optimiser's floor step; a step that keeps shrinking on
reversal converges (Robbins–Monro), and it must be per particle so that a part still in transport
keeps its step (a global decay froze the ear; the onset gate cut it short, z300).

| Form | Run | flips / step (wu) / low-band corr | det F | silIoU | video tail (stride 12; 40k value 0.0013) | Reading |
|---|---|---|---|---|---|---|
| reference | l300 | 0.74 / 0.0017 / −0.67 | 0.66 | 0.980 | 0.0016 | — |
| per particle | ac300 | 0.41 / 0.00095 / +0.04 | **0.39** | 0.979 | **0.0013** (stride-19 ALT 0.0007 vs 0.0011) | the alternation gone, the tail at the acceptance value; neighbouring particles' control updates differ by orders of magnitude → compression |
| smoothed, k ≈ 60 (coherence kNN) | ad300 | 0.54 / 0.0012 / −0.31 | 0.64 | 0.979 | 0.0015 | det F kept, the decay diluted |
| smoothed, k = 8 (the regulariser's kNN) | ag300 | 0.68 / 0.0017 / −0.32 | 0.66 | 0.978 | 0.0019 | the scales decayed to 0.05 and the layer breathed as before → the global step α had re-inflated 0.001 → 0.005–0.008 (the anneal's ×1.15 recovery) |
| + arrival gate (halve only arrived particles) | g41t (40k gallery) | — | C 0.53 | C 0.90 → 0.95 | — | C transports past its old stall; C's det F to read |
| + held global step (`--ctrl_rprop_hold`) | ai300 (running) | P142: ≤ 0.55 / ≤ 0.001 / > −0.2 | P143 ≥ 0.6 | ≥ 0.977 | P144 ≤ 0.0013 | the decisive form: one step control, per particle |
| combos with the KDE ear | ab300 (per particle), af300 (k ≈ 60), ah300 (k = 8, running) | ab300 0.58 / 0.0009 / −0.18 | 0.19 / 0.23 | 0.978 / 0.977 | 0.0015 | the best ears (0.965 / 0.966) and tails; det F collapses without the held step |

Also available as a deliverable rule (not a cure): the outer gate armed at the alternation's onset
(`--outer_latch_reversal`, r300b: no accepted reversal in the deliverable, the stop at 48 windows)
— incompatible with the slow KDE ear (z300) until an ear-aware onset exists.

## 4. Generalisation (the user's rule: every fix on the whole gallery)
The 40k gallery, 19 targets, the smoothed Rprop (g41s) against g41: silIoU up on 17 (+0.0002 …
+0.0099), flips down on 15, windows ≤ 1.2×; failed on C (−0.011: a legitimate direction change on
the curved path round the hole was read as an overshoot) and homer (det F −0.06). With the arrival
gate (g41t, first half): homer recovered (−0.011), C rose to 0.950 (from a 28-window stall to 113
windows) with det F 0.53 — the compression's place is the next reading. A second 300k target
(dragon: d300 silIoU 0.970 / det F 0.74 with the z300b recipe; dr300 with the arrival-gated
smoothed form) is running.

## 5. Acceptance criteria (docs/experiments.md 2026-09-23 night, restated in today's units)
| Criterion | 40k value | best 300k so far |
|---|---|---|
| layer normal step per window | ≤ 0.003 wu (0.0029) | 0.00095 (ac300), 0.0012 (ad300) |
| layer flip fraction (last 10 windows) | ≤ 0.55 | 0.41 (ac300), 0.54 (ad300) |
| video delivered tail per frame (stride 12) | ≤ 0.0013 | 0.0013 (ac300) |
| stride-19 alternating component | g41 0.0014 | 0.0007 (ac300) |
| det F min | ≥ 0.6 | 0.66 (l300), 0.64 (ad300) — the form must keep it |
| ear fill at the end / base pile / tongue thickness | — | 0.966 / ≤ 1.12× / 0.7–1.0 (ab300, af300) |
| end bump at the reference render | ≤ 1.2° (1.19°) | 1.19–1.28° |
| tip drawn in every frame | — | open (D2's leading edge) |

## 6. Next
1. ai300's verdict (the held step): if P141–P144 hold, the combo with the KDE ear at the held step
   is the 300k deliverable candidate; then the 19-target gallery with the same flags and the
   dragon twin decide adoption.
2. C's det F under the arrival gate (where the compression sits).
3. D2's leading edge: the sheet-aware splitting at the commit (physics), pre-registered on the
   census measures (pieces at the edge, tip capture, tip drawn every frame).

## 7. Addendum (19:15) — what the 40k twins settled this evening
| Twin (40k bunny) | What it isolates | Reading |
|---|---|---|
| g41h — held global step + arrival-gated smoothed Rprop | one step control, per particle | the bulk's reversal in ONE window of 36, the layer's low-band correlation −0.18, flips 0.47, det F 0.78, silIoU 0.963 (+0.002): **the alternation is gone at no cost**; the layer's per-window motion (0.019 spacings) and the video's alternating component (0.0014) unchanged |
| g41a — no plastic assimilation | the commit-time stress jump | worse (flips 0.64, step 0.036): **not the carrier** |
| g41f — freeze at arrival (control 0, velocity 0, stretch assimilated) | the persisting control | 80 % frozen by window 30, correlation +0.39, flips 0.34 — and the body still moves **0.0016 wu a window** (g41 0.002–0.004); silIoU −0.005: **the residual motion is the rollout's** (the grid carrying the last arrivals into settled material, the layer relaxation following); not adopted |
| g41t — arrival-gated gallery, 19 targets | generalisation | fit within −0.003 everywhere (C +0.049: its 28-window stall ends), flips/step improved on 15; **det F collapses on C (0.53) and beast (0.49)** at the arrived/in-transit boundary; the held-step sweep (g41u) is running |

Conclusion for D1: the optimiser's part — the window-to-window alternation — is removed by the
per-particle rule in its held-step, arrival-gated, smoothed form; a residual per-window motion of
1 % (40k) to 10 % (300k) of a spacing remains as a property of delivering simulated frames, and
the deliverable surface must not re-fit itself against it: the tracked mesh with a half-spacing
projection band (and the window-time moving average) is the pre-registered deliverable-side fix
(P151–P153, rendering now).

## 8. Addendum (20:30) — the physics levers exhausted at 40k; the 300k held-step reading
On top of H (held step, arrival gate, smoothing k = 8) five more levers were tried at 40k, each
its own twin: windows from rest, the onset gate, the fixed target from arrival, the settled
body's viscosity (η = 1/(T dt)), the settle-at-commit rollout. All give the same tail — flips
0.46–0.49, step 0.02 spacings, low-band −0.1 … 0.0, stride-19 alternating component 0.0014
(= the reference's) — with fit 0.964–0.965 (≥ g41's 0.961) and det F 0.76–0.79. The residual
jitter is the floor of delivering simulated frames; the physics side is closed with H (and the
freeze as the fit-costing extreme). At 300k, ai300 (l300 + H): α held at 0.0010, the layer's
step 0.00077 wu, flips 0.50, low-band +0.11, det F 0.70, video tail **0.0011** (below the 40k
value), silIoU 0.9766, bump 1.15°. The KDE-ear combo with H (al300) is the deliverable
candidate; the visible residual is the tracked-surface question (P151–P153).

## 9. Addendum (22:25) — the generalisation of H, and what the deliverable side can add
| gallery sweep (19 targets, 40k) | fit vs g41 | det F vs g41 | flips/step improved | failed |
|---|---|---|---|---|
| g41s smoothed Rprop | 17 up, C −0.011 | homer −0.06 | 15 / 13 | C (a transport direction change read as overshoot), homer |
| g41t + arrival gate | all within −0.003, C **+0.049** (its stall ends) | C 0.53, beast 0.49 | 16 / 18 | the arrived/in-transit boundary shears |
| g41u + held step | C −0.010, beast −0.006, nefertiti −0.006, V −0.004 | all within −0.06 | 16 / 16 | a held step ends slow transports early |
| g41x + hold from the onset | 17 ok; C −0.010, **nefertiti −0.041** (29 of 90 windows) | all within −0.025 | 11 | the global onset misfires on long curved transports |
| g41y + the onset read on arrived particles | running | | | |

The per-particle halving needed the arrival gate; the global hold needs the same reading — the
onset on the arrived particles only (the paced target's own mask, no constant). At 300k: ai300
(l300 + H) tail 0.0011 / stride-19 ALT 0.0005 / det F 0.70 / silIoU 0.9766; al300 (the KDE ear +
H) silIoU 0.9771, det F 0.51; am300 (the KDE ear + H with the arrival-read onset hold) running.

Deliverable side: the tracked mesh with a half-spacing band makes the tail WORSE (an advected
mesh follows the jitter in full: 0.0036 vs 0.0024), the window-averaged tracked mesh 0.0019 with
re-mesh pops, surfel memory 0.0020 / 0.0010 (z300b / ac300), and the particle positions
averaged over two control windows (`--frame_avg 38`) 0.0010 with the tongue's tip continuous in
every frame — a cosmetic option (the frames shown are averages, the transport lagged by a
window), to be stated as such if used.

## 10. Addendum (23:20) — the user's bar is ZERO: the pin (method.md 10.27)
The user's requirement is exact: no oscillation at all, and "once optimised, lock it" (the
plasticity idea). H removes the optimiser's alternation; what remains at 40k (0.02 spacings a
window) and at 300k (~0.001 wu) is the rollout's own floor — the freeze (g41f: control zeroed,
v zeroed at commits, stretch assimilated) left the settled body moving 0.0016 wu a window,
and the viscous forms (g41v/g41w) the same. The kernel's viscosity damps only the affine
velocity C, not v, so it cannot pin. The pin is the kinematic constraint inside the forward
model (eq. 50): an arrived, twice-reversed particle keeps its position, velocity 0, affine 0,
strain unchanged, no control, no relaxation move, from that window on; it still carries its
mass to the grid, so the transport sees the settled body as a fixed obstacle. Nothing is added
to the material: the delivered object responds to external forces through λ, μ alone, the pin
exists only while the morph runs (the user's viscosity concern answered).

Twins launched 23:20 (GPU 1): g41p (40k, the g41y form + pin), ap300 (300k, ai300's form +
pin). Read from the frames alone (`scratch/pin_probe.py`: a pinned particle's frame-to-frame
step is exactly 0 in float32). Pre-registered P181–P184 (experiments.md 23:20): the pinned
fraction ≥ 0.6 at the end with no un-pinning; stride-19 ALT ≤ 0.0007 at 40k and video tail ≤
0.0008 at 300k; fit cost ≤ 0.005; det F ≥ 0.6, strays ≤ 0.3 %. The decisive split for the
user's question "is it the particles or the re-mesh?": with the pinned body exactly still, any
motion left in the video over the pinned region is the reconstruction's (the per-frame fit),
and the advect-only render of ai300 (the mesh advected by the particles, never re-fitted)
gives the same split on the un-pinned run.

## 11. Addendum (00:10) — particles or re-mesh? Answered: the particles
The user asked whether the visible motion is the particles' or the per-frame re-mesh. On
ai300 (300k), the mesh advected by the particles and never re-fitted moves 0.0015 per frame
in the delivered tail; the re-fitted mesh 0.0011. The re-fit removes motion, it does not add
it. With M1 (normal-only twin = full change) and M2 (surface normal velocity = layer step),
the residual is a genuine normal motion of the settled surface carried by the particles
(~0.001 wu a window at 300k). Physics side: the pin (§10, g41p/ap300 running). Deliverable
side: only averaging (0.0009–0.0010), cosmetic.
