# Motion versus appearance in the existing mixed60 4K video

The 4K export does not optimize Gaussian parameters. It reads the saved particle
positions and derives density normals, Gaussian disc radii
and density support at each selected frame. Active-pin normals and radii are
locked; support remains live. The saved positions include the original hybrid
physical/surface-control pipeline, not a new motion simulation during rendering.

Measurement uses the literal attribute-calculation block extracted from the
frozen renderer that produced the current 4K video. No video or appearance was
changed, and no counterfactual image was rendered.

Discretization: bunny N=300000, T=20, dt=1/240, dx=0.3062907544 wu,
loss grid 36^3. Renderer spacing is the target median nearest-neighbour distance,
0.03493081033229828 wu. Throughout this report, `sp` means this renderer spacing,
not the source-native spacing used by some physics diagnostics. All 67 raw
indices used by the video were measured; all
782 raw delivered frames passed active-pin position validation. The final saved
hold transition, raw780 to781, is excluded from aggregates.

## Late interval: raw480 to780

The fixed material cohorts are selected from positions at raw480. Head means
y>=1.191810596 wu (target bounding-box height55%, 47826 particles); upper ears
means y>=2.403574407 wu (height75%, 10872 particles). This is a reproducible height
region, not an exhaustive thin-feature classifier or a visible-pixel selection.

Each transition normally spans12 saved raw frames. The table pools individual
particle-transition samples over25 transitions. Particles newly pinned within a
transition are reported separately in the JSON and excluded from both groups
below. Quantiles are exact nearest-rank selections on GPU.

| Upper-ear property | Already pinned at both endpoints | Unpinned at both endpoints |
|---|---:|---:|
| Position displacement, median / p95 | 0 / 0 sp | 0.1230 / 0.3964 sp |
| Normal angle, median / p95 | 0 / 0 degrees | 1.024 / 5.102 degrees |
| Direct normal-vector difference, maximum | exactly0 | 1.6804 |
| Relative radius change, median / p95 | 0 / 0 | 0.279% / 7.389% |
| Fraction of particle-transition samples with support change | 1.7097% | 7.1126% |
| Number of particle-transition samples | 79019 | 190100 |

Across the whole late interval,268 of4756 upper-ear particles having an already
active pin in a measured transition had at least one support change (5.635%).
The corresponding head count is449 of27860; the full cloud count is1058 of264687.
All measured already-pinned positions, normal vectors and radii are exactly
unchanged across selected frames, including outside the ear cohort.

In the unpinned upper-ear cohort,64256 of181444 eligible three-frame samples
(35.414%) reverse direction: the dot product of consecutive displacement vectors
is negative and both displacements are at least0.01sp. This is kinematic evidence
of sampled direction changes; it does not identify a periodic oscillation or
resolve motion between the temporally sampled frames. In the final raw660–780
interval the corresponding rate is39.710%, with displacement median0.1101sp and
p95=0.3866sp.

## Interpretation limits

Actual unpinned center motion and changing render attributes coexist. This
rules out attributing all apparent movement to an independently trained Gaussian
optimizer: no such optimizer runs in this export. It does not determine how much
of an observed pixel's flicker comes from center motion versus appearance.

Support is `min(1, count_of_8_neighbours_inside_radius / 4)`, so it changes in
0.25 increments; with opacity0.92, a one-level change is0.23 in that individual
splat's opacity, not necessarily0.23 in a composited pixel. This support changes
for some already-pinned particles even when their centers, normals and radii do
not move. Pixel shading can also change through occlusion/compositing by nearby
unpinned splats and the screen-space normal-buffer smoothing. Those contributions
were not separated by this numeric diagnostic. The measurement does not establish
absence of holes or all-particle rest.

The raw780-to781 held-state pair supplies a limited numerical control: all centers,
radii and support are unchanged, while unpinned normals differ by p95=0.00000605
degrees and at most0.000598 degrees (direct vector-difference maximum1.043e-5).
The CUDA normal calculation is therefore not claimed to be bitwise deterministic.
These tiny differences are far smaller than the degree-scale changes in the
late moving-ear cohort, but this is still not a pixel-causality measurement.

## Reproduction and provenance

The original executed `measure_appearance.py` is preserved under
`output/c291/gpuwork/appearance_diagnostic/`. The tracked CLI
`scripts/probes/appearance_motion.py` reproduces its numeric procedure with
explicit path arguments and source/dependency hash checks, using the stable server snapshot
`work/gpu_refactor/render1080/repo`, the existing mixed60 raw archive and the4K
metadata JSON. The original version2 script ran on hyde06 GPU2; corrected
measurement runtime34.71s.
No new trajectory, frame sequence or high-resolution render was saved.

Authoritative output: `appearance_diagnostic_v2.json`. The first JSON is retained
as superseded evidence: its cross-product angle formula produced approximately
1e-6-degree roundoff for identical normal vectors. Version2 explicitly checks
vector equality, reports their direct L2 difference, and requires all three
frames in reversal measurements to lie within the reported interval. The
appearance recomputation itself was unchanged.
The executed script originally described the displacement unit as "native";
the authoritative JSON corrects that metadata label to the renderer target median
nearest-neighbour spacing, with numeric results unchanged. The original version2
JSON is preserved as `appearance_diagnostic_v2_raw.json`.

Source NPZ SHA256:
`7896e4fcb5f95020c559c3fb12a570c1e2106c062f0c2887c230986f2aee336e`.
Renderer SHA256:
`e7f47cb7a4c34ac6c952944cbfd0835e28dd4bb72303cea07f0ece87646a59c1`.
The JSON also records the source run, measurement script, literal attribute block,
kNN and settled-appearance dependency hashes, all raw indices, per-transition
results and aggregate definitions.


The recorded original executed script SHA256 is
`aca52ad8dd294663e85ff60c7da2bc10186814df21db5da51dc68a6450de39f6`.
The new tracked CLI is a refactor of that script and has a different file hash;
it was compile-checked, but the reported measurements were produced by the
original script. A new CLI execution records both hashes separately. Example
(on hyde06, with CUDA device selection and caches set under `/data`):

```sh
python scripts/probes/appearance_motion.py \
  --out /data/relcfd/chayo/physmorph_v2/work/gpu_refactor/appearance_diagnostic/reproduced.json
```

The CLI defaults to the original source archive, its known SHA256, the frozen
renderer snapshot and the 4K metadata. It refuses source/snapshot hash mismatches
or render metadata referring to a different archive. The numeric diagnostic was
independently reviewed for cohort selection, unit definitions, pin partition,
reversal bounds and causal overreach; no blocker remained after the wording and
unit-label corrections described above.
