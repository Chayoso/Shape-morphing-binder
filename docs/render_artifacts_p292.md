# P292: current-density support and resolution-scaled normal filtering

This is an appearance-only comparison on the saved original mixed60 trajectory.
It does not change particle positions, repair physical holes, or settle unpinned
particles. Both changes are opt-in; the historical renderer defaults remain.

The first change replaces each hard neighbor indicator by a compact smoothstep:
`q=clamp((r_cov-d)/h,0,1)`, `w=q*q*(3-2*q)`,
`support=clamp(sum(w)/4,0,1)`. The same eight current nearest neighbors are used,
self excluded. `r_cov` remains the target median eighth-neighbor distance and
`h` is exactly one target median nearest-neighbor spacing. No temporal history,
opacity renormalization, enlarged radius, or frozen support is used. Consequently
support never exceeds the old hard support for the same positions. Departed
neighbors have zero contribution at and beyond the original radius. A sparse
region may become more transparent; this method cannot supply missing material.

The second change scales only the image-normal filter. Its odd footprint is
`2*max(1,floor(height/1080+0.5))+1`, giving 3 pixels at 1080p and 5 at 2160p.
Premultiplied normal and coverage buffers are averaged to estimate the normal,
but final compositing still uses the original unfiltered coverage. This addresses
the smaller angular normal-filter footprint of the old fixed 3-pixel 4K filter.
It is not a change to Gaussian radii, opacity, material roughness, or illumination.

`SettledAppearance` still validates pinned centers and freezes their normals and
radii. Density support remains live. The standard renderer rejects combining
smooth support with its historical frozen `material_support` option.

## Controls and comparison

`scripts/render_splat_photoreal.py` accepts `--smooth-support` and
`--scale-normal-filter` separately. `--compare-artifacts` produces four videos:
baseline, support only, normal filter only, and both. Every variant uses the same
positions, particle normals, covariances and pin state calculated once per raw
frame. The standard `render_splat_gpu.py` exposes the same two choices as
`--smooth_support` and `--scale_normal_filter`.

The 4K comparison uses the original bunny N=300000, dt=1/240, T=20,
dx=0.3062907544 wu and loss grid 36 cubed. The same 67 raw indices, stride12,
20fps, camera35/18 degrees and synthetic satin-ceramic material are retained.
Rendering uses an isolated copy of the successful original render snapshot;
ongoing physics changes cannot enter this comparison. Metadata records the exact
spacing, coverage radius, transition width, flags, filter footprint and code hashes.

The measured same-frame coverage differences compare each variant to that frame's
baseline. Temporal statistics use unencoded sRGB mean absolute difference and
unit image-normal vector L2. One shared mask, baseline coverage at least0.5 in
both consecutive frames, is used for every variant. These include real motion,
occlusion and opacity changes; they are not an isolated vibration metric. Their
denominator is reported. The final raw780-to781 held pair must be read separately.
Coverage comparisons do not replace raw physical-density measurements.

## Validation status

Compile checks and fifteen CPU tests passed (including the four preserved
historical MaterialSupport regressions): compact support continuity, monotonicity,
support no greater than the hard baseline, zero contribution outside the radius,
pin normal/radius invariance with live support, rejection of moving pinned centers,
normal filtering without coverage mutation, and historical default equivalence.
These test Torch algebra on CPU; the renderer uses CUDA tensors and no CPU fallback.

The pre-run adversarial review caught accidental replacement of the historical
`MaterialSupport` API. Its implementation and four original tests were restored
unchanged before execution; the new Torch helpers coexist with them. A returned
coverage-buffer shape issue was also corrected and covered by a CPU regression.
The reviewer then cleared the bounded four-variant GPU comparison.

## Measured result: smooth support rejected for promotion

The comparison ran on hyde06 GPU2, 2026-09-26 22:43:38 to22:45:27 UTC,
in107.66 measured seconds including four NVENC encodes. All782 raw pin frames
passed validation. Every output is H.264, 3840x2160, 67frames,20fps and3.35seconds.
The exact renderer target spacing is0.03493081033229828wu; transition width is
the same. The original source NPZ SHA256 remains
`7896e4fcb5f95020c559c3fb12a570c1e2106c062f0c2887c230986f2aee336e`.

The table averages per-transition means across the25 pairs fully inside raw480
through780. Each pair uses the same baseline-derived pixel mask in all variants.
The held780-to781 pair is excluded. Coverage loss pools pixel counts over the26
snapshots in that interval: pixels with baseline alpha>=0.5 and variant alpha<0.5,
divided by the baseline alpha>=0.5 count. This is an image-opacity threshold,
not a physical hole fraction.

| Variant | Temporal RGB MAE | Temporal unit-normal L2 | Lost baseline coverage |
| --- | ---: | ---: | ---: |
| Baseline | 0.00074772 | 0.00669363 | 0 |
| Smooth support | 0.00175653 | 0.01258217 | 2.9127% |
| Scaled normal filter | 0.00073309 | 0.00655048 | 0 |
| Both | 0.00174830 | 0.01246403 | 2.9127% |

Smooth support increases the late RGB difference by2.35x and unit-normal
difference by1.88x. Its worst frame loses7.209% of baseline alpha>=0.5 pixels
(raw180); the largest individual coverage decrease over the sequence is0.9271.
Visual inspection confirms a more porous/translucent growth front and more
surface mottling. The one-spacing transition width therefore fails the intended
quality goal. It remains opt-in diagnostic code and is not promoted as a fix.

The scaled normal filter preserves the unencoded coverage buffer exactly in all67
frames. Late RGB differences decrease1.96% and normal differences2.14%; this is
a small shading effect, not proof that physical oscillation is suppressed. It
does not repair the existing sparse growth front or rounded geometry. The
historical defaults remain unchanged.

Per-particle support never increases: the measured maximum increase is exactly0.
However, the approximate rasterizer's coverage is not pointwise monotone here:
support variants show maximum positive pixel coverage difference0.00114715.
The installed diff_gauss source (`cuda_rasterizer/forward.cu:414-426`) caps alpha
at0.99 and stops before accumulating the splat that would take transmittance
below0.0001. Lower opacity can postpone this skipped contribution and accumulate
farther. This supplies a plausible mechanism for the small positive difference;
no matching-kernel counterfactual was run, so it is not established attribution.
The guaranteed claim is input-opacity nonincrease, not strict pixelwise coverage
nonincrease. No raster kernel or opacity compensation was changed.

On the held780-to781 pair, baseline masked RGB difference is2.57e-9 and normal
L2 is1.50e-8; other variants are similarly near numerical precision. This control
does not remove actual motion from the other pairs or measure encoder flicker.

## Visual QA and artifacts

All268 encoded frames were decoded and visually inspected in12 paired contact
sheets. Native4K head crops at raw96,228 and780 were inspected additionally.
No clipping, corrupt frame or added crossfade appeared. There is no texture to
claim as material-bound. Support/both visibly worsen sparse regions; filter-only
retains the baseline silhouette and its limitations. These diagnostic renders
do not certify a closed physical solid or all-particle rest.

Local files are under `output/c291/photoreal_p292/`:
`mixed60_p292_baseline.mp4`, `mixed60_p292_support.mp4`,
`mixed60_p292_normal_filter.mp4`, and `mixed60_p292_combined.mp4`.
`mixed60_p292.comparison.json` retains every frame and metric denominator;
`comparison_summary.json` records the aggregation definitions;
`visual_qa.json` and `artifact_verification.json` record every inspected frame,
ffprobe results and video SHA256 hashes. Source/snapshot hashes and the complete
render log are preserved alongside the outputs. The original4K video is unchanged.

Final artifact/report adversarial review passed. The reviewer independently
checked all four video hashes, report/raw-JSON agreement and native head crops,
and relied on the implementer's recorded all268-frame QA for the full sequence.

The measured failure is consistent with loss of the old saturation plateau:
the transition width0.03493wu is roughly half the coverage radius0.06899wu.
At raw780 mean particle support falls from0.990996 to0.524800. Neighborhoods
formerly saturated at support1 now respond continuously to many changing
neighbor distances, and lower opacity reveals deeper splats/background. A
continuous formula therefore need not reduce image differences. This experiment
rejects this width and formula as a default, not every possible continuous
current-density estimator.

## Material-transported shading: also rejected for promotion

The second comparison changes shading normals only. Positions, covariance,
radii, live hard support, opacity, camera, material and the original3-pixel
image-normal filter are shared with its baseline. The optional flags are
`--material-shading` and `--compare-material-shading`; they reject combination
with the support/filter experiment. Historical defaults remain unchanged.

At the first exposed selected/rendered frame, the renderer anchors32 fixed
neighbor IDs, their full3D offsets and the current shading normal. It fits
`J=(D_current^T D_rest)(D_rest^T D_rest)^-1` and carries the anchor normal by
the cofactor ofJ, normalized. There is no sign flip toward the previous normal.
A full3D rest eigenvalue ratio above1e-4, positive relative determinant above
1e-4, nondegenerate carried normal and relative fit residual at most0.5 are
required. Invalid fits return the current refit and may reanchor an exposed,
valid neighborhood. They never retain a stale normal. Rank-deficient rest
neighborhoods remain on the current refit. A separate validated appearance latch
freezes admitted pinned shading normals. No unrendered raw-frame updates occur.

The pre-run review rejected reusing the old2D tangent-fit residual: comparing
that fit with full3D offsets falsely penalizes unchanged volumetric thickness.
The new full3D fit has explicit rank validity instead of a unit-dependent ridge.
CPU validation passed20 tests:18 in `tests/test_render_support.py`, including
the four historical regressions, and2 in `tests/test_render_splat_photoreal.py`.
New checks cover identity, a120-degree rigid rotation, shear, inversion fallback,
planar rank rejection and active-pin normal invariance. The reviewer independently
passed all18 support tests and cleared the bounded GPU comparison.

The comparison ran on hyde06 GPU0 from2026-09-26 23:00:04 to23:01:12 UTC,
in65.72 measured seconds including both encodes. It uses the identical source
NPZ, discretization and67 raw indices described above. All782 raw pin frames
passed validation. Both outputs are H.264,3840x2160,67frames,20fps,3.35seconds.

| Pair interval | Baseline RGB MAE | Material RGB MAE | Baseline normal L2 | Material normal L2 |
| --- | ---: | ---: | ---: | ---: |
| raw0..240,20 pairs | 0.00854286 | 0.00808387 | 0.07815490 | 0.07553032 |
| raw240..480,20 pairs | 0.00146526 | 0.00211835 | 0.01323330 | 0.01903850 |
| raw480..780,25 pairs | 0.00074772 | 0.00118670 | 0.00669363 | 0.01046468 |

The same baseline-derived pairwise mask and mean aggregation used in the first
comparison apply. Material transport improves early RGB differences5.37%, but
worsens late RGB58.7% and normal L2 56.3%. Every same-frame unencoded coverage
buffer is exactly equal to baseline in all67 frames: maximum increase/decrease
and lost-alpha>=0.5 pixel count are all0. On held780-to781, material masked RGB
difference is3.07e-9 and normal L2 is1.77e-8. These are image differences, not
physical-motion or hole metrics.

The change was active rather than merely falling back. The fixed upper-ear
cohort consists of10872 material IDs with raw480 y>=2.40357423wu (target lower
bound plus75% of its height). The table pools particle-frame counts over the26
snapshots raw480..780; the three total fractions sum to100% before rounding.
The attempted-fit column has a different denominator and excludes already
latched pins and unanchored points. These status masks are recorded before the
current frame's pin admission: newly admitted pins can appear under transported
or current-refit for that frame, then use that normal as their frozen value.

| Region | Transported / total | Current refit / total | Previously latched / total | Valid / attempted fit |
| --- | ---: | ---: | ---: | ---: |
| All300000 particles | 12.09% | 3.79% | 84.11% | 99.63% |
| Upper-ear10872 IDs | 65.91% | 5.20% | 28.89% | 99.78% |

No rest-rank rejection occurred. There were3476 invalid fits globally and411
in the upper-ear cohort during that late interval, with current-refit fallback.
This establishes that widespread fallback/no-op cannot explain the negative
result; it does not allocate pixel changes among individual causes. A valid
material affine fit need not track the normal of a newly exposed density
surface. The transported normals can also differ across overlapping splats
whose covariance still follows the baseline density estimate. Both are plausible
sources of the observed shading changes, not causally isolated diagnoses.

All134 encoded frames were decoded and visually inspected across12 paired
contact sheets, with native4K head crops at raw96,228,480 and780. Coverage and
silhouette remain unchanged, including the sparse early fringe. The material
variant has more mottled/rippled highlights on the ears, head and feet; no
clipping, corruption or added crossfade appeared. It is rejected as a quality
improvement. This experiment does not establish physical hole removal or
unpinned particle rest, and no texture-binding claim applies to uniform material.

Artifacts are under `output/c291/photoreal_p292_material/`:
`mixed60_p292_material_baseline.mp4` and
`mixed60_p292_material_material_shading.mp4`, plus complete metadata,
per-frame comparison JSON, aggregated counts/definitions, snapshot/source hashes,
render log, decoded frames and visual/ffprobe/hash QA records. Original delivered
videos are unchanged. Final artifact/report adversarial review passed: the
reviewer verified both MP4 hashes, independently recomputed the25 late-pair
means and all67 zero coverage deltas, checked the status denominators and native
raw96/480 crops, and relied on the recorded implementer QA for all134 frames.
