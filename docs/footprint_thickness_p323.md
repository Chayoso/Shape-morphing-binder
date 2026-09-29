# P323: separate surface coverage radius from normal thickness

Physical F health does not certify Gaussian appearance. The retained P300
photoreal renderer does not directly transform covariance with F; it uses
sigma = reference spacing * clamp(current r8 / reference r8, 1, 4), and axes
(sigma, sigma, sigma/4). Consequently sparse tangential coverage also thickens
the surface. See gaussian_footprint_followup.md for the separate F-based risk.

The opt-in `render_splat_photoreal.py --compare-normal-thickness` compares that
unchanged baseline with axes (sigma, sigma, reference spacing/4). Both images
share the exact particle positions, density normals, rotation basis, tangent
radii, live support/opacity, pin appearance latch, camera, lighting and normal
image filter. Construct both covariances from the same basis; do not subtract
a nearly equal covariance or alter physical F. This is not a learned appearance
field or a change to the optimization loss. The default renderer is unchanged.

Use the retained corrected P300 archive, N300k,T20,dt1/240,dx.3062907544wu,
loss36^3,iters8,24-window export, at3840x2160,35-degree azimuth/18-degree elevation,
stride12. Verify the source configuration before launch rather than assuming
the dimensions from this plan. Run only on hyde06 in an isolated work folder;
preserve before/corrected. Bind archive, report, code and generated files in a
receipt. Every generated frame must be inspected before shipping a video.

Observe supplied covariance's normal and tangent standard deviations,
anisotropy and normal/tangent coupling on GPU. Report all particles,
positive-support/opacity particles, active pins, free particles, the fixed upper
height cohort, and sparse support separately. These populations are not camera
visibility or an inferred thin-feature label. Invalid rows and empty populations
remain explicit; do not repair them inside the diagnostic. World covariance
observations are not exact projected raster footprints.

Inspect unencoded paired images and alpha coverage at every selected frame,
including boundary edge spread, visible streaks and any newly uncovered areas.
The exporter reports baseline alpha>=.5 pixels lost, coverage changes, and
temporal image/normal changes using identical baseline masks. These are
appearance observations, not independent physical hole metrics or proof of
oscillation. Shrinking a footprint can expose missing material. Do not accept a
sharper boundary by concealing or ignoring new gaps, and do not infer success
from image MAE alone. No new simulation or rendering-loss influence is implied;
this pair holds an already optimized archive fixed.

## Completed fixed-archive comparison

Frozen2261405, server work/p303/p323_thickness1.29 CPU cases pass (21 diagnostic,
6 covariance,2 existing pin checks), and both actual CUDA side-stream/single
host-packet cases pass in1.50s. Source archive/report hashes match before/after.
41 paired3840x2160 frames were generated in42.88s; this is export time, not a
new physical optimization. All82 footprint observations report valid inputs.

The concern is observable in the active representation: at raw108 the largest
normal standard deviation among positive-support/opacity rows is.03493081wu,
versus reference.00873270wu (4x); tangential maximum is.13972327wu. This population
is not camera visibility. The treatment fixes normal thickness at.00873270wu;
tangent radii remain unchanged to roundoff. Whole-population normal means change
much less than the extreme; do not generalize the4x maximum to all particles.

Coverage also changes: the maximum loss of baseline alpha>=.5 pixels is5719 of
997559 at raw72, with a largest individual alpha decrease.48953. The count alone
does not separate desirable outer-edge contraction from newly exposed interior
gaps. Original opaque-pixel loss is not a physical-hole metric.

Root visually inspected all41 paired full-frame overviews and selected native
body patches. Independent review inspected all41 native upper pairs and21
overview pairs. Original PNG hashes/dimensions and report hashes were checked.
No obvious newly opened large aperture or gross silhouette discontinuity was
found within those inspected views. Nevertheless raw48–156 shows clearer
individual streaks/separations, and late ear fringe/halo remains. Uniform ceramic
does not exercise texture advection. This review is of unencoded PNGs; encoded
video playback, unselected physical substeps, other cameras and19 shapes are
not certified. See evidence/p323 for exact inspected scopes and hashes.

Decision: normal-thickness decoupling alone is not the required artifact fix;
do not promote it. Keep the default and before/corrected deliverables unchanged.
This result leaves tangential footprint, shading-normal and support/opacity
effects to be distinguished, alongside the unresolved physical supply/rest work.
Rendering-loss influence is unchanged because the comparison optimizes nothing.
