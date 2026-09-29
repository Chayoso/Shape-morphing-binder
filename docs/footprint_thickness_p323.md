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

Implementation and CPU covariance/pin checks precede GPU comparison. There are
no completed visual-quality results or default-policy promotions in this record.
