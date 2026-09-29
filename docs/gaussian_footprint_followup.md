# Separate physical deformation from visible Gaussian size

User concern,2026-09-28: a better physical deformation F need not produce a
better Gaussian image. With Sigma=F Sigma0 F^T, large stretches can enlarge
tangent support or normal thickness. For isotropic Sigma0=s0^2 I, standard
deviation axes equal s0 times F's singular values; covariance eigenvalues scale
with their squares. For anisotropic surface splats, orientation relative to the
material tangent/normal also matters. Good detF alone does not bound anisotropy.

The code contains a smooth stretch saturation in pipeline/gauss_loss.py and
render/covariance.py, but that is not the active covariance path in the retained
P300 before/corrected4K photoreal exports. Those use density-normal discs and
current8NN spacing, with sigma=target_spacing*clamp(r8/target_median_r8,1,4).
The axes are sigma,sigma,sigma/4. Pin admission latches normals and sigma;
support and opacity remain live. Thus a
sparser patch can broaden both tangent support and normal thickness even without
feeding physical F directly to the covariance. F can still affect it indirectly
through material positions. Material-shading variants change shader normals,
not this covariance rule. Surface-common uses the same radius range statelessly.

Current P316/P320 rendering guidance uses18views at64pixels and no shared GS loss;
that alone cannot supervise the exact exported4K footprint. The opt-in shared
surface GS model has operator tests, but P303's weight1 quality comparison failed.
No unverified GS loss weight or saturation is promoted by this concern.

After the physical path/holes comparison, the high-resolution gate must separate:

- Physical F/J health and, when available, geometric/material deformation.
- Render covariance tangent radii, normal thickness and projected pixel axes,
  reported with opacity/support and sparse/thin-region membership.
- Edge spread, normal-buffer/PBR variation and coverage at the same positions,
  cameras and lighting before any footprint intervention.

Renderer-dependent footprint/appearance diagnostics are not raw simulation
quality metrics. Retain independent material supply/holes metrics. Shrinking
splats can reveal missing coverage; opacity/size must not hide physical holes.
Changing physical F merely to obtain smaller splats is not a justified remedy.
P323 now measures the active covariance and compares fixed-reference normal
thickness at identical archived poses (footprint_thickness_p323.md). Some
positive-support splats reach4x reference normal thickness, but decoupling that
thickness alone leaves streaks/fringe and is not promoted. That comparison is
not an F-covariance causal experiment or a completed visual-quality fix.
