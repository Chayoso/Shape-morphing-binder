# Survey: carrying sub-pitch surface relief physically in a 300k MLS-MPM morph

Written 2026-10-07 (local, no server use, no code). Question: how can fur-, scale-, tooth-sized relief (wavelengths of
2.7–5.4 particle pitches, below what 300k positions carry) be part of the simulated state and be driven by the
objective through the physics, instead of a display layer painted from the target reference (D111, rejected for the
paper).

Our fixed facts (docs/experiments.md, D12–D16, D70, the band analysis of D73): 300k particles, pitch 0.049 wu, grid cell
0.3 wu = 6 pitches; cubic B-spline transfers; the stress channel `dFc` shapes relief down to about one cell, the `u`
channel down to the sampling limit; the outer-layer relaxation erases relief of 6 spacings and halves 11; a 300k
volume sample of the target carries 0.36 / 0.66 of the mesh's relief at 2.7 / 5.4 pitches, our runs 0.28 / 0.55, a
1.2M / 2.4M sample 0.58–0.66 / 0.83–0.87. So the runs already hold about 80 % of the ceiling that 300k positions set;
the rest is a sampling (Nyquist) limit, not a physics limit. Every candidate below is judged against that.

Reading depth is marked per paper: [full] = full text read, [abs] = abstract and project page only.

---

## 1. Resolution where it matters

**Ferstl, Ando, Wojtan, Westermann, Thuerey, "Narrow Band FLIP for Liquid Simulations", CGF 35(2), Eurographics 2016.**
DOI [10.1111/cgf.12825](https://doi.org/10.1111/cgf.12825), [PDF](https://pub.ista.ac.at/group_wojtan/projects/2016_Ferstl_NBFLIP/nbflip.pdf). [full]
FLIP particles only in a band of width R = 3–4 cells inside the surface; the interior is a grid with semi-Lagrangian
advection; particles are re-seeded to at least 8 per band cell every step and deleted beyond R. The naive mix of band
and grid velocities fluctuates in energy; the fix is a sharp switch at r = R − 1 cell (particle velocities extrapolated
past the band are the cause). Up to 22× fewer particles, speed-up 2× on average, 5.6× when the solver does much
per-particle work; features thinner than 2r get no grid support; thinner bands fail. For us: the band logic (resolution
in a shell of fixed depth, re-seeding to a count per cell) is the template for D122 and surface-only N growth; the
energy lesson transfers: the hand-over between the two resolutions must be sharp and one cell inside the band.

**Gao, Tampubolon, Jiang, Sifakis, "An Adaptive Generalized Interpolation Material Point Method for Simulating Elastoplastic Materials", ACM TOG 36(6), SIGGRAPH Asia 2017.**
DOI [10.1145/3130800.3130879](https://doi.org/10.1145/3130800.3130879); read as chapter 3 of Gao's dissertation ([UW–Madison 2018](https://asset.library.wisc.edu/1711.dl/G24M44FI7YSL28I/R/file-abf33.pdf)). [full]
A multi-level SPGrid with an adaptive GIMP basis: T-junction nodes are constrained to their coarse parents, the C0
multilinear basis is convolved to C1 weights; finer grid and smaller particles near the free surface or collision
boundaries, coarse deep inside. Particles are split 8-way on a randomly rotated cube of half-diagonal dx/4 (mass and
volume shared, velocity and F duplicated) and merged with an SVD-averaged F. Pitfalls named by the authors: a mismatch
of particle size and grid level gives numerical fracture; sparse interior particles are driven to the surface in
energetic motion and "diminish surface details"; no split/merge within d_small of the surface or it is visible; the
speed-up is "modest" when the whole surface is refined; no C1 continuity in the adaptive basis beyond the GIMP
convolution; explicit integration only. For us: the only MPM precedent for a finer grid in a surface band; it is GIMP,
not MLS-MPM, and our cubic B-spline / MLS force form has no published two-level variant. Their interior-to-surface
turnover is our D13 turnover measured on a different problem.

**Xu, Wang, Yuan, Tao, Tang, Zhao, "Fluid-Solid Coupling Surface Detail Optimization Algorithm Based on MPM", J. CAD & CG 36(2), 2024.**
DOI [10.3724/SP.J.1089.2024.19763](https://www.jcad.cn/en/article/doi/10.3724/SP.J.1089.2024.19763). [abs]
Combines the two above for solids: adaptive GIMP grid near the surface for detail, narrow-band APIC with particle
resampling inside the solid; 80 % fewer particles, 50 % faster, "enhanced surface detail". No numbers on what detail;
cited as evidence that the band + fine grid combination has been run for solids.

**Vacondio, Rogers, Stansby, Mignosa, Feldman, "Variable resolution for SPH: a dynamic particle coalescing and splitting scheme", CMAME 256, 2013; and "… in three dimensions", CMAME 300, 2016.**
DOIs [10.1016/j.cma.2012.12.014](https://doi.org/10.1016/j.cma.2012.12.014), [10.1016/j.cma.2015.11.021](https://doi.org/10.1016/j.cma.2015.11.021). [abs]
Split and coalesce SPH particles with mass and momentum conserved; the 3-D splitting pattern is chosen by a
variational principle that minimises the density error; coalescing gives a 15× saving over splitting alone. For us:
the density error of a split is a measured, minimised quantity, and our minimum-spacing / density-spread probes (D70)
are the right ruler for it.

**Winchenbach, Hochstetter, Kolb, "Infinite Continuous Adaptivity for Incompressible SPH", ACM TOG 36(4), SIGGRAPH 2017.**
DOI [10.1145/3072959.3073713](https://doi.org/10.1145/3072959.3073713). [abs]
Continuous particle sizes with the target mass a function of the distance to the surface; a mass-redistribution scheme
keeps size variation spatially smooth, which the authors identify as the property that makes large adaptivity ratios
(up to 10^5) stable. For us: the rule "size set by distance to the surface, varying smoothly" is what D122's band
sampling should satisfy; an abrupt F× jump at the band edge is what the literature warns against.

**Adams, Pauly, Keiser, Guibas, "Adaptively sampled particle fluids", ACM TOG 26(3), SIGGRAPH 2007.**
DOI [10.1145/1276377.1276437](https://doi.org/10.1145/1276377.1276437). [abs]
Sampling condition from the local feature size; fewer particles deep inside and near thick flat surfaces; a surface
defined from per-particle distances carried with the particles so the surface is stable under resampling. For us: the
sampling criterion (feature size, not depth alone) is the honest version of "surface-adaptive".

**Ando, Thürey, Tsuruno, "Preserving Fluid Sheets with Adaptively Sampled Anisotropic Particles", IEEE TVCG 18(8), 2012.**
DOI [10.1109/TVCG.2012.87](https://doi.org/10.1109/TVCG.2012.87). [abs]
Thin regions found from the anisotropy of particle neighbourhoods and filled by splitting; collapsed in deep water.
For us: the anisotropy test is a measurable trigger for where resolution is needed (thin features), which our
thin-feature metrics already approximate.

**Yue, Smith, Chen, Chantharayukhonthorn, Grinspun, Kamrin, "Hybrid Grains", ACM TOG 37(6), SIGGRAPH Asia 2018.**
DOI [10.1145/3272127.3275095](https://doi.org/10.1145/3272127.3275095). [abs]
(The paper the brief probably means by "hybrid grids".) Adaptive coupling of DEM and MPM for granular media: discrete
where the continuum assumption fails, continuum elsewhere, with an oracle that switches. Not a grid refinement; listed
for completeness, not used below.

## 2. Codimensional surfaces inside MPM and explicit surface tracking

**Jiang, Gast, Teran, "Anisotropic Elastoplasticity for Cloth, Knit and Hair Frictional Contact", ACM TOG 36(4), SIGGRAPH 2017.**
DOI [10.1145/3072959.3073623](https://doi.org/10.1145/3072959.3073623), [PDF](https://math.ucdavis.edu/~jteran/papers/JGT17.pdf). [full]
Cloth and hair as Lagrangian meshes whose in-manifold F comes from the mesh and whose normal column d3 = F·D3 is
evolved by the grid; the energy is a QR split of F·D: 2-D fixed corotated in the manifold, a quadratic shear penalty
and a one-sided compression penalty in the normal; contact is this normal compression made plastic. Up to 1M DOF
under 30 s/frame. Limitation stated by the authors: the apparent thickness is set by the grid; with a coarse grid
"particles stay separated by about half a grid cell"; no fail-safe for penetration. For us: a surface layer coupled to
the body through the grid is held off the body by about half a cell (3 pitches) and cannot be placed to sub-pitch
accuracy; a layer must be bonded Lagrangian-ly (our bonds), not through the transfers.

**Guo, Han, Fu, Gast, Tamstorf, Teran, "A Material Point Method for Thin Shells with Frictional Contact", ACM TOG 37(4), SIGGRAPH 2018.**
DOI [10.1145/3197517.3201346](https://doi.org/10.1145/3197517.3201346), [PDF](https://www.math.ucdavis.edu/~jteran/papers/GHFGTT18.pdf). [full]
Kirchhoff–Love shells on subdivision finite elements inside MPM: control points are MPM particles, bending comes from
Kirchhoff–Love quadrature points that are not transferred to the grid, contact from the fiber (normal) compression as
in Jiang 2017; denting and wrinkling are plasticity in the Kirchhoff–Love part. 14k–3.1M particles (sand included),
under 1 s to 167 s per frame, CFL 0.3. The limitation that matters for us (their Fig. 16): "persistent wrinkling if the subd mesh resolution is too
high relative to the grid resolution" (mesh dx 0.02 on grid dx 0.04 wrinkles, on grid 0.02 it does not), and visible
separation if dx is too large. Translation: shell modes finer than the grid are invisible to the grid and are never
damped or moved by it; they persist. A sub-cell shell in our pipeline would keep whatever relief it has, which is the
feature we want and the flicker risk we have seen (D111 stage 1).

**Han, Gast, Guo, Wang, Jiang, Teran, "A Hybrid Material Point Method for Frictional Contact with Diverse Materials", PACMCGIT 2(2), 2019.**
DOI [10.1145/3340258](https://doi.org/10.1145/3340258). [abs]
Internal forces from a Lagrangian mesh (tets, curves), the Eulerian grid only for self-collision and coupling. For us:
the pattern "forces on the Lagrangian structure, the grid for contact" is how a bonded surface layer should be
implemented; our bonds already do this for fragments.

**Wojtan, Thürey, Gross, Turk, "Deforming Meshes that Split and Merge", ACM TOG 28(3), SIGGRAPH 2009.**
DOI [10.1145/1531326.1531382](https://doi.org/10.1145/1531326.1531382), [PDF](https://pub.ista.ac.at/group_wojtan/projects/2009_Wojtan_DMSM/wojtan_2010_DMSM.pdf). [full]
Explicit triangle surface mesh advected by the simulation (FEM in a BCC lattice, or an Eulerian fluid); topology
changes detected at a grid resolution (complex-cell test) and re-meshed locally by an isosurface, the rest of the mesh
kept. The authors' own statement: purely Eulerian and re-sampled hybrid surfaces "cannot represent a surface with
details smaller than the grid resolution"; only the Lagrangian mesh retains sub-grid detail between events. For us:
exactly the right property, but D13 measured it on our problem (a source surface carried through a morph: area ×3.7–4.6,
5–6 % folds, 10 % of the final surface uncovered) because a morph has material turnover: interior material becomes
surface. A tracked surface works for a deforming body, not for a morph. Not a candidate.

**Da, Hahn, Batty, Wojtan, Grinspun, "Surface-Only Liquids", ACM TOG 35(4), SIGGRAPH 2016.**
DOI [10.1145/2897824.2925899](https://doi.org/10.1145/2897824.2925899). [abs]
The whole state on a surface mesh (BEM for pressure, surface tension and contact). Shows that a surface-resident state
with its own dynamics is a complete simulation, not a display; it is for inviscid liquids and does not transfer, cited
only for the principle.

## 3. Physics-based detail augmentation (not target fitting)

**Müller, Chentanez, "Wrinkle Meshes", SCA 2010.**
[EG DL 10.2312/SCA/SCA10/085-092](https://diglib.eg.org/items/b3de6809-9e17-45cf-97db-d781b7661125), [PDF](https://matthias-research.github.io/pages/publications/wrinkleMeshes.pdf). [full]
A finer wrinkle mesh attached to the base mesh by Bézier interpolation; each wrinkle vertex may deviate from its
attachment within a painted maximum distance (one-sided to avoid collisions); distance and bending constraints with a
two-phase force profile; a quasi-static position-based solver runs beside the base simulation; wrinkles form where the
base mesh compresses. Fast, no dynamics in the wrinkle layer, the pattern comes from the base mesh's compression and the
wrinkle mesh's tessellation. For us: it is the mechanism of D111 with physics constraints instead of a render fit; the
relief is still not the target's, and it is explicitly a quasi-static layer. Useful only as the constraint form (an
allowed deviation band around an attachment) if a bonded layer is built.

**Kavan, Gerszewski, Bargteil, Sloan, "Physics-inspired Upsampling for Cloth Simulation in Games", ACM TOG 30(4), SIGGRAPH 2011.**
DOI [10.1145/2010324.1964988](https://doi.org/10.1145/2010324.1964988), [PDF](https://users.cs.utah.edu/~ladislav/kavan11physics/kavan11physics.pdf). [full]
A linear operator from coarse to fine vertices learned from paired coarse/fine simulations aligned by tracking
constraints (harmonic test functions, after Bergou 2007), with harmonic regularisation against overfitting; 0.8 ms;
high-frequency travelling waves are re-added as oscillatory modes. The fine detail is "physics-inspired" only through
the training data; at runtime it is a fixed linear map. For us: a learned map from our 300k state to a 1.5M-like
surface is the display layer again (data-driven instead of render-fitted); it is not state. Not a candidate.

**Rémillard, Kry, "Embedded Thin Shells for Wrinkle Simulation", ACM TOG 32(4), SIGGRAPH 2013.**
DOI [10.1145/2461912.2462018](https://doi.org/10.1145/2461912.2462018), [PDF](https://www.cs.mcgill.ca/~kry/pubs/ets/ets.pdf). [full]
A high-resolution thin shell (the skin) two-way coupled to a coarse C1 quadratic FEM lattice (the interior) by position
constraints H x = H B q that act only below the wrinkle frequency: each constraint is a Gaussian-weighted cluster
average of shell vertices made to match the same average of the embedded lattice (local constraints; global
Laplacian-eigenvector constraints spread ripples over the whole model). The cut-off comes from the film-on-substrate
wavelength λ = 2π h [(1 − ν_q²) E_x / (3 (1 − ν_x²) E_q)]^{1/3} (skin thickness h, skin E_x, interior E_q); the Gaussian's
σ = λ/π and the cluster spacing r = λ/2 (Nyquist). Wavelengths match theory and full volumetric simulations. For us:
the cleanest published definition of "a fine surface state attached to a coarse body at low frequency only, free at
high frequency, with its own energy"; the constraint matrix is a definition from material constants, not a tuned
weight. The shell is dynamic (not a static post-process) and the relief emerges from compression, i.e. it is generic
wrinkling, not a target's relief.

**Cerda, Mahadevan, "Geometry and Physics of Wrinkling", PRL 90, 074302, 2003.**
DOI [10.1103/PhysRevLett.90.074302](https://doi.org/10.1103/PhysRevLett.90.074302). [abs]
Scaling laws for a sheet of bending stiffness B on an effective substrate stiffness K under compression: wavelength
λ ∝ (B/K)^{1/4}, amplitude from the imposed compression; valid far from onset. For a stiff film on a soft elastic
substrate the standard forms are λ = 2π h (Ē_f / 3 Ē_s)^{1/3}, A = h (ε/ε_c − 1)^{1/2}, ε_c = (1/4)(3 Ē_s/Ē_f)^{2/3} (as in
Rémillard–Kry Eq. 3; the Soft Matter review by Li, Cao, Feng, Gao 2012, DOI
[10.1039/C2SM00011C](https://doi.org/10.1039/C2SM00011C), not read in full, covers the crease/fold regime when the
moduli are within a factor of a few, and period doubling). For us: the one mechanism that creates relief below the
discretisation of the body from constants alone; the wavelength in pitches is fixed by h and the modulus ratio, and at
a ratio near 1 the surface creases instead of wrinkling.

**Tallinen, Chung, Rousseau, Girard, Lefèvre, Mahadevan, "On the growth and form of cortical convolutions", Nature Physics 12, 2016.**
DOI [10.1038/nphys3632](https://doi.org/10.1038/nphys3632), [PDF](https://pages.ucsd.edu/~msereno/_596_2022/readings/03.14-Gyrification.pdf). [full, main text]
A cortical layer of thickness h perfectly adhered to a core, both soft neo-Hookean with similar moduli, the layer given
a prescribed tangential expansion g; the elastic equilibrium folds at a scale set by h; the placement and orientation
of folds follow the initial geometry, the fine pattern at the scale of h is sensitive to perturbations. For us: the
growth tensor on a layer is the physical form of "a controlled rest metric": the objective would choose g, the physics
makes the relief. Also the cautionary fact: the pattern is unique to the mechanics, not to a target.

## 4. Reconstruction from the particles themselves

**Yu, Turk, "Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels", ACM TOG 32(1), 2013.**
DOI [10.1145/2421636.2421641](https://doi.org/10.1145/2421636.2421641), [PDF](https://faculty.cc.gatech.edu/~turk/my_papers/particle_surfaces_tog.pdf). [full]
Each particle's kernel is stretched by G = (1/h) R Σ̃^{-1} Rᵀ from the weighted-PCA covariance of its neighbours
(constants k_r = 4 for the maximal stretch, k_s = 1400, k_n = 0.5 for isolated particles, N_ε = 25 neighbours), kernel
centres Laplacian-smoothed with λ = 0.9–1 (display only, never written back). Flat surfaces, thin streams and sharp
edges come out from the particle distribution; a connected-component test prevents separate bodies from attracting.
What it cannot do: it shapes the field from the positions' second moments, so it sharpens what the positions already
encode and cannot add structure below the spacing. For us: our Zhu–Bridson surface could be replaced by this with no
new state; the band analysis would move only where the current field blurs existing relief (the 1.2–1.4-pitch blur of a
sample's displayed surface in D73 is the field's kernel plus the sample), not at 2.7 pitches.

**Xie, Zong, Qiu, Li, Feng, Yang, Jiang, "PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics", CVPR 2024.**
[arXiv 2311.12198](https://arxiv.org/abs/2311.12198). [full]
The same Gaussian kernels are the MPM particles and the render primitives; each kernel's covariance is evolved by its
deformation gradient (A_p ← F_p A_p F_pᵀ) and its spherical harmonics rotated by the polar part of F; interior filling
particles are added for the volume. For us: F is simulated state, so a surfel tilted and stretched by F_p is honest at
the level of "the material around this particle is deformed this way"; it carries orientation (shading) at sub-pitch
scale, not position. Our outer-layer particles' F is shaped by `dFc` and the F repair, so the render loss reaches the
surfel orientation through the stress channel already. (PhysMorph-GS, the user's own prior paper, rendered F-deformed
Gaussians, so the mechanism is not new to this project.)

## 5. Differentiable-simulation precedents and the resolution gap

**Hu et al., "ChainQueen: A Real-Time Differentiable Physical Simulator for Soft Robotics", ICRA 2019** ([arXiv 1810.01054](https://arxiv.org/abs/1810.01054)) [full];
**Du et al., "DiffPD: Differentiable Projective Dynamics", ACM TOG 41(2), 2021** (DOI [10.1145/3490168](https://doi.org/10.1145/3490168)) [full];
**Liang, Lin, Koltun, "Differentiable Cloth Simulation for Inverse Problems", NeurIPS 2019** ([proceedings](https://proceedings.neurips.cc/paper/2019/hash/28f0b864598a1291557bed248a998d4e-Abstract.html)) [abs];
**Li, Du, Wu, Xu, Matusik, "DiffCloth", ACM TOG 42(1), 2022** (DOI [10.1145/3527660](https://doi.org/10.1145/3527660)) [full];
**Li et al., "PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields", ICLR 2023** ([arXiv 2303.05512](https://arxiv.org/abs/2303.05512)) [full].
ChainQueen differentiates MLS-MPM for control and co-design (about 3 000 decision variables); DiffPD differentiates
projective dynamics with contact for system identification, inverse design and real-to-sim (4–19× faster than Newton);
Liang 2019 and DiffCloth differentiate cloth with collisions (QR of a small matrix; dry friction in PD) for parameter
estimation, control and inverse design; PAC-NeRF binds a voxel radiance field to MPM particles and optimises physical
parameters and the initial geometry from video. In every one of them the loss sees the discretised state (nodes,
vertices, voxels, particles) and nothing finer: the inverse design happens at the element scale, the "resolution gap"
is closed by choosing the element size up front (DiffPD runs its napkin at 25²–100² voxels; PAC-NeRF's geometry is the
voxel grid; DiffPD notes that collisions on an Eulerian grid "may introduce artifacts depending on the resolution").
No differentiable-simulation precedent recovers sub-element detail through the physics. For us: the honest reading is
that an image loss can only shape the degrees of freedom the simulator has; to carry sub-pitch relief the state must
have sub-pitch degrees of freedom (more surface particles, or a surface layer with its own DOF), and the loss must reach
them through a control with a physical meaning.

---

## Ranked candidates for this pipeline

The ruler for all of them is the band analysis of D73 (in-phase share of the mesh's relief at 2.7 / 5.4 / 10.8 pitches,
bunny and dragon, against the 300k and 1.2M/2.4M samples), the roughness probe, the λ = 0 twin, and momentum
conservation (the render arm must beat its twin on every metric, user rule 2026-10-04). "Physical" below means a
definition from the discretisation or a constitutive law with no per-shape constant; "fit" means a quantity the
objective sets.

### 1. Resolution in a surface band: D122 (denser band at fixed N) → surface-only N growth → a fine grid level in the band

- **Definition.** The outer band (depth in pitches, e.g. the near band's berth plus one loss cell) is sampled F× denser
  (D122), then N grows only there (NB-FLIP: particles only where they are seen; Winchenbach: size a smooth function of
  the distance to the surface, no jump at the band edge; AGIMP: a smaller particle type belongs to a finer grid level).
  Stage 3 adds one finer grid level (dx/2) in a band of 3–4 fine cells (NB-FLIP's R), with the coarse level elsewhere
  and the transfers at the interface constrained as in AGIMP (T-junction nodes slaved to parents). No new energy, no new
  control: `dFc` on the fine level now acts at half a cell, `u` and the relaxation are in band spacings.
- **Physical vs fit.** Entirely physical (positions, mass, the same stress); the target enters as today, through the
  objective. Caveat that must be stated: the transport and proximity targets are the 300k target sample, which is
  itself Nyquist-limited; the band's finer relief can only be asked for by a target sample of matching surface density
  (proximity / near band on a denser surface sample) and by the render terms at 192 px. That is still "driven through
  the physics", but the target must be resampled, not only the body.
- **Cost at 300k.** D122: none. Surface-only N growth: the outer layer is about 4 % of the particles, a 2–3-pitch band
  perhaps 15–20 %; ×8 in the band is +1–1.5 N, so about 2× the window (14 → 25–30 s) and the run (16 → 30 min). The
  fine grid level: node count in the band ×8 per refined coarse cell, sparse; the adjoint and the CUDA graphs need a
  second level; the MLS-MPM force form needs a per-level dx; the interface basis is the largest engineering item (no
  published MLS-MPM two-level variant; AGIMP is GIMP, explicit, C0 + convolution).
- **Measurement.** Band share at 5.4 pitches from 0.55 toward the 2.4M sample's 0.83–0.87; at 2.7 from 0.28 toward
  0.58–0.66. Prediction for D122 alone: the gain is at 5.4, not 2.7, because the relaxation still erases 6 band
  spacings (= 3 coarse pitches at F = 8) and the stress channel still acts at one coarse cell; the 2.7 band needs stage
  3. Add the turnover ruler from D13 / AGIMP: the share of the final surface born in the coarse interior (dilution of
  the band); if it is large the band must be re-seeded (NB-FLIP's count per cell), which touches the mass rule.
- **Main risk.** Particle-size mixing (AGIMP: type/level mismatch → numerical fracture; interior particles reaching the
  surface), the spacing-defined constants (relaxation neighbourhood, berth, minimum spacing, the loss grid following N)
  all change meaning inside the band; the 40k gallery must be run with the band defined in pitches so that nothing
  changes at 40k. This is the only candidate that raises the Nyquist ceiling itself.

### 2. A bonded elastic shell with a controllable rest curvature (the codimensional layer as state)

- **Definition.** A surfel set on the exterior at a sub-pitch spacing (the D59/D111 lattice, 300k–600k surfels) made a
  Lagrangian shell: mass, velocity, in-manifold F (Jiang 2017's QR split: 2-D fixed corotated membrane), bending energy
  on the discrete curvature against a rest curvature κ̄ (Guo 2018's Kirchhoff–Love part with denting as plasticity),
  and a bond to its parent particles (Han 2019: internal forces on the Lagrangian structure, the grid only for the body;
  Jiang 2017 shows a grid-coupled layer sits half a cell off the body, which is why the coupling must be the bond). The
  control is Δκ̄ per surfel per window, released and settled like `dFc` and `u`; the render terms (silhouette and
  shading) are evaluated on this shell, so the objective reaches κ̄ through the shell's equilibrium with the body.
  Rémillard–Kry's low-frequency-only constraint (Gaussian clusters, σ and r from material constants) is the alternative
  coupling if the bond proves too stiff.
- **Physical vs fit.** The shell state, its energies, the bond and the released-end equilibrium are physical; what
  survives release is held by bending against membrane and bond. κ̄ is a fit, with exactly the status `dFc` has today (a
  rest-state increment chosen by the objective) but on a layer whose sub-pitch content has no source other than the
  render target (the body is Nyquist-limited). This must be said plainly in the paper: the relief is render-driven
  through a shell, not recovered from the body. A growth-tensor reading (Tallinen: a controlled rest metric/curvature
  of a cortical layer) gives κ̄ a physical name.
- **Cost at 300k.** Shell forces are Lagrangian, O(surfels × neighbours); forward about 10–20 % of a window at 300k
  surfels, the adjoint through 40 steps of shell state the larger part (perhaps +30–50 % per window); memory for the
  shell's trajectory. Cheaper than candidate 1's stage 3, more than D122.
- **Measurement.** Band analysis on the shell's surface at 2.7 / 5.4 pitches (target: the 1.5M sample's 0.75–0.77 at
  5.4, which D111 stage 0 reached by fitting); the λ = 0 twin receives no gradient on κ̄, so its shell stays at the body's
  relief: the render's influence is then the whole of the gain, which satisfies the "render must beat the twin" rule by
  construction and must be reported as such. A hold test: freeze the control after window W and check the relief
  persists through the released steps (the rest state holds it) and does not flicker across windows (Guo 2018: sub-grid
  shell modes are not damped by the grid; our D111 stage-1 flicker was 1.35× the base).
- **Main risk.** It is the mechanism nearest to D111 in spirit; the user may still call a rest-shape fit on a surface
  layer a trick. The difference from D111 is testable (dynamics, a constitutive bound on what relief can be held, the
  hold test, the twin), and that is the argument to make before building it, not after.

### 3. Bilayer wrinkling: a stiff outer layer with a tangential growth control (emergent relief)

- **Definition.** The outer layer (thickness h ≥ 1 pitch, on candidate 1's fine grid or as candidate 2's shell) gets a
  higher modulus E_f than the core E_s and a growth control g (rest-metric increment, Tallinen's tangential expansion);
  compression of the layer buckles it at λ = 2π h (Ē_f/3Ē_s)^{1/3} with amplitude h (ε/ε_c − 1)^{1/2} (Cerda–Mahadevan;
  Rémillard–Kry Eq. 3). The objective chooses g (where the shading loss sees texture); the pattern self-organises.
- **Physical vs fit.** Fully physical; nothing is fitted to the target's relief. The target enters only through where g
  is applied.
- **Cost at 300k.** That of the carrier (1 or 2); the control adds one scalar per layer particle.
- **Measurement.** The band's relief power (rms in the band) reaches the mesh's, but the in-phase share stays near zero
  because the pattern is not the target's; add a band-power ratio to the band analysis and report both. For λ = 5.4
  pitches with h = 1 pitch the modulus ratio needed is Ē_f/Ē_s ≈ 1.9, which is the crease regime (Li et al. 2012), not
  clean wrinkles; λ = 2.7 pitches is not reachable with h ≥ 1 pitch (it would need a film softer than the core, which
  does not wrinkle); a 2-pitch layer and a ratio of 10 give λ ≈ 19 pitches, above the band of interest.
- **Main risk.** It cannot match a given target; the morph's own compression wrinkles the layer wherever the body
  shrinks (unwanted, everywhere a sphere becomes a thinner part); at 40k the same λ in pitches is 5–10 % of the shape.
  A paper-side demonstration of physically generated texture at most, with g = 0 as the default.

### 4. F-shaped surfels on the exterior layer (PhysGaussian kinematics; no new state)

- **Definition.** Each exterior surfel's disc is the particle's reference disc transformed by its F_p (normal from
  F_p^{-T} n_0, extent from the polar stretch), as PhysGaussian evolves its kernels; the render terms on the layer then
  depend on F_p, and the gradient reaches F through `dFc` as today. Yu–Turk's position-covariance anisotropy is the
  no-F alternative (it sharpens existing relief; it cannot add sub-pitch structure).
- **Physical vs fit.** F is simulated state; the released-end F is an equilibrium of the fixed-corotated energy. No new
  dependence on the target.
- **Cost at 300k.** Negligible (one 3×3 per surfel).
- **Measurement.** The band analysis measures offsets and will show nothing at 2.7 / 5.4 pitches; add a normal-band
  analysis (relief of the displayed normal against the mesh's) and the shading loss on the layer. This is a shading
  gain, not a geometry gain, and must be reported as such.
- **Main risk.** The surface particles' F is shaped by the control history and the F repair, not by the target's relief;
  a surfel tilted by F while its neighbours are not is a sub-pitch normal map that positions cannot confirm. PhysMorph-GS
  already did it. Cheap to measure on existing runs as a diagnostic; weak as a paper claim.

**Ruled out by our own record:** an explicit surface mesh carried through the morph (Wojtan 2009; refuted by D13,
material turnover); learned upsampling (Kavan 2011) and wrinkle meshes (Müller–Chentanez 2010): post-hoc layers without
state, i.e. D111 with a different fitting rule.

## Recommendation

Read D122's band table first and against the prediction above (gain at 5.4 pitches, none at 2.7, with the turnover share
of the band measured): if the 5.4 share moves by less than 0.05 toward the sample the band is bound by the relaxation
and the stress channel, not by N, and the next step is candidate 1 stage 3 (a fine grid level in the band, with the
target resampled at the band's surface density), because it is the only mechanism in the literature that raises the
sampling ceiling itself and leaves every definition physical. In parallel, run candidate 4 as a diagnostic only on the
existing 300k runs (no code in the loop: a normal-band analysis of F-shaped surfels) to know how much shading relief
the state already holds. Candidate 2 (the bonded shell with a rest-curvature control) should be put to the user as a
question before any code: it is the one mechanism that can reach the 1.5M-sample relief at this N with dynamics, a
constitutive bound and a hold test, but its sub-pitch content is render-driven by construction, and whether that counts
as physics or as D111 with inertia is the user's call. Candidate 3 (bilayer wrinkling) is a demonstration of emergent
texture for the paper's discussion, not a route to the target's relief, and should wait until 1 or 2 carries it.
