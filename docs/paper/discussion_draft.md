# Paper draft: Related Work positioning and Comparison / Discussion (2026-10-15)

Draft for the SIGGRAPH 2027 submission, written for the reviewers. Rules this draft follows:

- Every number comes from `docs/experiments.md` (D138, D145 and its amendments, D147) or from
  `output/results_2026-10-14/d145_evidence/d145_first_tables.md`. The source is given in a comment after each paragraph.
- A number that does not exist yet is written `[TBD: which run]`. No number is estimated.
- Every citation was checked against its source page: arXiv, OpenReview, CVF, ACM DL / Crossref, PMLR / NeurIPS
  proceedings, Wiley / EG, or the project's GitHub. The keys are in `docs/paper/related_refs.bib`. A reference that
  could not be verified there is marked UNVERIFIED in the bib and is not cited in the prose.
- Our own prior work is cited as prior work, never as a competitor: "our prior F-control formulation [Xu et al. 2025]".
  The PhysMorph-GS preprint [Song and Hyde 2025] is this project's earlier version.

Citations are written author–year as in the ACM format, e.g. [Alexa et al. 2000]. The bib key is given at the first use
of each paper as `{key}`.

## Related Work

Our problem is unusual among morphing problems in three ways:

- The source is a sphere, so there is no correspondence to the target and, for most targets, no common genus.
- The path must be a physical body: mass, momentum, contact, external forces, and an end at rest.
- Images of the target steer that body, not its surface.

We organise prior work by what each family keeps of these three.

**Surface interpolation.** Classical morphs deform one surface into another:

- As-rigid-as-possible interpolation of local transformations [Alexa et al. 2000] `{alexa2000arap}`, and its
  gradient-domain forms [Sumner and Popović 2004; Xu et al. 2005] `{sumner2004deftransfer, xu2005poisson}`.
- Geodesics in shape spaces [Kilian et al. 2007] `{kilian2007shapespace}` and in spaces of shells [Heeren et al. 2012;
  Heeren et al. 2014] `{heeren2012shells, heeren2014shells}`; intersection-free geodesics via Repulsive Shells
  [Sassen et al. 2024] `{sassen2024repulsive}`; elastic matching with second-order Sobolev metrics [Hartman et al.
  2023] `{hartman2023h2}`.
- Learned or optimised interpolants that also estimate the correspondence: NeuroMorph [Eisenberger et al. 2021]
  `{eisenberger2021neuromorph}` and Spectral Meets Spatial [Cao et al. 2024] `{cao2024sms}`.

All of them deform one mesh connectivity from the first shape to the second, with a correspondence given, estimated,
or replaced by an unparametrised matching term. The genus therefore cannot change. A sphere reaches a genus-0 target
only through a separate parameterisation of the target onto the sphere, such as conformalized mean curvature flow
[Kazhdan et al. 2012] `{kazhdan2012cmcf}`, which is how we build the ARAP baseline.

These methods produce smooth surface paths (intersection-free in the case of Repulsive Shells), but no body. Closest
to ours in spirit are the divergence-free and Hamiltonian interpolations of Eisenberger and colleagues
[Eisenberger et al. 2019; Eisenberger and Cremers 2020] `{eisenberger2019divfree, eisenberger2020hamiltonian}`:

- They keep the volume exactly.
- The Hamiltonian one also conserves the momentum of a point set.
- Both work with one connectivity and with (possibly noisy) correspondences.
- Neither models contact or external forces, and neither opens a handle.

**Transport.** Displacement interpolation moves mass along an optimal plan [McCann 1997] `{mccann1997convexity}`.
Bonneel et al. brought it to graphics with Lagrangian mass transport [Bonneel et al. 2011]
`{bonneel2011displacement}`. Convolutional Wasserstein distances make transport barycentres cheap on grids and
surfaces [Solomon et al. 2015] `{solomon2015convolutional}`. Debiased Sinkhorn divergences give a smooth, positive
transport loss with gradients for millions of samples [Feydy et al. 2019] `{feydy2019sinkhorn}`. Transport keeps
mass by construction and is indifferent to topology, which is why our objective uses a Sinkhorn divergence on the
body's mass. As a path, however, transport has no cohesion. Our displacement-interpolation baseline ends on the target
sample but tears thin parts into clouds on the way. Our method uses transport as the objective of a body, not as the
path.

**Neural implicit morphs.** Level-set and velocity-field methods represent the shape as the zero set of a
time-dependent neural field, so topology may change freely:

- Neural implicit surface evolution [Novello et al. 2023] `{novello2023nise}`.
- Lipschitz-regularised networks that interpolate between shapes in a latent code [Liu et al. 2022]
  `{liu2022lipschitz}`.
- Explicit velocity fields that drive an implicit surface through a modified level-set equation, with a soft
  divergence term, between point clouds [Sang et al. 2025] `{sang2025isd}`. 4Deform adds distortion and
  stretching losses and shows topology change [Sang et al. 2025b] `{sang2025fourdeform}`.
- Volume-preserving neural morphing with an adaptive-divergence velocity [Buonomo et al. 2025]
  `{buonomo2025vpnsm}`.
- Flows that are invertible by construction: Occupancy Flow [Niemeyer et al. 2019] `{niemeyer2019occflow}`,
  ShapeFlow [Jiang et al. 2020] `{jiang2020shapeflow}`, FLOWING [Bizzi et al. 2025] `{bizzi2025flowing}`, and explicit
  flows for implicit surfaces [Buonomo et al. 2026] `{buonomo2026explicitflows}`. Such a flow is a homeomorphism, so it
  preserves the source's genus.

These methods give watertight, smooth surfaces at every time. Their "physical plausibility" is a property of the
velocity field: divergence, rigidity, or strain. A morph is not a body with mass, momentum and contact.

Several also need inputs that a sphere-to-target morph lacks:

- The released code of [Sang et al. 2025] expects correspondences between the two shapes.
- 4Deform expects estimated sparse correspondences.
- FLOWING relies on landmarks.

The level-set methods change genus freely, but nothing physical governs how. In our comparison, Buonomo et al.'s bob
frames pass through genus 31–44 on the way to a genus-1 target.

**Generative 3D morphing.** Structured 3D latents of large generators now support training-free morphing between
assets, through attention blending [Sun et al. 2026] `{sun2026morphany3d}`, correspondence-aware latent interpolation
[Liu et al. 2026] `{liu2026interp3d}`, an optimal-transport view of flow-model sampling [Yin et al. 2025]
`{yin2025wukong}`, or a 3D diffusion prior [Yang et al. 2025] `{yang2025regenerative}`. These methods regenerate each
frame from a blended latent. They produce semantically plausible in-betweens across categories, but the frames share
no material, the sequence has no mass or time, and the end is the generator's reconstruction of the target, not the
target itself. They answer a different question, "what lies semantically between two objects", from ours, "how does
one body of matter become another".

**Physics-based control toward a target.** Driving a simulation toward a target goes back to spacetime constraints
[Witkin and Kass 1988] `{witkin1988spacetime}`. The line continued through:

- keyframe and adjoint control of smoke and liquids [Treuille et al. 2003; McNamara et al. 2004; Fattal and Lischinski
  2004; Thürey et al. 2009] `{treuille2003keyframe, mcnamara2004adjoint, fattal2004targetdriven, thurey2009detail}`,
  and guide shapes for liquids [Nielsen and Bridson 2011] `{nielsen2011guide}`;
- reduced optimal control of deformable objects [Barbič et al. 2009] `{barbic2009reduced}`;
- recent differentiable fluid control that morphs densities into letters [Chen et al. 2024; Li et al. 2025]
  `{chen2024eigenfluids, li2025flowmapadjoint}`.

These methods control by external forces. A control force adds momentum: the matter is pushed toward the target, and
whatever the force put in must be taken out again before the matter comes to rest.

The material point method [Stomakhin et al. 2013] `{stomakhin2013snow}` became differentiable in simulators
[Hu et al. 2019; Hu et al. 2020; Huang et al. 2021] `{hu2019chainqueen, hu2020difftaichi, huang2021plasticinelab}`, as
did FEM [Du et al. 2022] `{du2022diffpd}`. These made gradient-based control of deformable bodies routine, for
actuators and for tools that shape plasticine toward a target mass distribution. Point-set transport losses on MPM
particles appear in dexterous manipulation [Li et al. 2023b] `{li2023dexdeform}`. gradSim [Jatavallabhula et al. 2021]
`{jatavallabhula2021gradsim}` closes the loop through a differentiable renderer and actuates an FEM object toward a goal
image. Its object keeps its mesh: it is posed, not morphed into another shape.

Our prior F-control formulation [Xu and Levin 2023; Xu et al. 2025] `{xu2023dgc, xu2025dmpm}` controls an MLS-MPM body
[Hu et al. 2018] `{hu2018mlsmpm}` through per-particle deformation-gradient increments. The stress cannot create net
force or torque, so momentum is kept, and the body can change topology. The PhysMorph-GS preprint
[Song and Hyde 2025] `{song2025physmorphgs}`, this project's earlier version, added splatted silhouette and depth losses
to that control.

The present work keeps the stress control and changes what the body is and what it is scored on:

- the volume is carried in F and assimilated plastically, and the densities are matched, so mass and volume are the
  body's own and the end is the body's rest;
- the transport is a Sinkhorn divergence on the body's mass;
- the render terms (silhouette and shading) are read on the exterior surface of the body [Zhu and Bridson 2005]
  `{zhu2005sand}`;
- their effect is measured against a λ = 0 twin.

To our knowledge, no other work drives a body by differentiable physics from one shape into an arbitrary other shape
under a rendering loss. The nearest are gradSim (a goal image, one fixed mesh) and the fluid-control methods (shape
targets, forces, density supervision).

**Rendering and physics.** Differentiable rendering has been coupled to MPM and spring-mass simulation to identify
materials and animate captured objects:

- PAC-NeRF [Li et al. 2023] `{li2023pacnerf}`;
- PhysGaussian [Xie et al. 2024] `{xie2024physgaussian}`, which drives 3D Gaussian splats [Kerbl et al. 2023]
  `{kerbl20233dgs}` by the MPM deformation gradient;
- Gaussian-informed continuum [Cai et al. 2024] `{cai2024gic}`;
- Spring-Gaus [Zhong et al. 2024] `{zhong2024springgaus}`;
- PhysDreamer [Zhang et al. 2024] `{zhang2024physdreamer}`;
- OmniPhysGS [Lin et al. 2025] `{lin2025omniphysgs}`.

These works fit physical parameters to observed motion, or simulate forward from a fitted asset. None drives a body
from one shape to another. In our method the image loss is not a reconstruction target for the representation. It
is a control signal that reaches the body only through the adjoint of the simulation.

**Render-only shape optimisation.** Inverse rendering can also deform a sphere into a target directly:

- Large Steps preconditions mesh gradients with a Laplacian [Nicolet et al. 2021] `{nicolet2021largesteps}`.
- Continuous remeshing adapts the mesh during the optimisation [Palfinger 2022] `{palfinger2022continuous}`.
- Isosurface representations make topology differentiable: DMTet in nvdiffrec [Munkberg et al. 2022]
  `{munkberg2022nvdiffrec}`, FlexiCubes [Shen et al. 2023] `{shen2023flexicubes}`, and SDF rendering
  [Vicini et al. 2022] `{vicini2022sdf}`.
- Collections of tetrahedral spheres [Guo et al. 2025] `{guo2025tetsphere}` and differentiable Delaunay meshes
  [Son et al. 2024] `{son2024dmesh}`.

The mesh-based ones keep the start's genus: Large Steps "cannot change the topology by punching holes", and its
authors start bob from a torus. A recent high-genus method needs a start primitive of the target's genus
[Gao et al. 2026] `{gao2026highgenus}`. The isosurface-based ones change topology, but their iterates are an
optimisation trace, not a motion. These methods are the "rendering without physics" end of our design space, and we
compare against two of them.

## Comparison and Discussion

### Protocol

All methods run on the six 300k targets used in the paper (bunny, dragon, teapot, bob, armadillo, homer). Each starts
from the same sphere, and each method's frames are converted to the same representation. The meshes of mesh- and
SDF-based methods are sampled by our stratified volume sampler at 300k in the target's frame. Particle methods keep
their particles. All frames are read by the same probes and drawn by the same display.

A method is never scored against the sample it was optimised toward. End metrics compare the result with an
independent stratified sample of the target, drawn with a seed no run uses. The sampling floor is the IoU / chamfer
between two independent samples of the same mesh. Intermediate frames are compared at matched transport progress
p = 1 − S(x_t, I) / S(x_0, I) = 0.25, 0.5 and 0.75, where S is the debiased Sinkhorn divergence [Feydy et al. 2019]
to the independent sample I. Volume is read by one routine for every method: voxel occupancy at 1.5 sampling pitches,
one closing, holes filled, against the same routine on the independent sample. Wall-clock times are on one RTX 6000
Ada with nothing else on the card.

Baselines:

- (i) Our prior F-control formulation [Xu et al. 2025] `{xu2025dmpm}`, reimplemented in our simulator. It was
  verified against the published scale: −97.9 % loss reduction against the paper's −98.3 % at N = 11,160. It keeps
  the published loss, schedule and morph lengths at the same N and samples, with step size 1e-2, the best by its own
  loss.
- (ii) Volume Preserving Neural Shape Morphing [Buonomo et al. 2025] `{buonomo2025vpnsm}`: the authors' code and
  defaults.
- (iii) As-rigid-as-possible interpolation [Alexa et al. 2000] `{alexa2000arap}` in its gradient-domain form
  [Sumner and Popović 2004; Xu et al. 2005] `{sumner2004deftransfer, xu2005poisson}`. The sphere correspondence comes
  from conformalized mean curvature flow [Kazhdan et al. 2012] `{kazhdan2012cmcf}`. Genus-0 targets only.
- (iv) Optimal-transport displacement interpolation on our particles (entropic plan at a blur of one sampling pitch,
  its barycentric map, linear in time): transport without a body.
- (v) Render-only mesh optimisation from the sphere: Large Steps [Nicolet et al. 2021] `{nicolet2021largesteps}`
  and FlexiCubes [Shen et al. 2023] `{shen2023flexicubes}` [TBD: B12 runs and readings].

The released code of [Sang et al. 2025] and of 4Deform [Sang et al. 2025b] expects point correspondences between the
two shapes, which a sphere-to-target morph does not have. 4Deform's repository also carries no licence. We therefore
report the correspondence-free re-implementation of [Sang et al. 2025] that ships with Buonomo et al.'s code
[TBD: INSD runs, deferred in D145 at about 10 h a pair].
<!-- Sources: D145 "Protocol text for the paper"; amendments 1, 4; B5 deferral note. -->

### Table: what each method models

| | ours | Xu et al. 2025 | Buonomo et al. 2025 | ARAP | OT displacement | Large Steps / FlexiCubes |
|---|---|---|---|---|---|---|
| representation | MPM particles + grid | MPM particles + grid | neural SDF + velocity | triangle mesh, fixed connectivity | particles | mesh / SDF grid |
| genus change from a sphere | yes (dragon genus 15, teapot, bob) | yes | yes (bob frames genus 31–44) | no (genus 0 only) | n/a (no surface) | no / yes |
| mass | exact by construction | exact by construction | none (volume kept by the velocity's divergence) | none | exact (equal weights) | none |
| momentum | conserved by the stress control (measured drift below) | conserved by the stress control | not defined | not defined | not defined | not defined |
| contact, external forces | multi-body contact, floor, gravity (measured below) | not in our protocol [TBD: what the TVCG paper shows] | none | none | none | none |
| end at rest | yes (kinetic below 1e-5) | no (0.0028–0.95) | n/a | n/a | n/a | n/a |
| correspondences | none | none | none | sphere map (CMCF) | none (computed plan) | none |
| render loss | silhouette + shading, through the physics | none | none | none | none | image losses on the geometry directly (B12's setup [TBD]) |

<!-- Sources: d145_first_tables.md (B4, B5, B11, volume), README "What it conserves", D147. The Large Steps / FlexiCubes
column is from the papers [Nicolet et al. 2021, Sec. 5; Shen et al. 2023] until B12 reads. -->

### Table: measured results (300k; min–max over seeds where three exist)

| measure | ours | Xu et al. 2025 | Buonomo et al. 2025 | ARAP | OT |
|---|---|---|---|---|---|
| end silhouette IoU vs independent sample, bunny (floor 0.9893) | [TBD: B1 L, `gallery_ind` metric] | [TBD: same metric] | 0.9601 | 0.9874 | [TBD] |
| same, dragon (floor 0.9895) | [TBD] | [TBD] | 0.9016 | N/A (genus 15) | [TBD] |
| end silhouette IoU, own sample, bunny / dragon | 0.987–0.988 | 0.958–0.969 / 0.969–0.971 | — | — | — |
| thin parts uncovered, own sample | 3.5–4.5 % | 31–50 % | [TBD: thin measure on its frames] | [TBD] | [TBD] |
| yardstick silhouette (exterior), bunny | 7.6–8.0e-4 | 0.0090–0.0099 | [TBD] | [TBD] | [TBD] |
| display 1 − IoU front, bunny | 0.0091 (seed 97) [TBD: three seeds] | 0.039–0.042 | [TBD] | [TBD] | [TBD] |
| apart particles at p = 0.25 / 0.5 / 0.75, bunny | 90 / 139 / 305 | 6540 / 4522 / 4102 | [TBD: harness] | [TBD: harness] | [TBD: harness] |
| same, dragon | 1 / 2 / 15 | 14919 / 6785 / 3844 | [TBD] | N/A | [TBD] |
| occupancy volume over the morph, bunny | 0.986–1.028 (end 1.005) | 0.990–1.070 | 0.971–1.028 | 0.987–1.016 | 0.986–1.088 |
| same, dragon | 0.964–1.075 (end 1.015) | 1.013–1.294 | 0.896–0.985 | N/A | 0.964–1.277 |
| end kinetic energy (last record) | below 1e-5 | 0.0028–0.95 | — | — | — |
| wall clock a pair | bunny 7.9–8.9 min, dragon 17.2–17.8 min | 2.3–6.8 min | 44–46 min morph + 20–39 min per SDF (CPU) | [TBD: B11 timing] | [TBD: B6 timing] |
| largest GPU memory | [TBD: B1 logs] | [TBD] | 7–8 GB | CPU | [TBD] |

Notes on the table:

- Ours at the end on the own sample, thin coverage and kinetic energy come from the deliverable's bunny and dragon
  (D138 / D144). The yardstick and display come from B1. The harness rows come from D135's kept runs (SB, FB), read
  as a dry run [TBD: re-read on B1's L97 when the frames are kept].
- Buonomo et al. ran at one seed with the authors' defaults. Its teapot morph failed (a static field whose zero set
  holds both shapes at every t); that row is reported, not dropped.
- The occupancy routine reads 0.986 (bunny) and 0.964 (dragon) for our source sphere at t = 0, which has exactly the
  target's volume. This 1–4 % shape bias of voxel occupancy applies to every column.

<!-- Sources: d145_first_tables.md B4, B5/B11 table, volume table, harness dry run, B2 seed 97; D138 batch 7 (runtime);
D145 B1 bunny; D145 B5 "Morph training 44–46 min a pair (200k points, 7–8 GB), SDFs on the CPU 20–39 min each". -->

### What a physical body buys

Every particle carries mass, and the stress control cannot create momentum. The control enters the grid transfer
as a stress, and the B-spline weights sum the stress's contributions to zero net force and torque. Mass is therefore
exact by construction, and the body's momentum changes only by drag and by the few position updates below the grid
that carry no velocity. We measure what remains on the bunny over three seeds. The largest displacement of the centre
of mass over the morph is 2e-4 sampling pitches with the render terms and 1e-4 without. About four fifths of either
arm's drift is the push of the minimum-spacing projection, whose pairwise pushes over frozen neighbour lists need not
cancel exactly. The grid transfer adds 3–5e-5 pitches, the float32 floor of P2G / G2P, equal in both arms. The
surface offset u, which is rigid-free by construction, adds about 1e-10.
<!-- Source: D145 "Where the centre-of-mass drift comes from"; README "What it conserves". -->

The same body takes part in physics that a geometric interpolant cannot express.

- Two separately thrown spheres collide, part in the air, and each becomes a bunny that lands and tumbles. This is the
  two-body scene, with 2 × 150k particles in 96 windows and 44 min. Each body has its own grid velocity field, with
  gated multi-field contact after [Bardenhagen et al. 2000] `{bardenhagen2000granular}` [TBD: section reference].
  At the first contact the morph is 45 % complete (silhouette progress 0.446). The bodies part at 6.25 s, land at 8.9 s and 11.0 s, and end with an end error of
  0.021 on the scene's own measure (pass at ≤ 0.05) [TBD: define the end error in the paper's terms], tumbled by 26°
  and 47°.
- Under gravity with a floor, the 40k morph still reaches its target at G = ρgH/E = 0.02 on bunny, dragon and bob
  (silhouette IoU 0.9734, 0.9755, 0.9763; bob's is below its no-gravity range of 0.9797–0.9802), but not on the
  armadillo (0.5937).

We state the limit as found: in a world-fixed objective, a gravity strong enough to move the centre of mass faster
than the morph (G = 0.05) stops three of four targets. The thrown scene works because its objective is invariant to
each body's rigid motion [TBD: section reference].
<!-- Sources: D147 "The 300k s97 on the continuous target"; D145 B9 (iii). -->

The morph also ends at the body's own rest. When the controls are released the end barely moves: the last kinetic
record is below 1e-5 (1.7–3.4e-6 on the bunny over three seeds). Our prior F-control formulation, run on the same
particles, ends with 0.0028–0.95 and keeps moving. Its objective scores only the mass on the grid (a log mass residual
with a minimum-mass term), not the end's motion.
<!-- Sources: d145_first_tables.md B4 and "Ours (D138 / D144)"; D145 B1. -->

### Topology

Three of the six targets are not genus 0: the dragon has genus 15, and the teapot and bob have genus 1. A sphere
reaches them only if the morph opens holes. ARAP and any interpolation of fixed connectivity cannot do that, so ARAP
is reported N/A for them.

Render-only mesh optimisation has the same limit. Its authors state that it "cannot change the topology by punching
holes or melding overlapping geometry" [Nicolet et al. 2021]. They start bob from a torus, and Planck and the T-shirt
from spheres with holes cut by hand. A recent fix for high-genus inverse rendering needs a start primitive whose
genus already matches the target's [Gao et al. 2026] `{gao2026highgenus}`.

Level-set morphs change topology freely, but not always as the target intends:

- Buonomo et al.'s bob frames pass through genus 31–44 with up to 181 components before settling.
- Its teapot morph found a static field whose zero set contains both shapes at every t.

Our body opens the dragon's 15 handles and the teapot's and bob's holes physically. The material parts where the
transport pulls it apart, and the end is a single body.
[TBD: genus / components per frame of ours on the six meshes, same routine as B5's native meshes; the dragon's
crevice is the known exception (beads 3.2–4.4 s, Limitations).]
<!-- Sources: d145_first_tables.md B5/B11 table and B11 amendment; Nicolet et al. 2021 Sec. 5 and Sec. 4 (Fig. 7 text);
Gao et al. 2026 abstract and Sec. on initialisation. -->

### Thin features

Thin parts are where a morph shows whether it carries material or only a surface. On our own samples, 3.5–4.5 % of
the target's thin outer points have no particle within 1.5 spacings at the end. Our prior F-control formulation leaves
31–50 % uncovered on the same particles.

In the intermediate frames, the share of material outside the largest connected piece separates the methods:

- Ours: bunny 90 / 139 / 305 particles at p = 0.25 / 0.5 / 0.75; dragon 1 / 2 / 15.
- Our prior F-control formulation: 4102–6540 on the bunny and 3844–14919 on the dragon.
- Transport alone (OT displacement): it ends on the target sample (a median 0.57 pitches from the nearest target
  point), but its occupancy rises to 1.277 on the dragon as torn material spreads.

[TBD: apart particles of OT, Buonomo et al. and ARAP at matched p, harness on B5 / B6 / B11 frames.]
[TBD: a thin-feature comparison against the SDF morph's frames. The plan predicts rounder ears, horns and fingers for
the SDF morph; not yet measured.]
<!-- Sources: d145_first_tables.md B4 and harness dry run; D145 first readings (B6 bunny). -->

### The render terms, measured against their own λ = 0 twin

The render terms (silhouette and shading of the exterior surface discs) never move a particle directly. They reach the
body only through the adjoint of the simulation, as changes of the stress control dFc and of the outer layer's offset
u. To isolate their effect, every render result is paired with a twin: the same code, seed and particles, with the
render weight set to zero.

**On the 40k gallery (19 meshes × 3 seeds):**

- The render arm's silhouette IoU against the independent sample is above the twin's at the three-seed mean on 19 of
  19 meshes; all three seeds agree on 17.
- Paired Wilcoxon test over the meshes: p = 3.8e-6.
- The gains are small in absolute terms (+0.0005 to +0.0030), a median 24 % of the twin's gap to the sampling floor.

**At 300k on the bunny (three seeds, ranges apart), against the independent sample:**

- The render arm lowers the exterior silhouette residual by 44 % and the shading residual by 28 %.
- The display's 1 − IoU falls by 26–31 % (front, thin-feature crop, back).
- The body's net rotation falls by 39 %.
- Within the seed spread (no difference): silhouette IoU on the particles, chamfer, thin coverage, end kinetic energy,
  the number of windows and the wall clock.
- One measure is worse, and we report it: the centre-of-mass drift, 2e-4 against 1e-4 pitches. Its source is the
  minimum-spacing projection (see above), not the render gradient.

Mechanism. A per-window counterfactual (the same window from the same start, render weight zero) shows where the
render acts:

- Its gradient on dFc is 7–9× larger per particle on the outer layer, 10–15× on the silhouette rim and 8–10× on thin
  parts than in the bulk.
- It falls to about a tenth of the layer's value at two MPM cells depth, the B-spline's reach.
- It lowers the window's silhouette term by 8 %, 23 %, 60 % and 78 % at windows 5, 10, 20 and 40.
- It leaves the transport term within ±4 % and moves the centre of mass by at most 6e-6 pitches.

The render weight is set every window so that the render gradient is half the physics gradient's norm: one third of
the step. Gradient components that oppose the physics are removed [Yu et al. 2020] `{yu2020pcgrad}`.

[TBD: B1's five other meshes and the dragon's seeds; T1 is read on five seeds for bunny and dragon (amendment 3).]
<!-- Sources: d145_first_tables.md B3; D145 B1 bunny; B7 render-gradient proof; README "Gradients". -->

One design choice is visible in this comparison. The render terms read the exterior surface discs, not the particle
cloud. On the bunny at 300k (three seeds), this puts the display 16 % and the exterior yardstick 24 % ahead of a
cloud-read render. The cloud read is ahead on thin coverage (4.1–4.5 % against 4.6–4.8 %) and, at 40k, on the
particle silhouette IoU. Each read is ahead on the representation it optimises. We report both.
<!-- Source: D145 B2 three seeds; B3 ablations (no-render_exterior above V on 13 of 19). -->

### Where we lose

**Surface smoothness.** A neural SDF morph produces a watertight, smooth zero set at every frame. Our surface is the
exterior of a particle body. Relief below the sampling pitch is limited by the 300k sample. The display draws the
surface discs of Zhu and Bridson's field with Solenthaler et al.'s correction [Solenthaler et al. 2007]
`{solenthaler2007unified}`, which moves 0.4–5.2 % of the discs by a median 1e-4 pitches.
[TBD: one roughness measure read on both ours and Buonomo et al.'s frames. None exists in D145 yet.]

**Runtime and memory.**

- A 300k morph takes 7.9–8.9 min (bunny) and 17.2–17.8 min (dragon) on one RTX 6000 Ada.
- That is slower than our prior F-control formulation on the same particles (2.3–6.8 min), and slower than ARAP and
  OT, which are CPU- or seconds-scale [TBD: B11 / B6 timings].
- It is faster than the SDF morph at this scale: 44–46 min of morph training a pair at 200k points, plus 20–39 min
  per target SDF on the CPU.
- Our memory is much larger than the SDF morph's 7–8 GB. A sampled lower bound of 36.1 GB was recorded for the 300k
  dragon before the runtime work [TBD: largest GPU memory per mesh from B1's logs on the deliverable code].

**Volume on the common measure.** We do not claim better volume preservation on the occupancy measure. The claims
are separate:

- Mass is exact by construction.
- The body's physical volume, Σ J V0 from the carried J, stays within −0.24 % … +1.28 % of the target's over the
  bunny morph (1.0067 at the end) [TBD: the dragon from G97].
- On the occupancy routine shared by all methods, ours is comparable to the baselines: bunny 0.986–1.028, dragon
  0.964–1.075.
- There is an early transient while the surface is spread (the dragon peaks at 1.075 at τ 0.06–0.10). On the dragon,
  Buonomo et al.'s range is narrower early (0.896–0.985), while ours ends nearer 1 (1.015 against 0.897 at the end).

**ARAP on genus-0 targets.** Where ARAP applies (bunny, armadillo, homer), it reaches the target's end shape almost at
the sampling floor: silhouette IoU 0.9874, 0.9849 and 0.9884 against floors of 0.9893, 0.9883 and 0.9916. It also
keeps one closed surface throughout, as an interpolation of fixed connectivity must. We do not claim a better end
shape than ARAP on these meshes [TBD: ours against the independent sample on the same three meshes]. What ARAP does not
give is a body: no mass, no momentum, no contact, no rest. It also has no route to the other three targets.

### What we do not claim

- We do not claim the best end shape on every target: on genus-0 meshes ARAP is at the sampling floor.
- We do not claim better volume preservation on the occupancy measure.
- We do not claim speed or memory parity with geometric interpolants.
- The render terms' gains on the particle silhouette are small. Their effect is larger on the surface they read (the
  exterior discs) and, at 300k, on rotation and the display.

The claims are:

- a morph that is a simulated body throughout, with exact mass and momentum change bounded by the measured drift,
  that ends at rest and takes part in contact and external forces;
- topology change from a sphere to genus 15;
- thin parts carried as material;
- a render channel that steers that body through its physics, with its effect measured against its λ = 0 twin.

## Open [TBD] items in this draft

1. Ours against the independent sample at 300k (silhouette IoU, chamfer; the `gallery_ind` metric) on the six meshes,
   from B1's L runs. This fills the first two rows of the measured table and the ARAP comparison.
2. Xu et al., OT: the same independent-sample metric, plus the yardstick and display for every method on all six meshes.
3. The harness at matched p on the baselines' frames (B5, B6, B11): apart particles, volume, display holes. Ours
   re-read on B1's L97 / L97-dragon instead of D135's SB / FB.
4. B12: Large Steps and FlexiCubes from the sphere (the B12 agent). Their end metrics, genus reached, intermediate
   frames, timings.
5. INSD, the correspondence-free re-implementation of [Sang et al. 2025] (deferred in D145).
6. Genus and components per frame of ours on the six meshes (same routine as B5's native meshes).
7. A roughness measure read on both ours and Buonomo et al.'s frames.
8. A thin-feature comparison against Buonomo et al.'s frames.
9. The largest GPU memory per mesh from B1's logs on the deliverable code. The timings of B11 (ARAP) and B6 (OT).
10. The dragon's physical volume (Σ J V0, G97) and the dragon's B1 / B2 seeds.
11. The section reference for the two-body scene's rigid-invariant objective.
12. Buonomo et al. at seeds 2 and 3 (queued in D145).
13. The two-body scene's end-error definition in the paper's terms, and whether the TVCG paper [Xu et al. 2025] shows
    contact or external forces (the methods table says only "not in our protocol").
14. If the coordinator runs it: the force-controlled MPM baseline (same simulator, objective and render terms, control
    = a force field instead of dFc). It would give the "why stress control, not forces" row: centre-of-mass drift, |L|,
    end kinetic energy. Optionally NISE from Buonomo et al.'s repository as a second level-set morph.
