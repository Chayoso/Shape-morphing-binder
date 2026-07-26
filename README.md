# PhysMorph-GS — Physically-Based, Texture-Coherent Morphing of Gaussian Assets

A textured 3D-Gaussian asset (a checkered sphere) morphs into a target shape as a
**physically-based, injective elastic deformation** (det F > 0 at *every* frame), rendered as a
**smooth shaded surface in pure 3D Gaussian Splatting**. The texture rides the deformation gradient;
thin features (bunny ears, armadillo limbs) stay smooth; nothing scrambles, nothing floats, no mesh.
**Physics governs the motion**, not just the endpoint: the animation is an *even-paced quasi-static
continuation* in which every intermediate frame is an elastic equilibrium — not an interpolation.

> 📖 **The paper is MatCast** → start at [`docs/matcast_paper_spec.html`](docs/matcast_paper_spec.html)
> (pipeline + equations + required results), then [`docs/README.md`](docs/README.md) for the doc index.
> Numbers: [`docs/matcast_results.md`](docs/matcast_results.md) · Engine equations:
> [`docs/method.md`](docs/method.md) · Earlier morphing line: [`docs/legacy/`](docs/legacy/).

## Layout
```
physmorph/        core library
  plasticity/     sliced-OT / Sinkhorn / auction baselines (sliced_ot_displacement, ...)
  sampling/       mesh load + volume sampling (load_mesh, sample_volume)
  render/         3DGS rasteriser wrapper (diff_gauss / gsplat) + Σ=FΣ₀Fᵀ covariance
  surface/        differentiable density-based surface weight
  viewer/         3DGS .ply export
  mpm/            Warp differentiable MPM (earlier rewrite; infrastructure, NOT the headline path)
  morph.py · morph_physical.py · style_transfer.py
scripts/          experiment drivers (see Quickstart + docs/README §scripts)
docs/             paper-shaped writeup (README, method, results, related_work)
assets/           source/target meshes (isosphere, bunny, armadillo, spot, heart, bob, dragon, ...)
tests/            unit tests (gradcheck, OT, kernels, silhouette, viewer)
output/           figures/ + per-experiment results (gitignored)
legacy/           archived C++ DiffMPM pipeline (pre-rewrite; kept for reference)
```

## Quickstart
```bash
# env: torch (cu12x) + scipy + trimesh + matplotlib  (+ a CUDA 3DGS rasteriser: diff_gauss or gsplat)

# HEADLINE — physics-in-motion smooth-surface morph (every frame an elastic equilibrium)
python scripts/morph_surfel_physical.py --target bunny      # --render gs (3DGS hero) | preview (rasteriser-free)

# central-claim benchmark (TexCoherence-Bench): guarded ours vs OT / ARAP / SLIM, seeds + figures
python scripts/morph_bench.py --targets bunny armadilo spot heart bob --seeds 3

python -m pytest tests/ -q                                  # unit tests
```

## Status
- ✅ **Injective elastic morph** (det F > 0, Theorem 1); smooth GS surfaces *incl. thin features*, no floaters.
- ✅ **Physics in the motion** — even-paced continuation; every frame an elastic equilibrium (verified min det F > 0).
- ✅ **M1 central-claim benchmark DONE** (n=40000, 5 targets × 3 seeds, on hyde01) → `output/texbench_main/`.
  ours & SLIM are the *only* flip-free-at-reach methods on every target; ours **ties SLIM** on all
  benchmark metrics ⟹ the uniqueness is **material control**, which is why the TOG-grade direction is
  **MatCast** (material-as-style — see `docs/matcast_plan.md`), not morphing.
- 🔜 MatCast make-or-break (Exp 1 recover + Exp 2 cross-asset transfer) needs the MPM engine (hyde06's
  working warp; hyde01's warp is broken) · 3DGS-hero render · modern baseline (VisionLaw/NeuMA/SemMorph3D).

The **honest claim** (see [`docs/results.md`](docs/results.md) §2): the win is *injectivity-at-reach* +
*physical material control*, **not** NP-magnitude dominance — only ours & SLIM reach the target flip-free,
and material control (E, ν, anisotropy) is the differentiator no morphing competitor offers.

---

<sub>Historical note (earlier Warp-MPM rewrite, `physmorph/mpm/`): a single differentiable autograd graph
(`L = D_vol + λ·D_img`) was built to remove a structural render-loss gate in the legacy C++ pipeline. Key
lesson: Warp autodiff mishandles **in-place** array read-writes — an in-place `grid_op` inflated MPM
gradients ~5000×; making it out-of-place fixed gradcheck to 0.1%. The MPM package remains as infrastructure;
the current headline pipeline is the injective elastic + surfel-GS morph above.</sub>
