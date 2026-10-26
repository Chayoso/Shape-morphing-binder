- **D161, the render carves a relief that only its pictures carry: the 3D target a smooth mesh, the render's target
  pictures the same mesh with embossed letters (pre-registered 2026-10-21 HH:MM CDT server clock, before any run of
  it; the user, 2026-10-21: "색이나 surface를 깎아서 진짜 그걸 만들게 하는 건 어때"; the coordinator's addition the same
  evening: the dragon's scales and the bunny's wrinkles placed on the response curve, "anything by which the render
  changes the direction of the physics"; code: branch `wip/d161` = `wip/d156` fa115a5 (freeze-2026-10-15-rt1 +
  `--render_tgt`, default off; D156's gates (suite, digest, deterministic SAME) passed on these pipeline files, which
  D161 leaves byte for byte as they are: `git diff fa115a5 -- physmorph scripts/pipeline_run.py tests` is empty) +
  probe scripts; server `repo_d161` (git archive) / `repo_d161s` (+ `tmp/d145/patch_s.py`, kept frames), `tmp/d161/`,
  `output/gpu/d161/`; scripts `scripts/probes/paper/d161/`; local `output/results_2026-10-21/d161_relief/` with a
  Korean README).**
  *Why.* Until now the render's pictures carried what the 3D target carries, so the render only polished rims
  (D152–D155, D157). D156 (an image-only target) could not carve concavities: the transport holds the hull's mass.
  D158: the in-loop pictures are as blurred as the run's own surface. Here the 3D target (transport, surface
  proximity, cleanups, the relaxation's relief reference) stays a smooth mesh, and the render's pictures are drawn,
  by the same operators, from the same mesh carrying an embossed pattern. "dF_c + layer u" (the λ = 0 twin) must end
  smooth; "dF_c + layer u + render loss (full)" must carve the pattern into its own particles. The user's standing
  directive (2026-10-07): sub-pitch relief from the physics, not a display layer. This is that test with a relief no
  other term asks for.
  *Papers (read before the design).* Alexa and Matusik 2010 (Reliefs as Images): a Lambertian height field fitted to
  images reads through slopes; a sub-pixel relief is fragile; the target's mean brightness must equal the flat
  surface's or the fit moves the whole shape (our relief mesh is volume-neutral: h = δ (P − P̄)). Weyrich et al. 2007
  (digital bas-relief): perceived shape lives in the gradients, depth can be compressed by orders of magnitude;
  readability is slope 2πδ/w, not δ. Mitra and Pauly 2009 (Shadow Art): a silhouette constrains only the outline;
  relief shows in a silhouette only where it crosses the rim, and only above a pixel. Belhumeur et al. 1999 (the
  bas-relief ambiguity) and Pentland 1990: under a headlight, a front-facing patch's shading is second order in the
  slope (I ∝ 1/√(1 + |∇h|²), the same for h and −h); first order only where the light is oblique; our loss's light is
  the headlight of each of 18 views, so a patch reads at first order only in the views that see it obliquely
  (34–81° for a face of the coin). Nicolet et al. 2021 (Large Steps): a Laplacian penalty costs k² per mode, a
  bi-Laplacian k⁴; a smoothing step behaves like a low-pass of 1/(1 + λμ_k); per-coordinate Adam undoes gradient
  smoothing. The outer-layer relaxation (d − d̄ relaxed at 1/T a step) is such a bi-Laplacian low-pass in spacings,
  and the exterior field (radius 3 pitches) a further low-pass that both the run and its pictures pass through.
  Implications fixed before the curve: design the pattern by slope; expect a band-pass (short wavelengths cut by the
  field, the pictures and the relaxation; long ones held by the transport, which sees mass moved over cells); measure
  each term's hold as a function of (δ, w) before choosing the scale.
  *Step 1, the response curve (`d161_curve.py`; measured before this entry, the pipeline's own operators on the CPU,
  float rounding only: the device KD-tree, distance transform and layer kNN by scipy).* ...
