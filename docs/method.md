# PhysMorph method contract (v3)

This file is what the code cites as `docs/SPEC.md` (historical name). **Equation numbers
(1)–(14) and section anchors §3.3 / §4.2 are load-bearing** — docstrings in
`physmorph/mpm/*` and `physmorph/losses/*` reference them; keep them stable.

---

## §1 Continuum + discretisation

Elastoplastic MLS-MPM, fixed corotated elasticity, multiplicative plasticity. Particle
state: x, v, APIC affine C, total deformation F, plastic rest state Fp, per-particle Lamé.
Domain grid: `MPMParams` (default dx=0.5, 64³, dt=1/240, F-smoothing 0.955 — the legacy
morphing line). Oracle: `legacy/DiffMPMLib3D` (Xu et al.), ported kernel-for-kernel.

## §2 Constitutive (code: mpm/constitutive.py)

```
(1)  λ = Eν/((1+ν)(1−2ν)),  μ = E/(2(1+ν))
(2)  ψ(F) = μ Σ(σ_i−1)² + λ/2 (J−1)²                     (fixed corotated energy)
(3)  P(F) = 2μ(F−R) + λ(J−1)J F^{−T},  R from proper SVD (reflection-repaired)
```

## §3.3 One MPM step (code: mpm/kernels.py, mpm/step.py)

```
(3') F_e = (F + dFc)·Fp^{−1};  stress uses P(F_e)         ← the CONTROL enters here
(4)  P2G mass:      m_g = Σ_p w_gp m_p                     (cubic B-spline 4³ stencil)
(5)  P2G momentum:  mom_g = Σ_p w_gp [m_p v_p(1−dt·drag) + G_p(x_g−x_p)],
                    G_p = −C0·dt·V_p·P(F_e)(F+dFc)^T + m_p C_p,  C0 = 3/dx²
(6)  grid:          v_g = mom_g/m_g + dt·f_ext             (+ floor contact, optional)
(7)  G2P:           v_p = Σ w v_g;  C_p = C0 Σ w v_g(x_g−x_p)^T  (+ η damping: eta_sym
                    damps only sym(C) — objective; eta_mode 1 = exp(−dt·η), dt-consistent)
(8)  v_max clamp    OFF in every blessed path (prm.v_max = 0)
(9)  update:        F ← blend[(I + dt·C)(F + dFc), F; smoothing];  x ← x + dt·v
(10) volumes:       V_p = m_p/ρ_p from a one-time mass P2G at rest
```

## §4.2 Differentiable rollout (code: mpm/traj.py, mpm/function.py)

Per-step arrays on a `wp.Tape`; a torch `autograd.Function` bridges leaves
`{dFc[t] (T,N,3,3), λ_i, μ_i (N,)}` → `(x_T, F_T, v_T)`. One backward yields every
per-layer control gradient at once (the reason autodiff beats the layer-by-layer C++).
Grads are read per-leaf only if that input required them. Constant-sequence ≡ shared
control is gate G1a; a finite-difference check of dL/ds is gate G1b.

## §5 Dynamic formulation (v2 blessed path; code: pipeline/optimizer.py, runner.py)

Window loss over horizon T (terminal):

```
(13) D_vol   = ½ Σ_cells [log(m+1) − log(m_tgt+1)]²        (mass matching, Xu et al.)
(14) D_render= mean_views mean_px [w_hole·relu(α_t−α)² + w_spray·relu(α−α_t)²],
               α = 1−exp(−k·CIC(x));  views = azimuth ring × elevations {0,±φ}
L = D_vol + λ_R·D_render + w_kin·mean|v_T|² + w_ctrl·Σ|dFc|²/(TN)
    + w_box·mean relu(|x_T|−r_box)² + w_mat·mean|s|²
```

- **λ_R** = α_λ·‖∇phys‖/‖∇render‖, estimated ONCE per window (single objective for the
  line search), EMA across windows; α_λ=0 ⇒ the physics-only arm, same code path.
- **Four render→physics channels**: dFc (adjoint), material s=(s_λ,s_μ) with
  Lamé = base·eˢ (same adjoint), Fp (assimilation of the optimised motion, §8), v_T
  (arrive-at-rest term).
- **Step control**: hand-rolled Adam over leaves + backtracking line search (reject +
  restore leaves AND moments) + adaptive α; acceptance requires a FINITE rollout state
  (NaN particles vanish from splats and can fake a lower loss). Candidate evaluations use
  a plain no-tape rollout.
- **Outer loop**: FULL state promotion (x, F repaired-and-counted, v, C); plastic
  assimilation; plateau freeze on RAW components (λ-free); guard counters must read zero.
- **The box leash** exists because pixels are local: a particle outside every viewport and
  the loss grid has exactly zero data gradient — escape must remain visible to the
  objective.

## §6 Sobolev / grid-GS render direction (v3; code: pipeline/grid_smooth.py)

The raw ∂D_render/∂x_T is a Jacobi-style signal (all particles react to the same pixels at
once; zero in coverage pockets; high-frequency dominated). Before the adjoint pullback we
optionally replace it with a Sobolev-metric gradient: scatter to the loss grid (CIC),
red-black screened-diffusion sweeps `(I+κL)u = ĝ`, gather, rescale to the raw norm. The
smoothed field seeds `∂x_T`-backward, so the pullback to dFc still goes through the exact
MPM adjoint — this is a **search-direction transform, not a physics change**; nothing here
needs to be differentiable. Lineage: Sobolev preconditioning (Repulsive Curves) and
Preconditioned Deformation Grids (PG 2025). Arm: `render_gs` (+ warm-started dFc).

## §7 VBD-MPM quasi-static arm (v3; code: vbd/solver.py, pipeline/runner_vbd.py)

Morphing is quasi-static at heart (the dynamic path spends w_kin forcing rest). The VBD
arm makes that structural: each commit solves, on the ACTIVE grid nodes (CIC stencil
frozen at commit start; rebinned between commits),

```
u* = argmin_u  Σ_p V_p ψ_SNH((I+∇u(x_p))·F·Fp^{−1}) + D_vol(x+u) + λ_R·D_render(x+u)
               + w_box·leash
```

ψ_SNH = stable Neo-Hookean (Smith 2018; SVD-free, defined for all J — the energy VBD
itself uses). Solver = VBD transplanted from mesh vertices to **grid nodes**: 2-color
parity blocks, per-node diagonal elastic preconditioning, per-color backtracking on the
total energy, a trust-region-like global step that grows 1.5× on clean sweeps (the data
terms' curvature is unknown a priori), stop at ‖∇E‖ ≤ tol·‖∇E₀‖. Then x ← x+u,
F ← (I+∇u)F, assimilation as in §8. Rest is by construction — no velocities exist.

Render information reaches the physics as a genuine **energy of the equilibrium**; λ_R is
balanced per commit from gradient norms on u. Differentiability (for a future material/
system-ID channel): validated in `scripts/probe_gs_differentiability.py` — unrolled and
IFT-adjoint gradients both exact (<0.1%) at convergence, biased when unconverged ⇒ solve
tolerance is a *correctness* gate.

## §8 Plastic assimilation (both families; code: plasticity/assimilation.py)

```
F_e = F·Fp^{−1} = R_e·S_e  (per-particle polar) ⇒ Fp ← clamp_sv(S_e^η·Fp)
⇒ F_e_new = R_e·S_e^{1−η}   EXACTLY
```

Rigid motion is a strict no-op; dilation is assimilated in full; fixed-corotated energy
decreases monotonically (tested). Rejected precursors (measured, in git history): OT
`update_fp` (symmetrised Jacobian fabricates strain from rotation; isochoric ⇒ blind to
dilation) and displacement-field polar assimilation (mismatches the dFc-inflated F —
commit-boundary stress spikes, kin 66→509).

## §9 Conditioning & guards (code: mpm/conditioning.py)

`condition_F(clamp=False)`: repairs ONLY non-finite rows and SVD reflections, both
counted; healthy F returns bit-exact. No silent singular-value projection anywhere in a
blessed path. Guard counters (domain clamp, NaN x/v/C, F resets/flips, any-step
inversions) are containment + telemetry: **a fired counter invalidates the run** (gate G2).
