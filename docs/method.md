  # Method & Architecture — Physically-Based, Texture-Coherent Morphing of Gaussian Assets

  > **STATUS (2026-07-27).** This file is the `docs/SPEC.md` that the engine docstrings reference
  > (`physmorph/mpm/*.py`, `physmorph/__init__.py` — the file was renamed, the docstring paths were
  > not). **The numbered equations are current and authoritative for the engine.** The *identity /
  > framing* sections below describe the earlier morphing line and are superseded by
  > `matcast_paper_spec.html` §1 (MatCast forward model) — the paper is MatCast; the morph survives
  > only as a disclosed application (audit: `matcast_results.md` §7f).

  *The implementation contract and the math source of truth. Every equation is numbered, derived,
  and mapped to the code (`file.py:function`). Gradient flow is given explicitly at each stage.*

  **Identity.** A **physically-based, injective elastic deformation** of a textured Gaussian body to a target: physics (material hyperelastic energy + a guaranteed-injective *guarded step*, §13) governs *how* it deforms; the texture rides the deformation gradient $F$; the result renders as a **smooth surface in pure 3DGS** (flat surfels, §14). The data term — *what* to become — is a 3-D target by default (all headline results); a differentiable multi-view **render** (image-only) supervision is a validated alternative (§5b). Material control (§4) and render-supervision are demonstrated extensions, not the headline.

  ---

  ## 0. Notation & data flow

  | symbol | meaning |
  |---|---|
  | $N$ | number of source Gaussians |
  | $x_i\in\mathbb R^3$ | rest (source) position of particle $i$ |
  | $u_i\in\mathbb R^3$ | **decision variable** — displacement; deformed position $y_i=x_i+u_i$ |
  | $c_i$ | Lagrangian colour (carried with the particle) |
  | $\mathcal N(i)$ | $k$ nearest neighbours of $i$ on the rest shape |
  | $F_i\in\mathbb R^{3\times3}$ | per-particle deformation gradient $\approx \nabla(x+u)\big|_i$ |
  | $W(\cdot)$ | hyperelastic energy density; $P=\partial W/\partial F$ first Piola–Kirchhoff stress |
  | $B=\{b_j\}$ / $\{I_v\}$ | target as a 3-D point set / as rendered images from views $\pi_v$ |
  | $\Sigma_i=\sigma_0^2 F_iF_i^\top$ | rendered Gaussian covariance (texture rides $F$) |

  **Pipeline (one optimisation step).**

  ```
              precompute (rest, no grad)                 per iteration (autograd)
  x ──kNN──► E, dx_ij, M_i, M_i^{-1}, V_i      u ─►F(u)─►W,P─►E_mat ┐
                                                │                    ├─► L ─►∇_u L ─► Adam ─► u
                                                └►y=x+u─►Data(y) ────┘
  final: F ─► Σ=σ0²FFᵀ ─► (scales,quats | cov3D) ─► 3DGS raster ─► image
  ```

  Two gradient paths reach $u$: the **physics path** $u\!\to\!F\!\to\!W\!\to\!E_{\text{mat}}$ and
  the **supervision path** $u\!\to\!y\!\to\!\text{Data}$ (3-D Chamfer or differentiable render).

  ---

  ## 1. Representation & variables

  The source is $N$ Gaussians at rest positions $x_i$ with isotropic rest covariance
  $\Sigma_0=\sigma_0^2 I$ ($\sigma_0$ from mean nearest-neighbour spacing) and Lagrangian colours
  $c_i$. The only optimisation variable is the displacement field $u\in\mathbb R^{N\times3}$,
  initialised from a heavily-smoothed sliced-OT field (a coarse global alignment).

  *Code:* `sample_volume`, `sigma0_from_nn` (`physmorph/render/covariance.py`); init
  `_smooth_field(sliced_ot_displacement(...))`.

  ---

  ## 2. Stage A — rest-shape operators (precompute, no gradient)

  For each particle build the $k$-NN graph and the **moving-least-squares (MLS) shape matrix**:

  $$
  dx_{ij}=x_j-x_i,\qquad
  M_i=\frac1k\sum_{j\in\mathcal N(i)} dx_{ij}\,dx_{ij}^\top+\varepsilon I
  \;\in\mathbb R^{3\times3}. \tag{A.1}
  $$

  $M_i$ is symmetric positive-definite when the neighbours span $\mathbb R^3$ (the $\varepsilon I$
  ridge guarantees invertibility). $M_i^{-1}$, the edge list $E=\{(i,j)\}$, the rest edge vectors
  $dx_{ij}$, and rest volumes $V_i$ (taken uniform) are precomputed **once**.

  *Code:* `morph_material_neo.py:morph` — `cKDTree(src).query`, then
  `M = index_add(DX⊗DX)/k`, `Minv = inv(M+εI)`.

  ---

  ## 3. Stage B — deformation gradient $F$ via MLS (differentiable)

  Displacement differences $du_{ij}=u_j-u_i$ define

  $$
  B_i(u)=\frac1k\sum_{j\in\mathcal N(i)} du_{ij}\,dx_{ij}^\top,\qquad
  J_i=B_iM_i^{-1},\qquad
  \boxed{\,F_i=I+B_iM_i^{-1}\,}. \tag{B.1}
  $$

  $F_i$ is **linear in $u$**, hence cheaply differentiable.

  **Proposition 1 (MLS exactness for affine fields).** *If the displacement is affine on
  $\mathcal N(i)$, $u(x)=a+J(x-x_i)$, then $F_i=I+J$ exactly (as $\varepsilon\to0$).*

  *Proof.* Affinity gives $du_{ij}=J\,dx_{ij}$. Then
  $B_i=\frac1k\sum_j J\,dx_{ij}dx_{ij}^\top=J\big(\frac1k\sum_j dx_{ij}dx_{ij}^\top\big)=JM_i$
  (at $\varepsilon=0$), so $J_i=B_iM_i^{-1}=J$ and $F_i=I+J=\nabla(x+u)$. $\;\square$

  So $J_i$ is the least-squares estimate of $\nabla u$: it minimises
  $\sum_{j}\lVert du_{ij}-J_i\,dx_{ij}\rVert^2$, whose normal equations
  $J_i\big(\sum dx\,dx^\top\big)=\sum du\,dx^\top$ give exactly (B.1).

  **Gradient of $F_i$ w.r.t. $u$.** From (B.1), for any matrix cotangent $\bar F_i$,
  $$
  \big\langle \bar F_i,\;dF_i\big\rangle
  =\frac1k\sum_{j\in\mathcal N(i)}(du_j-du_i)^\top \underbrace{\bar F_i M_i^{-\top}dx_{ij}}_{\in\mathbb R^3},
  \tag{B.2}
  $$
  i.e. the vector-Jacobian product distributes onto the two endpoints $u_i,u_j$ of each edge
  (used in Prop 5). *Code:* `Ffield(u)` — `B = index_add(DU⊗DX)/k; F = I + bmm(B, Minv)`; the VJP
  is produced by autograd.

  ---

  ## 4. Stage C — material energy $W(F)$ (the physics; differentiable)

  **Neo-Hookean** energy density, parameterised by Young's modulus $E$ and Poisson ratio $\nu$:

  $$
  W(F)=\frac\mu2\big(I_C-3\big)-\mu\ln J+\frac\lambda2(\ln J)^2,\quad
  I_C=\operatorname{tr}(F^\top F)=\lVert F\rVert_F^2,\;\; J=\det F, \tag{C.1}
  $$
  $$
  \mu=\frac{E}{2(1+\nu)},\qquad \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}. \tag{C.2}
  $$

  Total elastic energy $E_{\text{mat}}(u)=\sum_i V_i\,W(F_i)$.

  **Proposition 2 (first Piola–Kirchhoff stress).**
  $$
  P(F)=\frac{\partial W}{\partial F}=\mu\big(F-F^{-\top}\big)+\lambda(\ln J)\,F^{-\top}. \tag{C.3}
  $$

  *Proof.* $\dfrac{\partial I_C}{\partial F}=2F$. For the determinant, Jacobi's formula gives
  $\dfrac{\partial \det F}{\partial F}=\det F\,F^{-\top}=J F^{-\top}$, so
  $\dfrac{\partial \ln J}{\partial F}=\dfrac1J\,J F^{-\top}=F^{-\top}$. Therefore
  $\dfrac{\partial W}{\partial F}=\dfrac\mu2(2F)-\mu F^{-\top}+\lambda(\ln J)F^{-\top}
  =\mu(F-F^{-\top})+\lambda(\ln J)F^{-\top}.\;\square$

  **Proposition 3 (physical injectivity — no fold-over).** *Every finite-energy configuration has
  $J_i=\det F_i>0$ for all $i$; hence the piecewise-linear deformation is locally
  orientation-preserving and locally injective.*

  *Proof.* In (C.1), $\frac\mu2(I_C-3)\ge-\frac{3\mu}2$ (since $I_C\ge0$) and
  $\frac\lambda2(\ln J)^2\ge0$ are bounded below, while $-\mu\ln J\to+\infty$ as $J\to0^+$. Thus
  $W(F)\to+\infty$ as $J\to0^+$, and $W=+\infty$ for $J\le0$ (the log barrier). Any minimiser (or
  any iterate with finite energy) therefore satisfies $J_i>0\;\forall i$. A continuous deformation
  whose Jacobian is everywhere orientation-preserving is locally injective; in particular no local
  fold-over occurs and the texture cannot overlap itself. $\;\square$

  > This is why injectivity is **physical, not a penalty**: the $-\mu\ln J$ barrier of a real
  > hyperelastic material is exactly the term that forbids the texture from folding. (Global
  > injectivity additionally requires non-self-collision; enforced empirically by the data
  > schedule.)

  **Anisotropy (optional).** A fibre $a$ ($\lVert a\rVert=1$) stiffened by $k_a$:
  $$
  W_{\text{aniso}}(F)=\tfrac12 k_a\mu\big(\lVert Fa\rVert^2-1\big)^2,\qquad
  \frac{\partial W_{\text{aniso}}}{\partial F}=2k_a\mu\big(\lVert Fa\rVert^2-1\big)(Fa)\,a^\top. \tag{C.4}
  $$
  It penalises stretch along $a$, producing preferential deformation (and directional texture
  stretch) orthogonal to $a$.

  *Code:* `morph_material_neo.py:morph` — `J=det(F); Ic=(F*F).sum; W = .5μ(Ic-3) - μ logJ + .5λ logJ²`
  (+ aniso term). Autograd produces $P(F)$; we never hand-code (C.3).

  ---

  ## 5. Stage D — supervision (the data term)

  ### 5a. 3-D target — symmetric Chamfer

  $$
  \mathrm{Data}_{3D}(y)=\frac1N\sum_i\min_j\lVert y_i-b_j\rVert^2
  +\frac1M\sum_j\min_i\lVert b_j-y_i\rVert^2. \tag{D.1}
  $$

  **Proposition 4 (exact gradient via Danskin / stop-grad argmin).** *Let
  $\phi(y_i)=\min_j\lVert y_i-b_j\rVert^2$ and $j^\*(y_i)=\arg\min_j$. Where $j^\*$ is unique,
  $\phi$ is differentiable with $\nabla_{y_i}\phi=2\,(y_i-b_{j^\*})$.* Hence computing the
  nearest-neighbour index **without gradient** and differentiating only the gathered distance is
  exact almost everywhere.

  *Proof.* $\phi=\min_j\psi_j$ with $\psi_j$ smooth; by Danskin's theorem the subdifferential is
  the convex hull of $\{\nabla\psi_j:j\in\arg\min\}$, a singleton $\{2(y_i-b_{j^\*})\}$ when the
  minimiser is unique (a.e.). $\;\square$

  *Code:* `chamfer(p,q)` — `with no_grad: nn=argmin cdist(...)`; loss `(p-q[nn]).square().sum`.
  Memory is $O(N)$ (no retained pairwise matrix); the chunked argmin temporaries are freed.

  ### 5b. Render target — differentiable 3DGS rasterisation (a validated capability demo, *not* the headline)

  > **Scope (one role only).** Render-supervision is a **demonstrated capability** — image-only
  > (multi-view silhouette) target is differentiable end-to-end and its gradient is verified
  > non-zero — but the signal is **weak** (grad-norm $\sim10^{-3}$, silhouette-only) and it is **not**
  > a contribution pillar. The headline is the injective hyperelastic *motion* (§13–§14) with a 3-D
  > target. This subsection documents that the pipeline *can* be driven by images; nothing in the
  > main results depends on it.

  For each view $v$ with world→view $W_v$ and intrinsics, render the **deformed** Gaussians and
  match the target's image:
  $$
  \hat I_v(u)=\mathcal R\big(\{y_i\},\{\Sigma_i\},\{c_i\},\{o_i\};\pi_v\big),\qquad
  \mathrm{Data}_{\text{render}}(u)=\sum_v\big\lVert \hat I_v(u)-I_v^{\text{tgt}}\big\rVert^2. \tag{D.2}
  $$

  **3DGS forward (EWA splatting).** Camera-space centre $t=W_v\,[y_i;1]$; screen projection
  $\mu^{2D}_i=\pi(t)$. The 2-D covariance is the pushforward of $\Sigma_i$ through the projection
  Jacobian $J^{\pi}$ and view rotation $\mathcal W$:
  $$
  \Sigma^{2D}_i=J^{\pi}\,\mathcal W\,\Sigma_i\,\mathcal W^\top J^{\pi\top},\qquad
  J^{\pi}=\begin{bmatrix}f_x/t_z&0&-f_xt_x/t_z^2\\[2pt]0&f_y/t_z&-f_yt_y/t_z^2\end{bmatrix}. \tag{D.3}
  $$
  Per pixel $p$, opacity-weighted Gaussian footprint and front-to-back $\alpha$-compositing:
  $$
  \alpha_i(p)=o_i\exp\!\Big(-\tfrac12(p-\mu^{2D}_i)^\top(\Sigma^{2D}_i)^{-1}(p-\mu^{2D}_i)\Big),\quad
  C(p)=\sum_i c_i\,\alpha_i(p)\!\prod_{l<i}(1-\alpha_l(p)). \tag{D.4}
  $$

  The CUDA rasteriser supplies the **adjoint** $\partial C/\partial\{\mu^{2D},\Sigma^{2D},c,o\}$;
  the chain rule to $u$ closes through (D.3) and the geometry coupling
  $$
  \frac{\partial \hat I_v}{\partial u}
  =\frac{\partial \hat I_v}{\partial \mu^{2D}}\frac{\partial \mu^{2D}}{\partial y}\,\frac{\partial y}{\partial u}
  +\frac{\partial \hat I_v}{\partial \Sigma^{2D}}\frac{\partial \Sigma^{2D}}{\partial \Sigma_i}\frac{\partial \Sigma_i}{\partial F}\frac{\partial F}{\partial u},
  \quad \frac{\partial y}{\partial u}=I,\;\; \Sigma_i=\sigma_0^2F_iF_i^\top. \tag{D.5}
  $$
  For **silhouette** supervision we render white opaque isotropic Gaussians so $C$ is the coverage
  mask; the shape gradient then flows purely through $\mu^{2D}(y)=\mu^{2D}(x+u)$ — robust and
  needing no covariance differentiation. **Verified:** a render→backward self-check yields a
  finite, non-zero $\partial\,\mathrm{loss}/\partial u$ (grad-norm $\sim10^{-3}$).

  *Code:* `morph_render_driven.py` — `make_views`, `render_sil`, and `selfcheck`; rasteriser via
  `diff_gauss` (hyde06) or `diff_gaussian_rasterization` (hyde01), same `physmorph.render` path.

  ---

  ## 6. Stage E — objective, elastic forces, optimisation

  $$
  \boxed{\;L(u)=\underbrace{\sum_i V_i\,W(F_i)}_{E_{\text{mat}}\;(\text{physics})}
  \;+\;w_{\text{data}}\,\underbrace{\mathrm{Data}(y)}_{\text{render and/or 3-D}}\;} \tag{E.1}
  $$

  **Proposition 5 (discrete elastic force = gradient of $E_{\text{mat}}$).** *With
  $P_i=P(F_i)$ and $q_{ij}=\frac1k\,V_i\,P_iM_i^{-\top}dx_{ij}\in\mathbb R^3$,*
  $$
  \frac{\partial E_{\text{mat}}}{\partial u_j}=\!\!\sum_{i:\,j\in\mathcal N(i)}\!\! q_{ij},\qquad
  \frac{\partial E_{\text{mat}}}{\partial u_i}=-\!\!\sum_{j\in\mathcal N(i)}\!\! q_{ij}. \tag{E.2}
  $$

  *Proof.* $dE_{\text{mat}}=\sum_iV_i\langle P_i,dF_i\rangle$. Substituting (B.2):
  $\langle P_i,dF_i\rangle=\frac1k\sum_j(du_j-du_i)^\top P_iM_i^{-\top}dx_{ij}$ (using
  $\langle P_i,\,w\,dx^\top M^{-1}\rangle=w^\top P_iM^{-\top}dx$). Collecting the coefficient of
  each $du_j$ and $du_i$ gives (E.2). These are equal-and-opposite **forces along graph edges** —
  the discrete analogue of $\operatorname{div}P$. $\;\square$

  The **total** gradient is $\nabla_uL=\nabla_uE_{\text{mat}}+w_{\text{data}}\nabla_u\mathrm{Data}$
  (Prop 5 + Prop 4/(D.5)), assembled automatically by autograd. We **anneal** $w_{\text{data}}$
  (coarse→fine): elastic-led global shape first, then pulled onto the target, so the *material*
  genuinely shapes the correspondence rather than being overridden. Optimiser: **Adam** on $u$.

  *Code:* `morph_material_neo.py:morph` — `(W.mean() + w_data*data).backward(); opt.step()` inside
  the $w_{\text{data}}$ schedule.

  ---

  ## 7. Stage F — texture rides $F$ (rendering & coherence)

  The rendered covariance is $\Sigma_i=\sigma_0^2F_iF_i^\top$ (eq. D.5); colour $c_i$ is Lagrangian.

  **Proposition 6 (texture coherence under injective $F$).** *If $J_i>0$ (Prop 3), neighbouring
  rest particles stay neighbours: for $j\in\mathcal N(i)$,*
  $$
  \sigma_{\min}(F_i)\,\lVert x_i-x_j\rVert\;\le\;\lVert y_i-y_j\rVert\;\le\;\sigma_{\max}(F_i)\,\lVert x_i-x_j\rVert,
  \quad \sigma_{\min}(F_i)>0. \tag{F.1}
  $$
  *Hence the colour field is **advected and stretched**, never remixed: no two far-apart rest
  particles collide, so adjacent rendered colours come from adjacent rest colours (a coherent
  texture). This is exactly what optimal transport — measure-preserving but non-injective —
  violates, producing colour noise at matched reach.*

  *Proof.* To first order $y_i-y_j\approx F_i(x_i-x_j)$; the singular-value bounds give (F.1).
  $J_i=\prod\sigma(F_i)>0\Rightarrow\sigma_{\min}>0$, so distinct neighbours never coincide and the
  local map $F_i$ is a bijection of the neighbourhood. $\;\square$

  *Code:* `cov_from_F`, `clampF`, `render_3dgs` (`physmorph/render/`); the morph metric
  `color_coh` / `nbr_preserve` measure (F.1) empirically.

  ---

  ## 8. Gradient-flow summary

  $$
  \frac{\partial L}{\partial u}
  =\sum_i \underbrace{\frac{\partial F_i}{\partial u}^{\!\top}\!\big[V_iP(F_i)\big]}_{\text{physics (Prop 2,5)}}
  \;+\;w_{\text{data}}\Big(
  \underbrace{\tfrac{\partial \mathrm{Data}_{3D}}{\partial u}}_{\text{Prop 4}}\;\text{or}\;
  \underbrace{\tfrac{\partial \mathrm{Data}_{\text{render}}}{\partial u}}_{\text{(D.5), rasteriser adjoint}}\Big).
  \tag{8.1}
  $$

  - **Physics path** $u\to F\to W\to E_{\text{mat}}$: linear $F(u)$ (B.1) → PK stress (C.3) →
    edge forces (E.2). $O(Nk)$, sparse, hand-verifiable.
  - **3-D path** $u\to y\to$ Chamfer: stop-grad argmin (Prop 4), $O(N)$ memory.
  - **Render path** $u\to y,\Sigma\to$ image: rasteriser adjoint (D.5); $O(N + \text{pixels})$ per view.

  ---

  ## 9. Architecture / data flow & cost

  | phase | device | work | cost |
  |---|---|---|---|
  | precompute | CPU (scipy) | kNN, $M_i^{-1}$, edges, $V_i$, OT warm start | once, $O(Nk + N\log N)$ |
  | per-iteration | GPU (torch) | $F$ (index_add) → $W,P$ → $E_{\text{mat}}$; $y\to$ Data; backward; Adam | $O(Nk)$ + (Chamfer $O(N)$ \| render $O(\sum_v\text{px})$) |
  | final render | GPU (rasteriser) | $\Sigma=\sigma_0^2FF^\top$ → splat with colour, stray-cull | $O(N+\text{px})$ |

  Fidelity knob is the **particle count $N$** (the method is grid-free): higher $N$ ⇒ denser thin
  features and crisper splats (heroes use $N=2.5\times10^5$). Annealing length and $w_{\text{data}}$
  schedule trade reach vs material-shaping.

  ---

  ## 10. Equation → code map

  | eq | quantity | code |
  |---|---|---|
  | (A.1) | $M_i,\;M_i^{-1}$ | `morph_material_neo.py:morph` (`index_add(DX⊗DX)`, `inv`) |
  | (B.1) | $F_i=I+B_iM_i^{-1}$ | `Ffield(u)` |
  | (C.1)(C.2) | neo-Hookean $W$, $\mu,\lambda$ | `morph_material_neo.py:morph` |
  | (C.3) | $P=\partial W/\partial F$ | autograd of `W.mean()` |
  | (C.4) | anisotropy | `aniso` branch in `morph` |
  | (D.1) | symmetric Chamfer | `chamfer(p,q)` |
  | (D.2)–(D.5) | differentiable render, silhouette | `morph_render_driven.py` (`make_views`,`render_sil`,`selfcheck`) |
  | (E.1) | total objective + anneal | `morph` loss + $w_{\text{data}}$ schedule |
  | (F.1) | $\Sigma=\sigma_0^2FF^\top$, coherence | `cov_from_F`,`render_3dgs`; `color_coh`,`nbr_preserve` |

  *Validated baselines using this method:* Pareto dominance (`morph_pareto.py`), OT head-to-head on
  hard checker (`morph_headtohead.py`), loss-term
  ablation (`morph_ablation.py`), material-control (`morph_material_neo.py`).

  ---

  ## 11. Central claim — theorem, caveats, and `TexCoherence-Bench`

  The paper's load-bearing claim, stated as a theorem (upgrading Prop 6) and **measured**, not
  narrated. We adopt the standard injectivity precedents: Lipman, *Bounded Distortion Mapping
  Spaces* (2012); Aigerman & Lipman, *Injective and Bounded-Distortion Mappings in 3D* (2013);
  Smith et al., *Stable Neo-Hookean Flesh Simulation* (2018) (the origin of our $W$ and its
  $\ln J\!\to\!-\infty$ barrier); Schüller et al., *Locally Injective Mappings* (2013).

  **Theorem 1 (injectivity $\Rightarrow$ local texture coherence).**
  *Let $\varphi=\mathrm{id}+u$ with $F(x)=I+\nabla u(x)$ the kNN-MLS deformation gradient (B.1).
  Suppose the morph minimises the neo-Hookean energy (C.1) under a step that keeps
  $\det F(x_i)>0$ for every particle $i$ (§ guarded step / axis c). Then:*
  1. *each $F_i$ is orientation-preserving ($\det F_i>0$) and invertible (Prop 3);*
  2. *on every MLS neighbourhood, $\varphi$ is locally injective, and the colour field carried by
    $\Sigma_i=\sigma_0^2F_iF_i^\top$ is a continuous, non-folding reparametrisation — no two
    source texels within a neighbourhood map to overlapping image-space support;*
  3. *hence neighbourhood-preservation $\mathrm{NP}\to1$ (eq. F.1) and the texture is advected and
    stretched, never remixed.*

  **Three caveats (stated, not hidden — a theory reviewer will demand them).**
  - **(L) Local, not global.** $\det F_i>0\ \forall i$ does **not** forbid two *distant* particles
    from colliding (global self-overlap). Global injectivity additionally needs a boundary
    bijection (Aigerman–Lipman 2013) or continuous-collision detection (IPC, Li et al. 2020; IDP,
    Fang et al. 2021). **We claim local injectivity; global overlap is handled empirically.**
  - **(D) Discrete sampling.** $\det F>0$ at particles $\neq \det F>0$ on the continuum between
    them. The guarantee is at the sampled set unless the MLS interpolant is separately bounded.
  - **(O) Map vs. optimum.** The *soft* $-\mu\ln J$ barrier does not by itself keep iterates
    feasible; clause 1 is conditional on the guarded step (axis c) actually maintaining
    $\det F>0$. Without it an Adam step can cross $J=0$. This is why the guarded step is
    load-bearing for the theorem, not decorative.

  **Contrast (the failure we beat).** Optimal transport / auction is measure-preserving but
  **non-injective**: neighbouring source particles can map to far-apart targets, so $\mathrm{NP}$
  collapses and the texture scrambles. This is a *provable* failure mode (no $\det F>0$ control),
  which `TexCoherence-Bench` measures directly.

  **`TexCoherence-Bench` (the central-claim benchmark).** Over shape pairs
  {sphere→bunny, →armadillo, →spot}, methods {**ours**, auction-OT, sliced-OT flow,
  Laplacian-smoothed flow, ARAP-energy}, report at **matched reach**:

  | metric | definition | injective ideal |
  |---|---|---|
  | $\min_i\det F_i$ | smallest per-particle Jacobian | $>0$ |
  | flip-fraction | $\#\{i:\det F_i\le0\}/N$ | $0$ |
  | $\mathrm{NP}@k$ | mean fraction of source $k$-NN that stay $k$-NN | $\to1$ |
  | cell-scramble | fraction of final $k$-NN pairs with **different** checker-cell labels | $\to$ source value |
  | reach | symmetric Chamfer to target | low |

  The headline figure is the **reach vs $\mathrm{NP}$ Pareto front** (we dominate); the headline
  table is the per-pair metric grid. *Code:* `morph_bench.py`.


---

## 12. Interactive 3D viewer (planned)

Offline figure strips under-sell a morph whose whole point is that the texture *rides F* in 3D.
An **interactive 3DGS viewer** is part of the deliverable: orbit the morph, scrub the timeline,
toggle what the colour encodes, and compare methods side-by-side. It makes Theorem 1 *legible* —
you watch the checker stretch-not-scramble, and watch OT shred it at the same reach.

**(1) Export — morph frames -> standard 3DGS `.ply`.** Every morph frame is already a set of
coloured, F-deformed Gaussians; serialise them in the canonical 3DGS PLY layout so any existing
viewer loads them with zero custom code:

| field | value | derivation |
|---|---|---|
| `x,y,z` | $y_i=x_i+u_i$ | deformed position |
| `f_dc_{0,1,2}` | SH DC from RGB | $f_{dc}=(c_i-0.5)/C_0,\ C_0=0.2820947918$ |
| `opacity` | logit$(o_i)$ | inverse-sigmoid of splat opacity |
| `scale_{0,1,2}` | $\log s_i$ | $s_i$ = sqrt-eigenvalues of $\Sigma_i=\sigma_0^2F_iF_i^\top$ (`decompose_cov`) |
| `rot_{0..3}` | quaternion (wxyz) | eigenvector frame of $\Sigma_i$ |

Frame `t` -> `frame_{t}.ply`; a morph is the ordered sequence. *Code (to add):*
`export_morph_ply.py`, reusing `cov_from_F` + `decompose_cov` (the SAME $\Sigma=\sigma_0^2FF^\top$
the rasteriser uses) and `RGB2SH`.

**(2) Viewer — web gsplat + timeline.** Target a browser viewer (small `three.js` +
`gsplat`/`antimatter15-splat`, or SuperSplat) so it runs anywhere with no CUDA:
- **orbit / zoom** a single frame (the core "texture rides F" inspection);
- **timeline scrubber** $t\in[0,1]$ hot-swapping `frame_{t}.ply` (a continuous morph, not 6 stills);
- **colour-channel toggle** — (i) texture, (ii) **det F heat-map** (incompressible stays uniform),
  (iii) **flip map** (any $\det F\le0$ lights red = injectivity violation), (iv) NP/cell-scramble;
- **A/B split-screen** — ours vs auction-OT vs ARAP on the *same* scrub: scramble-vs-coherence live.

**(3) Acceptance (what it must reveal).** (a) the checker stretching coherently under ours and
shredding under OT at matched reach; (b) det F staying $>0$ everywhere for ours (Thm 1 / axis c)
and the genus-preserving "hole-not-punched" behaviour of axis (b); (c) the material spectrum
(stiff / soft / incompressible) on one scrub.

*Status:* exporter + static web viewer (load frame, orbit, channel toggle) first, then timeline +
A/B split. Visualization milestone, scheduled after axes (a)/(b).


### 12.4 Viewer — metric tracking, gradient flow, parameters (required)

The viewer is not a passive splat player; it is an **instrument** for the morph. Beyond orbit +
timeline (12.2) it must expose, synced to the scrubber $t$:

- **Per-frame metric tracking.** A live panel plotting, vs $t$: $\min_i\det F$, flip-fraction,
  $\mathrm{NP}@k$, cell-scramble, reach, and total energy $L(u)$ / $E_{\text{mat}}$ / data. The
  current $t$ is a moving cursor; any $\det F\le0$ frame is flagged red on the timeline itself.
- **Gradient-flow field.** Per-particle arrows / streamlines of the **descent direction**
  $-\nabla_{u_i}L = f_i^{\text{elastic}} (\text{E.2}) + w\,f_i^{\text{data}}$ — i.e. the physical
  force driving the morph at that instant — coloured by magnitude. Toggle elastic-only vs
  data-only vs total, to *see* physics vs supervision pulling the body. (This is what makes the
  "render guides / physics governs" split visible.)
- **All parameters, live.** A panel for $E,\nu$, anisotropy $(k_a,a)$, $w_{\text{data}}(t)$
  schedule, $\sigma_0$, $k$, guarded-step $\alpha$ / barrier $\varepsilon$, frame count — read out
  per run, and (stretch) re-runnable from the viewer.
- **3DGS in-viewer.** The frames ARE 3DGS PLYs (12.1); the viewer renders them as real Gaussians
  (web gsplat), not points.

### 12.5 Acceptance: the motion must be PHYSICALLY PLAUSIBLE

A morph is judged by its **motion**, not a final still. The trajectory must read as a continuous
elastic deformation — the sphere *flows* into the target — **not** the
scatter-then-reconverge that a naive Chamfer fit produces (warm-start teleports particles, then
optimisation tidies up; this is exactly the regime that yields the benchmark's flips/scramble).
Acceptance criteria, judged on a **GIF / video** pulled locally (static frames hide this):

1. **Monotone, non-scattering** — particle paths are short and coherent; no global teleport then
   collapse. Quantify by path-length / straightness and frame-to-frame $\Delta u$ smoothness.
2. **Feasible throughout** — $\det F>0$ at *every* frame (guarded step / axis c), so no frame
   tears; the timeline shows zero red.
3. **Elastic look** — intermediate frames are valid elastic states (quasi-static continuation
   from rest, ramping $w_{\text{data}}$), so the body bulges/stretches like material, not like a
   point cloud annealing.

*Deliverable:* `export_morph_ply.py` (frames → PLY) **and** `render_morph_gif.py` (trajectory →
GIF/MP4, pulled to local for judging) come before the interactive web viewer. The trajectory
quality (1–3) is a prerequisite for the viewer to show anything worth scrubbing.


---

## 13. Guarded-step continuation morph — the working method + physical-motion result

§11 caveat (O) predicted, and `morph_bench.py` empirically confirmed, that a *soft* barrier with
an unguarded optimiser does NOT stay injective: the naive neo-Hookean morph (Adam, J clamped)
produced **4.7–37 % flipped particles** (det F<0), worse coherence than OT, and — driven by a
sliced-OT warm-start teleport — a *scatter-then-reconverge* motion that is not physically
plausible. The fix is the guarded step (axis c), now implemented and validated.

**Method (`morph_physical_gif.py:guarded_morph`).** Start from REST (u=0, the sphere) — no
warm-start jump. Ramp the data weight $w_{\text{data}}$ low→high (quasi-static continuation). At
every sub-step: take the Adam step, then **guarded line search** — backtrack the step
($\alpha\!\leftarrow\!\alpha/2$, up to 25×, reject if needed) until $\det F(x_i)>\epsilon$ for
ALL particles. Energy = neo-Hookean (C.1) + symmetric Chamfer + a mild Dirichlet on $u$
(de-spray) ; a coverage down-weight and an F-stretch cull clean thin tips at render. Because the
guard keeps $J>\epsilon$ at every accepted state, **every frame is feasible** and the trajectory
is a continuous elastic deformation (Theorem 1 holds along the whole path, not just at the end).

**Result — feasible, physically-plausible morphs from a sphere (n=60k, 120 frames).**

| target | reach (fit) | max flip % | min det F (whole trajectory) | quality |
|---|---|---|---|---|
| heart | 0.061 | **0.00** | 0.215 | clean (paper-grade) |
| spot (cow) | 0.058 | **0.00** | 0.226 | recognizable |
| armadillo | 0.058 | **0.00** | 0.138 | recognizable |
| bob (C-shape) | 0.066 | **0.00** | 0.046 | deep concavity by bending (genus-0, injective) |
| bunny | 0.095 | **0.00** | 0.030 | recognizable (minor ear residue) |
| dragon / teapot / car | 0.07–0.09 | 0.00 | >0 | feasible but thin features under-resolve at n=60k |

Every morph has **flip-fraction 0 % and min det F > 0 across all 120 frames** — the guarded step
delivers the injectivity Theorem 1 assumes (contrast the naive 4.7–37 %). The motion is a
continuous deformation from rest (no scatter); the checker rides $F$ throughout. Gallery:
`output/figures/GALLERY_physical_morphs.png`; per-target GIFs `output/figures/pg_*.gif`;
acceptance criteria §12.5 (1)(2) met (monotone, feasible-throughout); thin-feature spray is the
remaining polish item (chunky/medium targets are clean; very thin features need higher $n$).

*This supersedes the naive-method rows of `morph_bench.py` for "ours": the benchmark's negative
result motivated the guarded step, which then passes.* `morph_physical_gif.py`.


---

## 14. Smooth-surface rendering in PURE 3D Gaussian Splatting (surfel-GS + target-uniform)

User feedback: the volume-point splat render looked like a *scattered cloud*, not a surface; and
switching to a mesh would abandon Gaussian splatting. The fix keeps GS and yields a smooth shaded
surface — it is the 2DGS / SuGaR family of surface-from-Gaussians.

**(1) Flat surfels (surface-aligned Gaussians).** Each Gaussian is a flat disk tangent to the
surface, with the normal `n`:
  Σ_0 = σ_t² (I − n nᵀ) + σ_n² n nᵀ ,   σ_n ≪ σ_t .
Riding the deformation, Σ = F Σ_0 Fᵀ. Dense overlapping flat disks compose an opaque, continuous
surface. Lambertian shading is BAKED into the per-Gaussian colour via the deformed normal
n' = F⁻ᵀn (normalised): c ← c · (a + (1−a)·max(0, n'·L)). Still the same diff_gauss rasteriser —
no mesh, no new renderer. `scripts/morph_surfel_gs.py`.

**(2) Floating-Gaussian removal (3-stage static cull).** After the guarded injective solve, drop
surfels that do not lie on the surface: (i) distance — NN-to-target > k·median; (ii) stretch —
max singular value of F* > τ; (iii) **isolation** — final kNN spacing > ι·median (floating
strands are sparse in the final configuration). This cleans the off-surface splats.

**(3) Smooth THIN FEATURES — the target-uniform fix (key).** A *uniform source sphere* maps too
few, over-stretched surfels onto thin target parts (bunny ears, armadillo limbs) → sparse /
floating there. Instead **sample the TARGET surface at uniform density** (area-uniform; ears and
limbs included), and for each target sample find its **sphere preimage** through the injective
elastic correspondence (solve sphere→target once; invert by kNN in the morphed sphere). Morph
sphere-preimage → target. Now every part of the target — thin or thick — has uniform surfel
density, so it renders as a smooth surface, and there are NO floaters by construction (every
surfel lies on the target surface). Normals interpolate sphere→target; texture rides via the
Lagrangian colour on the sphere preimage. `scripts/morph_surfel_uniform.py`.

**(4) Physics IN the motion — even-paced quasi-static continuation (the headline deliverable).**
The smooth surface above can be animated two ways. The weaker way *interpolates* the correspondence
(smoothstep) — intermediate frames are geometric blends, not physical states. The **headline** does
the stronger thing: it makes **every intermediate frame an elastic equilibrium**. We solve the
injective correspondence once (guarded neo-Hookean, det F>0), then run an **even-paced
quasi-static loading** on a coarse sphere — as the load fraction α rises uniformly 0→1, minimise
$W(F)+w\lVert (V_0+u)-\mathrm{lerp}(V_0,P,\alpha)\rVert^2$ with the **guarded step** (§13), warm-started
from the previous α. Each recorded state is thus a physically-valid equilibrium (det F>0), and the
even α schedule removes the front-loaded jump of the raw trajectory. The dense target-uniform
surfels are then **embedded** in this coarse elastic continuum (embedded deformation: each surfel
rides the coarse motion via fixed inverse-distance weights), with the per-surfel
embedding-reconstruction residual ramped in linearly so the **final frame lands exactly on the
crisp target surface** (sharp thin features) while the motion stays the physical trajectory. This is
what earns the noun "morph**ing**": the physics governs *the motion you watch*, not merely the
endpoint correspondence. `scripts/morph_surfel_physical.py`.

> **Status (honest).** The continuation **code is hardened and one-shot-ready** (compiles; endpoint
> residual added); the physics-motion GIFs are produced by a single GPU run of
> `morph_surfel_physical.py` per target — **pending GPU access** (cluster currently unreachable). The
> existing `un_*.gif` / `MONTAGE_*.gif` in `output/figures/` are the **correspondence-interpolation
> preview** (smoothstep, subsection 3); they will be replaced by the physics-continuation versions
> on the next GPU run. The two differ in the *intermediate frames* only — both share the same
> injective correspondence and same crisp endpoint.

**Result.** Smooth shaded surfaces for all targets *including thin features* — bunny ears and
armadillo limbs render as solid smooth tubes; heart/spot/bob are perfectly clean; bob's hole
forms by bending. Pure 3DGS throughout. `output/figures/GALLERY_surface_morphs.png`,
`MONTAGE_surface_morphs.gif`. (Open3D shaded-mesh renders exist only as a *reference* — not used,
to keep the Gaussian-splatting representation.)

**Honest note.** The checker cell size grows on strongly-compressed thin parts (the source-sphere
preimage is compressed there) — a texture-parametrisation artefact, not a surface one. The
correspondence solve uses the volume-based guarded morph (Theorem 1, det F>0); the surfel render
is a post-process on top of that injective correspondence.
