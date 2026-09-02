# Problem dossier 2 — Near-optimum oscillation (optima 근처의 흔들림)

Status: **root causes closed; final validation = a converged-budget flagship run
whose held tail is visually still.** Done-criteria: freeze fires within budget, the
held tail passes G3 (jitter_rel & drift < 0.3% diag), and the replay tail shows no
visible motion.

## 1. Phenomenon

As the morph approaches the target, the body keeps visibly moving — micro-creep and
"breathing" — instead of coming to rest. Reported repeatedly from the live viewer
("여전히 optima로 갈수록 진동이 보여").

## 2. What it turned out to be (measured decomposition)

The wobble was FOUR separate mechanisms, uncovered in sequence. None of them was
"noise":

| # | mechanism | forensic evidence | fix | status |
|---|---|---|---|---|
| 1 | **λ runaway** — the render weight diverged once D_render saturated, injecting huge late-run steps | λ 1.1e3→1.77e5 live; mid-window inversion in tow | λ cap (5e3) + per-window λ freeze + EMA | closed (pre-session era) |
| 2 | **volumetric plastic ratchet** — assimilation baked volume each commit; internal rearrangement never stopped because volume kept drifting | \|J−1\|>0.3: 0→34–41% monotone across the run; 32% of particles still changing volume at the tail | isochoric assimilation (det Fp ≡ 1, log-space band∩det projection) | closed |
| 3 | **permanent volumetric spring** — the isochoric fix's cost: ALL volume strain stayed elastic forever, so the body breathed against data terms and armed single-step inversions | restoring dψ/dJ = −3889 at J=0.9; elastic tail \|dJ\| drift ↑ after iso | **sKL volume prior** w_jvol·(J−1)·log J through the adjoint (prevention-in-energy, not rejection) | closed — \|J−1\|>0.3 → 0.0%, J∈[0.92,1.09], tail move halved (0.009→0.003–0.004) |
| 4 | **unfinished descent misread as oscillation** — at production budgets the optimizer was still genuinely improving every track | at 200c: d_vol still −9…−27% per 40 commits; freeze correctly withheld; **400c probe: true convergence at commit 282, then 118 commits held at jitter 7e-5** | run the flagship past its resting point (≥300c); freeze machinery already correct | closed |

Also ACQUITTED by forensics: per-window gate flicker (0.7%/5 commits — not the
oscillation source), and the three-track freeze (it withheld freezing because
improvement was real, not because tracks were noisy — after the gated-track fix).

Supporting structural fixes along the way: full-state promotion + λ-free freeze
tracks (pre-session), the trajectory-min-det acceptance guard with the effective
deformation F+dFc (single-step inversions had slipped through the terminal-only
check), null-commit on line-search exhaustion (a truncation bug had masqueraded as
convergence), and per-term freeze tracks with the UNGATED d_dt statistic (a dying
gate had masqueraded as progress).

## 3. Related papers (organized by remedy family)

**Energy-side prevention vs step rejection (mechanism 3)**
- Smith, de Goes, Kim — Stable Neo-Hookean (ToG 2018): finite, restoring energies
  through inversion, chosen by differentiable pipelines precisely for gradient
  availability; our fixed corotated λ/2(J−1)² is already SNH-shaped.
- Chen et al. (SGP 2024): inversion-aware line-search filters "may stall
  completely" — the warning that kept us from a divergent barrier.
- Stomakhin et al. — Energetically Consistent Invertible Elasticity (SCA 2012):
  inversion handling belongs in the energy (consistent gradients), not a bolt-on
  projection — the constitutive analog of guard-vs-prior.
- IPC (Li et al., ToG 2020): det-safe step-LENGTH filtering as the principled guard
  upgrade (queued, not yet needed since w_jvol removed the trigger).

**Volume/plasticity treatment (mechanism 2/3)**
- Klar et al. — Drucker-Prager sand (ToG 2016): Case-III trace-preserving flow = the
  canonical isochoric split (validates assim_iso).
- Stomakhin et al. — MPM snow (ToG 2013): plastic volume change is admissible ONLY
  with exp-hardening (their E=1.4e5/ν=0.2 is literally our material); bare η_vol is
  the ratchet on a longer fuse — why we rejected slow volumetric relaxation.
- Tampubolon et al. (ToG 2017): the per-particle log-volume ledger — the published
  "plasticity ate my volume" fix; queued for control-injected volume drift.
- Yanovsky/Leow et al. (UCLA CAM 07-49; TMI 2007): the sKL/log-unbiased Jacobian
  regularizer (J−1)·log J — zero iff J=1, log-symmetric, soft barrier as J→0⁺ — the
  form we adopted as w_jvol.
- Viscoelasticity standard practice (COMSOL theory; Roylance): bulk response is
  taken elastic, relaxation is deviatoric-only — the physics that killed η_vol.

**Optimizer-side stability (mechanisms 1/4)**
- McNamara et al. (2004) adjoint control lineage: per-window fixed objective — a
  drifting per-iteration λ makes "monotone acceptance" meaningless (our per-window
  λ freeze + EMA + cap).
- Anderson acceleration for physics (TOG 2018): the guarded-extrapolation family —
  evaluated and NOT adopted (see §4); kept in the backlog with the line search as
  the guard if a residual settling problem ever re-emerges.

## 4. v1's remedy vs ours — global damping vs cause removal

**v1 (PhysMorph-GS C++/paper)** treated late-run unrest with GLOBAL, symptom-side
smoothing — the "extrapolation-like" controls the user recalls: temporal F smoothing
(s = 0.955 EMA on the deformation gradient), damping-toward-identity on F_p, and
kNN displacement-field smoothing (k=64, 3 iterations). These act like a low-pass
filter on the whole state: they suppress the VISIBLE oscillation everywhere, at the
cost of also suppressing legitimate fine motion, blurring thin-feature detail, and
leaving the underlying driver (volume drift + λ drift) in place — the unrest
returns whenever the filter weakens.

**v2's position**: no global damping term was added at all. Each driver was isolated
by forensics and removed at its source — λ capped and frozen per window, plastic
volume made isochoric, the residual elastic spring neutralised by the sKL volume
prior *through the adjoint* (the optimizer stops COMMANDING volume heterogeneity,
rather than having its commands filtered afterwards), and the remaining motion
proven to be honest unfinished descent by running past the true resting point
(commit 282). The one v1 idea we kept is temporal smoothing INSIDE the simulator's
F update (s=0.955, inherited) — and the stack-review showed even that can hide
effective-deformation inversions, which is why the trajectory guard checks F+dFc.

## 5. Current state & open work

At 400c the flagship freezes at commit 282 and holds still (jitter 7e-5, hole 0.00%).
Production runs now use ≥300c. Remaining validation for THIS dossier: the h17
converged 4-pair batch — confirm every pair freezes in budget and the held tails
pass G3; then a replay-tail visual check. If any pair fails to freeze, the residual
driver gets its own forensic before any new mechanism.
