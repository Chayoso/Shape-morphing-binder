# REFUTE round — local-global surface channel design

You are an adversarial reviewer. Your ONLY job is to try to BREAK the design in
docs/local_global_design.md before it is implemented. Do not summarize it; attack it.
Read it together with docs/root_analysis.md, docs/oscillation.md, docs/floaters.md,
physmorph/pipeline/runner.py, physmorph/pipeline/optimizer.py,
physmorph/plasticity/assimilation.py, physmorph/pipeline/surface_local.py,
physmorph/pipeline/gauss_loss.py, physmorph/render/children.py.

The design in one line: global MPM windows stay byte-identical; per accepted commit a
LOCAL channel (Tier D: bounded tangent-plane sub-Nyquist dressing DOFs on the massless
render children, optimized directly against the gauss image loss; Tier R: the retired
surface-band solver revived with the gauss objective, returning a VIRTUAL displacement
field whose symmetric strain is absorbed as F_virt = exp(clamp(sym grad u, tau)) * F
through the existing assimilate_elastic contract) routes high-frequency render
corrections around the low-pass MPM adjoint.

Attack surfaces we specifically need checked with file:line evidence:
1. Tier R soundness: is exp(clamp(sym grad u)) * F really absorbed EXACTLY by the
   existing isochoric/SV-band projections for every branch of assimilate_elastic
   (isochoric on/off, grow on/off)? Does the polar-relaxation step commute with the
   A-premultiply, or does the demand leak into rotation?
2. Ratchet: tau = median accepted elastic log-stretch of the just-committed window —
   construct a scenario where this self-calibration ratchets volume or surface area
   monotonically (the fill-v2 30:1 late-dominance and volume-ratchet history are the
   precedents). Does Fp_predemand rollback actually cover EVERY outer-reject path,
   including null commits and c2f rebuild commits?
3. Objective mismatch: the lg_sweeps guard was added because a local quadratic energy
   undid accepted W1/fill steps and assimilation ratcheted the regression. The design
   claims re-arbitration by the next window + outer gate replaces that guard. Show a
   concrete sequence where the demand and the W1/nn cleanup terms fight per-commit and
   the outer merit (which contains dt) oscillates or deadlocks (rejection loops are an
   absorbing state precedent: s4 a62-a69).
4. Tier D honesty: dressing DOFs are excluded from all gate metrics (raw-state), but
   the GAUSS LOSS ITSELF is a gate track (rend_track) when use_gauss_loss is on — if
   dressing lowers the gauss residual, does the freeze/anneal accounting see phantom
   improvement? Which exact rec[] fields does dressing touch?
5. The frequency-gap claim: the design says demand crosses the dynamics ONCE forward.
   But the next window's rollout starts from Fp-modified state and the OPTIMIZER will
   respond to the changed elastic energy — can the global adjoint UNDO the demand
   (dFc anti-correcting), wasting both channels? What telemetry would catch it?
6. Cost: per-commit KDTree/PCA + band solve + dressing iterations at N=40k, T=20 —
   estimate wall-clock per commit and whether the 300-commit paced budget survives.
7. Any invariant hole: finiteness, det>1e-4 margin, isochoric det Fp=1, SV band,
   commit-rollout revalidation, warm-start safeguard interaction with a changed Fp.

Output format: numbered findings, each with severity (BLOCKER/MAJOR/MINOR), the
file:line or design-section anchor, a concrete failure scenario, and — only if one
exists — the minimal repair. End with a verdict: IMPLEMENT / IMPLEMENT-WITH-REPAIRS /
REDESIGN.
