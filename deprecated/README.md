# deprecated/ — VBD-MPM quasi-static arm (frozen 2026-09-01)

Retired from the main line by the coherence/positioning decision (docs/rationale.md §4,
experiments.md): the dynamic family (elasto + render adjoint) is the deliverable — it
modifies the physics literally (control stress, per-particle Lamé, Fp, v_T) and passes
every gate at full scale; the quasi-static arm either crawls (honest material memory,
η=0.5/E=2e3 — the equilibrium itself resists) or degenerates toward Sobolev-regularised
registration (η=1.0/E=300 — fast, maximally coherent displacement, but no material
memory).

What this code contributed before retirement (kept in the papers' analysis):
- the memory-vs-coherence trade-off measurement (`scripts/coherence_check.py`, still live):
  dynamic arm nbr_overlap 0.387 @ move 0.614 (plastic shear mixes neighbourhoods, field
  smooth 0.070); vbd η=0.5: 0.962 @ 0.106; η=1.0 row in output/coherence.json.
- solver findings: fringe-node poisoning (weight-thresholded active set + relative diag
  floor + per-node trust radius), 8-color CIC decoupling, exact quadratic line search,
  and the equilibrium-scale diagnosis (per-term exit gradients 67/51/26 cancelling).
- differentiability route for a future material channel: IFT adjoint validated in
  `scripts/probe_gs_differentiability.py` (kept in scripts/).

Nothing here is imported by the live tree; `test_vbd.py` is out of pytest's testpaths.
To revive: `git mv` back and re-add the vbd_* fields to PipelineConfig (removed).
