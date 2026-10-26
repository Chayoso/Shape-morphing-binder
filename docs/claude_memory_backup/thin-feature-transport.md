---
name: thin-feature-transport
description: "2026-09-15 \"scatter then return\" (ear spray) dossier CLOSED — cause = D_vol descent filling ear cells with a stretched stream below the loss cell; not volume/render/APIC/fracture; every window-end regulariser and the support gate falsified; levers = control basis + --ppc 8 / density units"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-15T20:42:46.267Z
---

User question (2026-09-15, 40k GIFs): ears fill with a spray that later thickens; strays at the
feet; "volume or something else?". Dossier `docs/thin_feature_transport.md` (§5a–§5e results,
§4 pre-registration with verdicts), probes `scripts/probes/gate_probe.py` + hyde06
`/tmp/scatter_probe2.py` ("sparse" = 8-NN mean > 2 target spacings ≈ 0.26 wu).

Closed findings (20k dx 0.5 flagship recipe, one seed, code 13b9db7):
- Spray peak 0.58 at commit ~5 → end 0.38. Same peak with render OFF (0.560, `--lambda_auto 0`),
  with the APIC affine transfer halved (support gate, 0.569), without the cleanup pulls (batch k),
  with `w_coh`/`w_bond`/`vol_frontier` (0.58). Vanguard is NOT grid-isolated (3³ count 0.31 of
  interior at 37 ppc; 0.4 at 40k ppc 8) — "numerical fracture" reading struck out. Not volume.
- Cause: the D_vol mass-matching descent fills ear cells with the nearest particles at ~2×
  spacing because the loss cannot see spacing below its 0.5 wu cell and a per-particle control
  can move single particles. The render loss only amplifies the fill (+0.9 pt silIoU, more mass,
  higher end sparsity).
- Levers: control basis (any 6–36³ → peak 0.50; end/delivered monotone in resolution, 36³ best:
  end 0.24, 0.195/0.194, chamfer 0.1601); `--ppc 8` (peak 0.44–0.48, END strays 0.01–0.2 % = the
  mass-ejection fix); density units (peak 0.36 but under-fills at 20k). `w_bond 100` HARMFUL at
  40k ppc 8. 40k ppc 8 + 24³ in LEGACY units froze at commit 46 (line-search exhaustion).
- Implemented, kept opt-in/default off: `w_coh`, `w_bond`/`bond_s0`, `vol_frontier`, support-gated
  APIC (`--gate_lo/--gate_hi`, `MPMParams.gate_*`, method.md eq 22, `tests/test_support_gate.py`).
- 40k `--ppc 8` + density + recipe + 24³ basis (o1): froze at commit 135, chamfer 0.1175 vs 0.1142,
  spray within noise → the basis does NOT add at fine dx; flagship candidate stays per-particle.
- 20k density units + `--loss_res 128` (0.25 wu loss cell, per-particle; q1): peak 0.493, end 0.300,
  fill 0.176, chamfer 0.1553 (−2.9 %) → "what the loss can see" is a lever equal to the basis.
  20k density/128 + 36³ basis (q3) and 40k 36³ (q2) do not add: same ~0.49 (20k) / 0.43 (40k) floor.
- Ear–head "connection" (user, 40k frame 1746): PROJECTION — the wedge is the far ear edge-on in the
  target's own az-0.6 view; morph-only pixels 1 % (≤0.2 wu fringe), 0 particles > 0.25 wu from the
  target, notch preserved at az 1.4. Real residual = ear UNDER-fill (1.2 % ear points uncovered at
  0.15 wu). Probes: scripts/probes/{web_probe,ear_views,cover_diff}.py. Next arm: loss cell < ear
  sheet thickness (density units, --loss_res > 149 at 40k).

**Why:** the user wants the spray fixed under "rendering controls physics"; window-end
regularisers cannot undo a stream already stretched inside the window, and the two mechanistic
hypotheses (render loss, APIC fling) are measured false.
**How to apply:** do not propose window-end regularisers or transfer gates for the spray again;
recommend the basis + discretisation-contract recipe; any further gain must change what the
loss can see (finer loss cell in density units) or what the control can move (basis). See
[[render-controls-physics-contract]], [[hyde06-ops-etiquette]].
