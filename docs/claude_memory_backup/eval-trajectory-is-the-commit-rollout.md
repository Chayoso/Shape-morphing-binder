---
name: eval-trajectory-is-the-commit-rollout
description: optimize_window's persistent no-grad tr_eval (optimizer.py ~L343) is the line-search AND commit rollout — any per-particle rollout state (eta, pin, bonds, layer) must be passed to it as well as to RolloutSpec, or it never reaches the delivered frames
metadata:
  type: project
---

In `physmorph/pipeline/optimizer.py::optimize_window` the adjoint rollouts take the `RolloutSpec`, but the
persistent no-grad `Trajectory` `tr_eval` (built directly ~L343, CUDA-graph captured) is what runs the line
search, the warm-start comparison and the COMMIT rollout whose per-step frames are archived. 2026-09-24:
`settle_eta` and the first `settle_pin` only reached the RolloutSpec, so the delivered frames were inviscid /
unpinned while the gradient was not (g41v's reading withdrawn; the pin's 3k smoke showed 0 still particles).
Fixed by `layer=layer, eta=spec.eta, pin=spec.pin` on tr_eval.

**Why:** the two rollouts must describe the same system; a state added to one only silently splits the
optimiser from the delivered animation.
**How to apply:** when adding any per-particle field to `RolloutSpec`/`Trajectory`, add it to `tr_eval` too
and verify from the archived frames (`scratch/pin_probe.py`-style), not from the optimiser's logs. See
[[until-zero-oscillation]], [[no-parameter-only-fixes]].
