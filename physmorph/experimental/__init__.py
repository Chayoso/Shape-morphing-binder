"""Quarantined v1 loops — NOT part of the v2 blessed path (docs/pipeline_v2.md §4).

Kept because the ablation figures come from here:
  morph.py            greedy per-frame dFc + fixed-lambda render term (shown inert) + the
                      non-physical stabiliser stack (cohesion / max_move / pull_outliers).
  morph_physical.py   displacement-space OT/render guidance (render gradient never passes
                      through the simulator — Type B coupling).
  style_transfer.py   displacement-space style loop on top of morph_physical.

Nothing here may be cited for a v2 claim; nothing here is maintained for new features.
"""
