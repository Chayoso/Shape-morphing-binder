"""Plasticity: OT rest-state migration (v1) + objective commit assimilation (v2)."""
from .sinkhorn import sinkhorn_displacement, displacement_jacobian, update_fp
from .sliced_ot import sliced_ot_displacement
from .assignment import balanced_assignment, assignment_displacement
from .assimilation import assimilate_fp

__all__ = ["sinkhorn_displacement", "displacement_jacobian", "update_fp",
           "sliced_ot_displacement", "balanced_assignment", "assignment_displacement",
           "assimilate_fp"]
