"""Plasticity: OT rest-state migration (Sinkhorn barycentric + sliced-OT balanced)."""
from .sinkhorn import sinkhorn_displacement, displacement_jacobian, update_fp
from .sliced_ot import sliced_ot_displacement
from .assignment import balanced_assignment, assignment_displacement

__all__ = ["sinkhorn_displacement", "displacement_jacobian", "update_fp",
           "sliced_ot_displacement", "balanced_assignment", "assignment_displacement"]
